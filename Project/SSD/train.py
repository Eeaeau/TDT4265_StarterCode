import sys
assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import time
import click
import torch
import pprint
import tops
import tqdm
from pathlib import Path
from ssd.evaluate import evaluate
from ssd import utils
from tops.config import instantiate
from tops import logger, checkpointer
from torch.optim.lr_scheduler import ChainedScheduler
from omegaconf import OmegaConf
torch.backends.cudnn.benchmark = True

def train_epoch(
        model, scaler: torch.cuda.amp.GradScaler,
        optim, dataloader_train, scheduler,
        gpu_transform: torch.nn.Module,
        log_interval: int):
    grad_scale = scaler.get_scale()
    for batch in tqdm.tqdm(dataloader_train, f"Epoch {logger.epoch()}"):
        batch = tops.to_cuda(batch)
        batch["labels"] = batch["labels"].long()
        batch = gpu_transform(batch)

        with torch.cuda.amp.autocast(enabled=tops.AMP()):
            bbox_delta, confs = model(batch["image"])
            loss, to_log = model.loss_func(bbox_delta, confs, batch["boxes"], batch["labels"])
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        if grad_scale == scaler.get_scale():
            scheduler.step()
            if logger.global_step() % log_interval:
                logger.add_scalar("stats/learning_rate", scheduler._schedulers[-1].get_last_lr()[-1])
        else:
            grad_scale = scaler.get_scale()
            logger.add_scalar("amp/grad_scale", scaler.get_scale())
        if logger.global_step() % log_interval == 0:
            to_log = {f"loss/{k}": v.mean().cpu().item() for k, v in to_log.items()}
            logger.add_dict(to_log)
        # torch.cuda.amp skips gradient steps if backward pass produces NaNs/infs.
        # If it happens in the first iteration, scheduler.step() will throw exception
        logger.step()

    return


def print_config(cfg):
    container = OmegaConf.to_container(cfg)
    pp = pprint.PrettyPrinter(indent=2, compact=False)
    print("--------------------Config file below--------------------")
    pp.pprint(container)
    print("--------------------End of config file--------------------")


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--evaluate-only", default=False, is_flag=True, help="Only run evaluation, no training.")
def train(config_path: Path, evaluate_only: bool):
    logger.logger.DEFAULT_SCALAR_LEVEL = logger.logger.DEBUG
    cfg = utils.load_config(config_path)
    print_config(cfg)

    tops.init(cfg.output_dir)
    tops.set_AMP(cfg.train.amp)
    tops.set_seed(cfg.train.seed)
    dataloader_train = instantiate(cfg.data_train.dataloader)
    dataloader_val = instantiate(cfg.data_val.dataloader)
    cocoGt = dataloader_val.dataset.get_annotations_as_coco()
    model = tops.to_cuda(instantiate(cfg.model))
    optimizer = instantiate(cfg.optimizer, params=utils.tencent_trick(model))
    scheduler = ChainedScheduler(instantiate(list(cfg.schedulers.values()), optimizer=optimizer))
    checkpointer.register_models(
        dict(model=model, optimizer=optimizer, scheduler=scheduler))
    total_time = 0
    if checkpointer.has_checkpoint():
        train_state = checkpointer.load_registered_models(load_best=False)
        total_time = train_state["total_time"]
        logger.log(f"Resuming train from: epoch: {logger.epoch()}, global step: {logger.global_step()}")

    gpu_transform_val = instantiate(cfg.data_val.gpu_transform)
    gpu_transform_train = instantiate(cfg.data_train.gpu_transform)
    evaluation_fn = functools.partial(
        evaluate,
        model=model,
        dataloader=dataloader_val,
        cocoGt=cocoGt,
        gpu_transform=gpu_transform_val,
        label_map=cfg.label_map
    )
    if evaluate_only:
        evaluation_fn()
        exit()
    scaler = torch.cuda.amp.GradScaler(enabled=tops.AMP())
    dummy_input = tops.to_cuda(torch.randn(1, cfg.train.image_channels, *cfg.train.imshape))
    tops.print_module_summary(model, (dummy_input,))
    start_epoch = logger.epoch()
    for epoch in range(start_epoch, cfg.train.epochs):
        start_epoch_time = time.time()
        train_epoch(model, scaler, optimizer, dataloader_train, scheduler, gpu_transform_train, cfg.train.log_interval)
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time
        logger.add_scalar("stats/epoch_time", end_epoch_time)

        eval_stats = evaluation_fn()
        eval_stats = {f"metrics/{key}": val for key, val in eval_stats.items()}
        logger.add_dict(eval_stats, level=logger.logger.INFO)
        train_state = dict(total_time=total_time)
        checkpointer.save_registered_models(train_state)
        logger.step_epoch()
    logger.add_scalar("stats/total_time", total_time)


if __name__ == "__main__":
    train()
