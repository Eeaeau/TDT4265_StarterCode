from .tdt4265_init_weights import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)


anchors.aspect_ratios = [ [1, 2] , [1,10], [10, 1], [2, 1], [20, 1], [1, 20] ]