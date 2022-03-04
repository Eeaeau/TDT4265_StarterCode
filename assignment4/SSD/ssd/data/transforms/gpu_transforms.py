import torch
import torchvision


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).float().view(1, len(mean), 1, 1)
        self.std = torch.tensor(std).float().view(1, len(mean), 1, 1)
    
    @torch.no_grad()
    def forward(self, batch):
        self.mean = self.mean.to(batch["image"].device)
        self.std = self.std.to(batch["image"].device)
        batch["image"] = (batch["image"] - self.mean) / self.std
        return batch

class ColorJitter(torchvision.transforms.ColorJitter):

    @torch.no_grad()    
    def forward(self, batch):
        batch["image"] = super().forward(batch["image"])
        return batch
