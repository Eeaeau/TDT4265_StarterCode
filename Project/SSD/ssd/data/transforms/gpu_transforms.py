import torch
import torchvision
import torchvision.transforms as T

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

class ColorJitter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.brightness= 0.25
        self.contrast =0.25
        self.saturation =0.25
        self.hue = 0
        self.jitter = T.ColorJitter(brightness=self.brightness, contrast = self.contrast, saturation =self.saturation, hue=self.hue)
    @torch.no_grad()
    def forward(self, batch):
         batch["image"] =self.jitter(batch["image"])
         return batch