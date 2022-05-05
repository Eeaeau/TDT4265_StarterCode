from torchvision.io.image import read_image
import matplotlib.pyplot as plt
import torch.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image
from performance_assessment.save_comparison_images import get_config, get_trained_model, get_dataloader


# from ssd.modeling.retinanetOutputWrapper import RetinaNetOutputWrapper

config_path = "configs/tdt4265.py"
# cfg = LazyConfig.load(config_path)


# cfg = get_config(config_path)
# model = get_trained_model(cfg)

# print(model)

# Get your input
img = read_image("data/tdt4265_2022/images/train/trip007_glos_Video00000_3.png")
plt.imshow(to_pil_image(img))
plt.show()
