import datasets
import src.config as config
import torch.utils.data
import torchvision.datasets as datasets
from src.data.fakeimagenet.dataloader import image_transforms

data_load_name="fakeimagenet"

def load_data(conf : config.Config) -> torch.utils.data.Dataset:
    """Simple function to load a dataset based on the provided config object.
    """

    # TODO: make this dynamic and add to config
    folder = "/home/slurm/comp597/students/mreil2/fakeimagenet/FakeImageNet/train"
    return datasets.ImageFolder(folder, image_transforms())