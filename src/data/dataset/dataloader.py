# from https://github.com/mila-iqia/milabench/blob/master/benchmate/benchmate/dataloader.py
# Modified by mreil2 to remove dali option
# Modified by mreil2 to costumize args

import argparse
import os

import torch
import torch.cuda.amp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_tensors(batch_size, shapes, device):
    """
    Examples
    --------

    >>> generate_tensors(128, [(3, 224, 224), (1000)], "cuda")
    [Tensor((128, 3, 224, 223)), Tensor(128, 1000)]

    >>> generate_tensors(128, [("x", (3, 224, 224)), ("y", (1000,))], "cuda")
    {"x": Tensor((128, 3, 224, 223))  "y": Tensor(128, 1000)}
    """
    tensors = []
    if len(shapes[0]) == 2:
        tensors = dict()

    for kshape in shapes:
        if len(kshape) == 2:
            key, shape = kshape
            tensors[key] = torch.randn((batch_size, *shape), device=device)
        else:
            tensors.append(torch.randn((batch_size, *kshape), device=device))

    return tensors


def generate_tensor_classification(original_model, batch_size, in_shape, device):
    model = original_model.to(device=device)
    inp = torch.randn((batch_size, *in_shape), device=device)
    out = torch.rand_like(model(inp))
    return inp, out


class FakeInMemoryDataset:
    def __init__(self, producer, batch_size, batch_count):
        self.data = [producer(i) for i in range(batch_size * batch_count)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x, y = self.data[item]
        return {"input": x, "target": y}


class FakeImageClassification(FakeInMemoryDataset):
    def __init__(self, shape, batch_size, batch_count):
        def producer(i):
            return (torch.randn(shape), i % 1000)

        super().__init__(producer, batch_size, batch_count)


class DictImageFolder(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.dataset = datasets.ImageFolder(folder, transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {
            "input": image,
            "target": label,
        }

class SyntheticData:
    def __init__(self, tensors, n, fixed_batch):
        self.n = n
        self.tensors = tensors
        self.fixed_batch = fixed_batch

    def __iter__(self):
        if self.fixed_batch:
            for _ in range(self.n):
                yield self.tensors

        else:
            for _ in range(self.n):
                yield [torch.rand_like(t) for t in self.tensors]

    def __len__(self):
        return self.n


def pytorch_fakedataset(folder, batch_size, num_workers):
    train = FakeImageClassification((3, 224, 224), batch_size, 60)

    return torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )


def image_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return data_transforms


def pytorch(folder, batch_size, num_workers, distributed=False, epochs=60, rank=None, world_size=None):
    train = DictImageFolder(folder, image_transforms())

    kwargs = {"shuffle": True}
    if distributed:
        kwargs["sampler"] = DistributedSampler(train, rank=rank, num_replicas=world_size)
        kwargs["shuffle"] = False

    # The dataloader needs a warmup sometimes
    # by avoiding to go through too many epochs
    # we reduce the standard deviation
    if False:
        kwargs["sampler"] = torch.utils.data.RandomSampler(
            train, replacement=True, num_samples=len(train) * epochs
        )
        kwargs["shuffle"] = False

    return torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )


def synthetic(model, batch_size, fixed_batch):
    return SyntheticData(
        tensors=generate_tensor_classification(
            model, batch_size, (3, 244, 244), device=device
        ),
        n=1000,
        fixed_batch=fixed_batch,
    )


def synthetic_fixed(*args):
    return synthetic(*args, fixed_batch=True)


def synthetic_random(*args):
    return synthetic(*args, fixed_batch=False)


def data_folder(args):
    if not args.data:
        data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
        if data_directory:
            args.data = os.path.join(data_directory, "FakeImageNet")
    return args.data


def imagenet_dataloader(args, model, rank=0, world_size=1):
    if args.loader == "synthetic_random":
        return synthetic(model=model, batch_size=args.batch_size, fixed_batch=False)

    if args.loader == "synthetic_fixed":
        return synthetic(model=model, batch_size=args.batch_size, fixed_batch=True)

    if args.loader == "pytorch_fakedataset":
        return pytorch_fakedataset(
            None, batch_size=args.batch_size, num_workers=args.num_workers
        )

    folder = os.path.join(data_folder(args), "train")

    return pytorch(folder, args.batch_size, args.num_workers, 
                   distributed=world_size > 1, rank=rank, world_size=world_size)



from dataclasses import dataclass, asdict
from copy import deepcopy

@dataclass
class Dataloader:
    """
    
    Specify the common configuration of the dataloader
    >>> helper = LazyDataLoader(num_workers=8, pin_memory=True, collate_fn)

    Instantiate the loader for each split
    >>> train = helper.dataloader(train, batch_size=128, shuffle=True)

    >>> vali = helper.dataloader(validation, batch_size=1024, shuffle=False)

    """
    @dataclass
    class Arguments:
        batch_size: int = 1
        shuffle: bool = False
        num_workers: int = 0
        pin_memory: bool = False
        drop_last: bool = False
        timeout: float = 0
        prefetch_factor: int = None
        persistent_workers: bool = False
        pin_memory_device: str = ''

    argparse_arguments = Arguments()

    def __init__(self, 
            dataset: 'Dataset',
            sampler: 'Sampler' = None,
            batch_sampler: 'BatchSampler' = None,
            generator=None,
            worker_init_fn=None,
            multiprocessing_context=None,
            collate_fn=None,
            **kwargs
        ):
        self.base_args = {
            "dataset": dataset,
            "sampler": sampler,
            "batch_sampler": batch_sampler,
            "generator": generator,
            "worker_init_fn": worker_init_fn,
            "multiprocessing_context": multiprocessing_context,
            "collate_fn": collate_fn,
            **kwargs
        }

        defaults = asdict(Dataloader.Arguments())
        argparse = asdict(self.argparse_arguments)

        for k in defaults.keys():
            vdef = defaults.get(k)
            vset = argparse.get(k)
            vcode = self.base_args.get(k)

            if vcode is None:
                self.base_args[k] = vset
            else:
                # code set the arguments and it is different from the default
                if vcode != vdef and vset == vdef:
                    self.base_args[k] = vcode

                # code set the arguments but it is the default 
                if vcode == vdef and vset != vdef:
                    self.base_args[k] = vset

                if vcode != vdef and vset != vdef:
                    print(f"ambiguous value for {k} code set {vcode} argparse set {vset} default is {vdef}")
                    self.base_args[k] = vset

    def dataloader(self, **kwargs):
        from torch.utils.data import DataLoader
        args = deepcopy(self.base_args)
        args.update(kwargs)
        return DataLoader(**args)

    def train(self, **kwargs):
        kwargs.setdefault("shuffle", True)
        return self.dataloader(**kwargs)

    def validation(self, **kwargs):
        kwargs.setdefault("shuffle", False)
        return self.dataloader(**kwargs)
    
    def test(self, **kwargs):
        kwargs.setdefault("shuffle", False)
        return self.dataloader(**kwargs)