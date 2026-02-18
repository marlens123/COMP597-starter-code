from src.config.util.base_config import _Arg, _BaseConfig

class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_folder = _Arg(type=str, help="The storage folder of the dataset.", default="")
        self._arg_loader = _Arg(type=str, help="Which dataloader to use.", default="pytorch")
        self._arg_split = _Arg(type=str, help="How to split the dataset (ex: train[:100])", default="")
        self._arg_load_num_proc = _Arg(type=int, help="Number of threads used to load the dataset.", default=0)
