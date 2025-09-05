from typing import Dict, Optional
from torch.utils.data import Dataset, DataLoader


from accelerator.utilities.hydra_utils import instantiate
from accelerator.utilities.default_config import _DefaultConfig
from accelerator.typings.base import ConfigType


class DataModuleDefaults(_DefaultConfig):
    train_name: str = 'train'
    val_name: str = 'val'
    test_name: str = 'test'



class DataModule:
    def __init__(
        self, 
        datasets: Dict[str, Dataset], 
        dataloaders: Dict[str, DataLoader], 
        config: Optional[ConfigType] = None
    ):
        self._cfg: ConfigType = DataModuleDefaults.create(config)
        self._datasets: Dict[str, Dataset] = datasets
        self._dataloaders: Dict[str, DataLoader] = dataloaders

    @staticmethod
    def initialize_from_config(cfg: ConfigType) -> 'DataModule':
        datasets = {}
        if cfg.get('datasets'):
            for name, dataset_cfg in cfg.datasets.items():
                datasets[name] = instantiate(dataset_cfg)

        dataloaders = {}
        if cfg.get('dataloaders'):
            for name, loader_cfg in cfg.dataloaders.items():
                dataset = datasets.get(name)
                if dataset is not None:
                    params = instantiate(loader_cfg)
                    dataloaders[name] = DataLoader(dataset=dataset, **params)

        instance = DataModule(datasets, dataloaders, cfg)
        return instance

    @property
    def cfg(self) -> ConfigType:
        return self._cfg

    @cfg.setter
    def cfg(self, value: ConfigType) -> None:
        self._cfg = DataModuleDefaults.create(value)

    @property
    def datasets(self) -> Dict[str, Dataset]:
        return self._datasets
    
    @property
    def dataloaders(self) -> Dict[str, DataLoader]:
        return self._dataloaders

    @property
    def loader_names(self) -> list:
        return list(self.cfg.get('dataloaders', {}).keys())

    @property
    def dataset_names(self) -> list:
        return list(self.cfg.get('datasets', {}).keys())

    def get_dataloader(self, name: str) -> Optional[DataLoader]:
        return self._dataloaders.get(name)

    def get_dataset(self, name: str) -> Optional[Dataset]:
        return self._datasets.get(name)

    @property
    def train_loader(self) -> Optional[DataLoader]:
        return self.get_dataloader(self._cfg.tra)

    @property
    def val_loader(self) -> Optional[DataLoader]:
        return self.get_dataloader('val')

    @property
    def test_loader(self) -> Optional[DataLoader]:
        return self.get_dataloader('test')