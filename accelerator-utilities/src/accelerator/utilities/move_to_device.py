from typing import Union, Any
from accelerator.utilities.apply_func import apply_to_collection
from accelerator.utilities.typings import _DEVICE
from accelerator.utilities.api_desc import APIDesc
import torch
import abc

_BLOCKING_DEVICE_TYPES = ['cpu', 'mps']

class _TransferableDataType(abc.ABC):
    """A custom type for data that can be moved to a torch device via ``.to(...)``."""

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is _TransferableDataType:
            to = getattr(subclass, "to", None)
            return callable(to)
        return NotImplemented
    

@APIDesc.developer(dev_info='Ryabykin Alexey r00926208')
@APIDesc.status(status_level='Stable')
@APIDesc.see_also([
    'https://github.com/Lightning-AI/pytorch-lightning/blob/0.9.0/pytorch_lightning/utilities/apply_func.py#L92-L124',
    'https://pytorch-lightning.readthedocs.io/en/0.9.0/api/pytorch_lightning.utilities.apply_func.html'
])
def move_data_to_device(batch: Any, device: _DEVICE) -> Any:
    """Transfers a collection of data to the given device. Any object that defines a method ``to(device)`` will be
    moved and all other objects in the collection will be left untouched."""
    if isinstance(device, str):
        device = torch.device(device)

    def batch_to(data: Any) -> Any:
        kwargs = {}
        if (
            isinstance(data, torch.Tensor) 
            and 
            isinstance(device, torch.device) 
            and 
            device.type not in _BLOCKING_DEVICE_TYPES
        ):
            kwargs["non_blocking"] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        return data

    return apply_to_collection(batch, dtype=_TransferableDataType, function=batch_to)