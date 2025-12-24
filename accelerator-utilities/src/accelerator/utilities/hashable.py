import collections.abc

from accelerator.utilities.api_desc import APIDesc


@APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
@APIDesc.status(status_level="Internal use only")
class _HashableConfigMixin:
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.config == other.config

    def __hash__(self) -> int:
        def make_hashable(value):
            if isinstance(value, dict):
                return tuple((k, make_hashable(v)) for k, v in sorted(value.items()))
            elif isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            elif isinstance(value, set):
                return frozenset(make_hashable(v) for v in value)
            elif isinstance(value, collections.abc.Hashable):
                return value
            else:
                return repr(value)

        if not hasattr(self, "config"):
            raise TypeError(f"{self.__class__.__name__} must define a 'config' attribute to be hashable")

        return hash((self.__class__, make_hashable(self.config)))
