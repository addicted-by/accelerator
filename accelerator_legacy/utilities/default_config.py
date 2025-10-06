from typing import Any, Dict, TypeVar, Type, Union, get_type_hints, get_origin, get_args
from dataclasses import dataclass, is_dataclass, asdict

from accelerator.utilities.api_desc import APIDesc


T = TypeVar("T", bound="_DefaultConfig")


@dataclass
class _DefaultConfig:
    """Base class for all configuration objects with deep merge capabilities."""

    @classmethod
    def create(cls: Type[T], overrides) -> Dict[str, Any]:
        instance = cls()
        if overrides:
            return instance.merge(overrides)
        return cls.to_dict(instance)

    def merge(self, overrides: Union[Dict[str, Any], T]) -> Dict[str, Any]:
        """Merge this config with overrides and return a new instance."""
        if isinstance(overrides, type(self)):
            overrides = asdict(overrides)

        merged_dict = self._deep_merge(asdict(self), overrides)
        return merged_dict

    @APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
    @APIDesc.status(
        status_level="Experimental",
        details="Once all configs should be refactored in dataclasses but I was too lazy...",
    )
    def _dict_to_instance(self, data: Dict[str, Any]) -> T:
        """Convert a dictionary back to an instance of this class."""
        valid_keys = get_type_hints(self.__class__)
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        for key, value in list(filtered_data.items()):
            if key in valid_keys:
                field_type = valid_keys[key]
                if get_origin(field_type) is Union and type(None) in get_args(
                    field_type
                ):
                    non_none_types = [
                        t for t in get_args(field_type) if t is not type(None)
                    ]
                    if non_none_types:
                        field_type = non_none_types[0]

                if isinstance(value, dict) and is_dataclass(field_type):
                    filtered_data[key] = field_type(**value)

                elif isinstance(value, list) and get_origin(field_type) is list:
                    item_type = get_args(field_type)[0] if get_args(field_type) else Any
                    if is_dataclass(item_type) and all(
                        isinstance(item, dict) for item in value
                    ):
                        filtered_data[key] = [item_type(**item) for item in value]

        return self.__class__(**filtered_data)

    @staticmethod
    def _deep_merge(base: Any, override: Any) -> Any:
        """Deep merge two values, with override values taking precedence."""
        if override is None:
            return base

        if not isinstance(base, type(override)) and not (
            isinstance(base, dict) and isinstance(override, dict)
        ):
            return override

        if isinstance(base, dict) and isinstance(override, dict):
            result = base.copy()
            for key, value in override.items():
                if key in result and (
                    isinstance(result[key], dict) or is_dataclass(result[key])
                ):
                    if isinstance(result[key], dict):
                        result[key] = _DefaultConfig._deep_merge(result[key], value)
                    else:
                        result[key] = _DefaultConfig._deep_merge(
                            asdict(result[key]),
                            value if isinstance(value, dict) else asdict(value),
                        )
                else:
                    result[key] = value
            return result

        if isinstance(base, list):
            return override

        if is_dataclass(base):
            if is_dataclass(override):
                return _DefaultConfig._deep_merge(asdict(base), asdict(override))
            elif isinstance(override, dict):
                return _DefaultConfig._deep_merge(asdict(base), override)

        return override

    def to_dict(self) -> Dict[str, Any]:
        """Convert this config to a dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """String representation with better formatting."""
        lines = [f"{self.__class__.__name__}:"]
        for k, v in asdict(self).items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
