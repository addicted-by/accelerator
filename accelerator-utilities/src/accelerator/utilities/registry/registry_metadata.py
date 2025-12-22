"""Registration metadata for registry objects."""

from dataclasses import dataclass
from typing import Callable

from accelerator.utilities.registry.domain import Domain


@dataclass
class RegistrationMetadata:
    """Metadata for a registered object.

    Attributes:
        name: The name of the registered object
        registry_type: The type of registry (e.g., 'loss', 'model')
        domain: The machine learning domain classification
        obj: The registered callable object
    """

    name: str
    registry_type: str
    domain: Domain
    obj: Callable

    @classmethod
    def create(
        cls, 
        name: str, 
        registry_type: str, 
        domain: Domain, 
        obj: Callable
    ) -> "RegistrationMetadata":
        """Factory method to create metadata with current timestamp.

        Args:
            name: The name of the registered object
            registry_type: The type of registry
            domain: The machine learning domain classification
            obj: The callable object to register

        Returns:
            RegistrationMetadata instance with current timestamp
        """
        return cls(
            name=name,
            registry_type=registry_type,
            domain=domain,
            obj=obj,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary containing metadata (excluding the callable object)
        """
        return {
            "name": self.name,
            "registry_type": self.registry_type,
            "domain": self.domain.value,
        }
