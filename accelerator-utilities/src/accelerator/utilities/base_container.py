from abc import ABC, abstractmethod
from typing import Any

from accelerator.utilities.logging import get_logger

logger = get_logger(__name__)


class BaseContainer(ABC):
    """
    Foundational container class providing systematic representation architecture
    and extensibility framework for specialized container implementations.
    """

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        summary = self._get_summary_info()

        lines = [f"{class_name}({summary})"]

        sections = self._get_representation_sections()
        for section_title, section_content in sections:
            if section_content:
                lines.append(f"  {section_title}:")
                for item in section_content:
                    lines.append(f"    {item}")

        return "\n".join(lines)

    @abstractmethod
    def _get_summary_info(self) -> str:
        """Generate concise summary information for the container header."""
        pass

    @abstractmethod
    def _get_representation_sections(self) -> list[tuple[str, list[str]]]:
        """Generate structured sections for detailed container representation."""
        pass


class ComponentContainer(BaseContainer):
    """
    Specialized container for component aggregation with systematic
    active/inactive component enumeration and status analysis.
    """

    def _get_component_mapping(self) -> dict[str, Any]:
        """Return mapping of component names to instances for analysis."""
        return {}

    def _get_summary_info(self) -> str:
        component_mapping = self._get_component_mapping()
        active_count = sum(1 for comp in component_mapping.values() if comp is not None)
        total_count = len(component_mapping)
        return f"{active_count}/{total_count} active"

    def _get_representation_sections(self) -> list[tuple[str, list[str]]]:
        sections = []
        component_mapping = self._get_component_mapping()

        active_components = []
        inactive_components = []

        for name, component in component_mapping.items():
            if component is not None:
                component_type = type(component).__name__
                active_components.append(f"{name}: {component_type}")
            else:
                inactive_components.append(name)

        if active_components:
            sections.append(("Active", active_components))

        if inactive_components:
            sections.append(("Inactive", inactive_components))

        additional_sections = self._get_additional_sections()
        sections.extend(additional_sections)

        return sections

    def _get_additional_sections(self) -> list[tuple[str, list[str]]]:
        """Override point for specialized component information."""
        return []


class StateContainer(BaseContainer):
    """
    Specialized container for state management with temporal progression
    tracking and metrics presentation capabilities.
    """

    def _get_state_metrics(self) -> dict[str, Any]:
        """Return current state metrics for analysis."""
        return {}

    def _get_progression_info(self) -> dict[str, Any]:
        """Return temporal progression indicators."""
        return {}

    def _get_summary_info(self) -> str:
        progression = self._get_progression_info()
        if "epoch" in progression and "step" in progression:
            return f"epoch={progression['epoch']}, step={progression['step']}"
        return "initialized"

    def _get_representation_sections(self) -> list[tuple[str, list[str]]]:
        sections = []

        progression = self._get_progression_info()
        if progression:
            progression_items = []
            for key, value in progression.items():
                if key not in ["epoch", "step"]:
                    if isinstance(value, (int, float)) and key.endswith("_metric"):
                        progression_items.append(f"{key}: {value:.6f}")
                    else:
                        progression_items.append(f"{key}: {value}")

            if progression_items:
                sections.append(("Progression", progression_items))

        metrics = self._get_state_metrics()
        for category, category_metrics in metrics.items():
            if category_metrics:
                metric_items = []
                for key, value in category_metrics.items():
                    if isinstance(value, (int, float)):
                        metric_items.append(f"{key}: {value:.6f}")
                    else:
                        metric_items.append(f"{key}: {value}")

                if metric_items:
                    sections.append((category, metric_items))

        additional_sections = self._get_additional_sections()
        sections.extend(additional_sections)

        return sections

    def _get_additional_sections(self) -> list[tuple[str, list[str]]]:
        """Override point for specialized state information."""
        return []
