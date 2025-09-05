from __future__ import annotations

import functools
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml
from rich.console import Console
from rich.prompt import Prompt

from accelerator.runtime.loss.registry import LossType, registry
from accelerator.typings.base import PathType

console = Console()


@dataclass
class ComponentSpec:
    template_path: Path
    merge_fn: Callable[
        ["ComponentConfigGenerator", Dict[str, Any], str, Dict[str, Any], Any],
        Dict[str, Any],
    ]
    output_dir: Callable[[Dict[str, Any]], Path]
    index_file: Union[PathType, Callable[[Dict[str, Any]], Path]] = None


class ComponentConfigGenerator:
    """
    Generates YAML stubs for any supported component type.

    Public methods (Fire CLI entry points):
        • model(class_path, config_name)
        • optimizer(...)
        • scheduler(...)
        • callback(...)
        • loss(...)
        • transform(...)
        • datamodule(...)

    """

    _REGISTRY: Dict[str, ComponentSpec] = {}

    def model(self, class_path: str, config_name: str):
        return self._generate("model", class_path, config_name)

    def optimizer(self, class_path: str, config_name: str):
        return self._generate("optimizer", class_path, config_name)

    def scheduler(self, class_path: str, config_name: str):
        return self._generate("scheduler", class_path, config_name)

    def callback(self, class_path: str, config_name: str):
        return self._generate("callback", class_path, config_name)

    def loss(self, class_path: str, config_name: Optional[str] = None):
        return self._generate("loss", class_path, config_name)

    def transform(self, class_path: str, config_name: str):
        return self._generate("transform", class_path, config_name)

    def datamodule(self, class_path: str, config_name: str, data_name: str):
        return self._generate_datamodule(
            "datamodule", class_path, config_name, data_name
        )

    def sync_losses(self, overwrite: bool = False):
        """Synchronize all registered losses with configuration files.
        
        This method generates configuration files for all losses currently
        registered in the loss registry, ensuring that the config system
        stays in sync with the registry.
        
        Args:
            overwrite: If True, overwrite existing configuration files.
                      If False, skip existing files.
        """
        if not self._REGISTRY:
            self._build_registry()
            
        console.print("[bold blue]Synchronizing registered losses with configuration files...[/]")
        
        # Get all registered losses
        all_losses = registry.list_losses()
        total_losses = sum(len(losses) for losses in all_losses.values())
        
        if total_losses == 0:
            console.print("[yellow]No losses found in registry.[/]")
            return
            
        console.print(f"[cyan]Found {total_losses} registered losses across {len(all_losses)} categories[/]")
        
        generated_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each loss type
        for loss_type, loss_names in all_losses.items():
            if not loss_names:  # Skip empty categories
                continue
                
            console.print(f"\n[bold cyan]Processing {loss_type} losses ({len(loss_names)} items):[/]")
            
            for loss_name in loss_names:
                try:
                    # Get the registered loss object
                    loss_obj = registry.get_loss(loss_type, loss_name)
                    
                    # Determine the class path for the loss
                    class_path = self._get_loss_class_path(loss_obj)
                    
                    if class_path is None:
                        console.print(f"  [yellow]⚠ Skipping {loss_name}: Cannot determine class path[/]")
                        skipped_count += 1
                        continue
                    
                    # Check if config already exists
                    spec = self._REGISTRY["loss"]
                    template = self._load_default_template(spec.template_path)
                    
                    # Create a temporary merged config to determine output path
                    temp_merged = {"loss_type": loss_type, "name": loss_name}
                    output_dir = spec.output_dir(temp_merged)
                    config_path = output_dir / f"{loss_name}.yaml"
                    
                    if config_path.exists() and not overwrite:
                        console.print(f"  [dim]→ Skipping {loss_name}: Config already exists[/]")
                        skipped_count += 1
                        continue
                    
                    # Generate the configuration
                    try:
                        # Load the target class to get parameters
                        tgt_cls = self._load_target_class(class_path)
                        params = self._gather_class_params(tgt_cls)
                        self._current_params = params
                        
                        # Merge the configuration
                        merged = self._merge_loss(template, class_path, params, tgt_cls)
                        self._current_merged_config = merged
                        
                        # Write the configuration file
                        out_path = self._write_config_file_sync(
                            spec, "loss", loss_name, merged, overwrite
                        )
                        
                        # Update index file
                        if spec.index_file is not None:
                            self._update_index_file(spec, loss_name)
                        
                        console.print(f"  [green]✓ Generated config for {loss_name}[/]")
                        generated_count += 1
                        
                    except Exception as e:
                        console.print(f"  [red]✗ Error generating config for {loss_name}: {e}[/]")
                        error_count += 1
                        
                except KeyError:
                    console.print(f"  [red]✗ Error: Loss {loss_name} not found in registry[/]")
                    error_count += 1
                except Exception as e:
                    console.print(f"  [red]✗ Error processing {loss_name}: {e}[/]")
                    error_count += 1
        
        # Print summary
        console.print(f"\n[bold green]Synchronization complete![/]")
        console.print(f"  Generated: {generated_count}")
        console.print(f"  Skipped: {skipped_count}")
        if error_count > 0:
            console.print(f"  [red]Errors: {error_count}[/]")
            
        return {
            "generated": generated_count,
            "skipped": skipped_count,
            "errors": error_count
        }

    def _generate_datamodule(
        self, component_type: str, class_path: str, config_name: str, data_name: str
    ) -> str:
        if not self._REGISTRY:
            self._build_registry()

        spec = self._REGISTRY.get(component_type)
        if spec is None:
            raise ValueError(f"Unsupported component type: {component_type}")

        tgt_cls = self._load_target_class(class_path)
        params = self._gather_class_params(tgt_cls)
        self._current_params = params

        config_path = spec.output_dir({}) / f"{config_name}.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = yaml.safe_load(f) or {}
        else:
            template = self._load_default_template(spec.template_path)
            existing_config = yaml.safe_load(yaml.dump(template))

        merged = spec.merge_fn(existing_config, class_path, params, tgt_cls, data_name)

        self._current_merged_config = merged

        out_path = self._write_config_file(spec, component_type, config_name, merged)

        if spec.index_file is not None:
            self._update_index_file(spec, config_name)

        self._announce_creation(f"{component_type} ({data_name})", out_path)
        return str(out_path)

    def _generate(self, component_type: str, class_path: str, config_name: str) -> str:
        if not self._REGISTRY:
            self._build_registry()

        spec = self._REGISTRY.get(component_type)
        if spec is None:
            raise ValueError(f"Unsupported component type: {component_type}")

        tgt_cls = self._load_target_class(class_path)
        params = self._gather_class_params(tgt_cls)
        self._current_params = params

        template = self._load_default_template(spec.template_path)
        merged = spec.merge_fn(template, class_path, params, tgt_cls)

        # Store merged config for index file path resolution
        self._current_merged_config = merged

        out_path = self._write_config_file(
            spec, component_type, config_name or merged["name"], merged
        )

        # Update index file if component has index_file specified
        if spec.index_file is not None:
            self._update_index_file(spec, config_name or merged["name"])

        self._announce_creation(component_type, out_path)
        return str(out_path)

    def _find_loss_in_registry(self, loss_class) -> Optional[str]:
        """Find which registry type contains the given loss class.

        Args:
            loss_class: The loss class to search for in the registry

        Returns:
            The registry type string if found, None otherwise
        """
        # Search through all loss types
        for loss_type in LossType:
            # Get all losses registered under this type
            losses_dict = registry.list_losses(loss_type.value)
            if loss_type.value in losses_dict:
                loss_names = losses_dict[loss_type.value]

                # Check each registered loss in this type
                for loss_name in loss_names:
                    try:
                        registered_loss = registry.get_loss(loss_type.value, loss_name)
                        # Check if this is the same class (handle adapters)
                        if self._is_same_loss_class(registered_loss, loss_class):
                            return loss_type.value
                    except KeyError:
                        continue

        return None

    def _is_same_loss_class(self, registered_loss, target_class) -> bool:
        """Compare registered loss with target class, handling adapter wrappers.

        Args:
            registered_loss: The loss retrieved from the registry
            target_class: The target class to compare against

        Returns:
            True if they represent the same loss class, False otherwise
        """
        # Handle adapted losses
        if hasattr(registered_loss, "__original_cls__"):
            return registered_loss.__original_cls__ == target_class

        # Handle function adapters
        if hasattr(registered_loss, "__original_fn__"):
            return registered_loss.__original_fn__ == target_class

        # Direct comparison
        return (
            registered_loss == target_class or registered_loss.__class__ == target_class
        )

    def _build_registry(self):
        base = Path("configs")

        self._REGISTRY = {
            "model": ComponentSpec(
                tpl := base / "model" / "default.yaml",
                self._merge_model,
                lambda cfg, p=tpl.parent: p,
            ),
            "optimizer": ComponentSpec(
                tpl := base / "training_components" / "optimizer" / "default.yaml",
                functools.partial(self._merge_top_level, exclude_params=["params"]),
                lambda cfg, p=tpl.parent: p,
            ),
            "scheduler": ComponentSpec(
                tpl := base / "training_components" / "scheduler" / "scheduler.yaml",
                self._merge_top_level,
                lambda cfg, p=tpl.parent: p,
            ),
            "loss": ComponentSpec(
                tpl := base / "training_components" / "loss" / "default.yaml",
                self._merge_loss,
                lambda cfg, root=tpl.parent: root / str(cfg.get("loss_type", "custom")),
                index_file=lambda cfg, root=tpl.parent: root / "index.yaml",
            ),
            "transform": ComponentSpec(
                tpl := base / "training_components" / "transform" / "default.yaml",
                self._merge_transform,
                lambda cfg, root=tpl.parent: root
                / str(cfg.get("transform_type", "undefined")),
                index_file=lambda cfg, root=tpl.parent: root
                / str(cfg.get("transform_type", "undefined"))
                / "index.yaml",
            ),
            "datamodule": ComponentSpec(
                tpl := base / "datamodule" / "default.yaml",
                self._merge_datamodule,
                lambda cfg, p=tpl.parent: p,
            ),
        }

    def _merge_model(
        self, template: Dict[str, Any], class_path: str, params: Dict[str, Any], *_
    ) -> Dict[str, Any]:
        """
        Inserts _target_ and parameter defaults under template["model_core"].
        """
        result = yaml.safe_load(yaml.dump(template))
        core = result.setdefault("model_core", {})
        core["_target_"] = class_path

        for name, meta in params.items():
            core.setdefault(name, meta["value"])

        return result

    def _merge_top_level(
        self,
        template: Dict[str, Any],
        class_path: str,
        params: Dict[str, Any],
        *_,
        exclude_params: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        result = yaml.safe_load(yaml.dump(template))
        if result is None:
            result = {}
        result["_target_"] = class_path
        for name, meta in params.items():
            if exclude_params:
                if name in exclude_params:
                    continue
            result.setdefault(name, meta["value"])
        return result

    def _merge_loss(
        self,
        template: Dict[str, Any],
        class_path: str,
        params: Dict[str, Any],
        tgt_obj,
    ) -> Dict[str, Any]:
        exclude_params = [
            "net_result",
            "ground_truth",
            "device",
            "context",
            "inputs",
            "additional_info",
        ]

        result = (
            yaml.safe_load(
                yaml.dump(template, sort_keys=False, default_flow_style=True)
            )
            or {}
        )

        reg_type = getattr(tgt_obj, "registry_type", None)

        if reg_type is None:
            reg_type = self._find_loss_in_registry(tgt_obj)

        if reg_type is None:
            available_losses = registry.list_losses()
            raise ValueError(
                f"Loss '{tgt_obj.__name__}' not found in registry. "
                f"Available losses: {available_losses}. "
                f"Please register it with 'registry.register_loss(<loss_type>, <loss_name>)' "
                f"or 'registry.add_loss(<loss_type>, <loss_class>, <loss_name>)'."
            )

        result["loss_type"] = reg_type
        result["name"] = tgt_obj.__name__

        for key in exclude_params:
            params.pop(key, None)
            result.pop(key, None)

        for key, meta in params.items():
            result[key] = meta["value"]

        return result

    def _merge_transform(
        self, template: Dict[str, Any], class_path: str, params: Dict[str, Any], tgt_obj
    ) -> Dict[str, Any]:
        exclude_params = []

        result = (
            yaml.safe_load(
                yaml.dump(template, sort_keys=False, default_flow_style=True)
            )
            or {}
        )
        reg_type = getattr(tgt_obj, "registry_type", None)
        if reg_type is None:
            raise ValueError(
                dedent(
                    f"""{tgt_obj} Please, firstly decorate your object 
                with `registry.register_transform(<transform_type>, <transform_name (Optional)>)`"""
                )
            )
        result["transform_type"] = reg_type
        result["name"] = tgt_obj.__name__

        for key in exclude_params:
            params.pop(key, None)
            result.pop(key, None)

        for key, meta in params.items():
            result[key] = meta["value"]
        return result

    def _merge_datamodule(
        self,
        existing_config: Dict[str, Any],
        class_path: str,
        params: Dict[str, Any],
        tgt_cls,
        data_name: str,
    ) -> Dict[str, Any]:
        """
        Add dataset and dataloader configuration for a specific data_name (e.g., train, val).
        """
        result = yaml.safe_load(yaml.dump(existing_config))

        if "datasets" not in result:
            result["datasets"] = {}
        if "dataloaders" not in result:
            result["dataloaders"] = {}

        dataset_config = {"_target_": class_path}
        for name, meta in params.items():
            dataset_config[name] = meta["value"]

        result["datasets"][data_name] = dataset_config

        dataloader_config = {
            "batch_size": 32,
            "shuffle": True if data_name == "train" else False,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
        }

        result["dataloaders"][data_name] = dataloader_config

        return result

    def _load_target_class(self, class_path: str):
        module_path, cls_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)

    def _gather_class_params(self, obj):
        if hasattr(obj, "__original_cls__"):
            obj = obj.__original_cls__

        if hasattr(obj, "__original_fn__"):
            obj = obj.__original_fn__

        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            type_hints = get_type_hints(obj, include_extras=True)
        else:
            sig = inspect.signature(obj.__init__)
            type_hints = get_type_hints(obj.__init__, include_extras=True)

        params: Dict[str, Dict[str, Any]] = {}
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            th = type_hints.get(name, Any)
            params[name] = {
                "type": self._format_type_hint(th),
                "value": self._default_for(th, p),
                "has_default": p.default is not inspect.Parameter.empty,
            }
        return params

    def _load_default_template(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Default template for component not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _confirm_overwrite(self, path: Path) -> bool:
        """Ask the user—with Rich styling—whether to overwrite an existing file."""
        console.print(f"[bold yellow]File exists:[/] {path}")
        answer = Prompt.ask("[cyan]Overwrite?", choices=["y", "n"], default="n")
        return answer.lower() in {"y", "yes"}

    def _write_config_file(
        self,
        spec: ComponentSpec,
        component_type: str,
        config_name: str,
        data: Dict[str, Any],
    ) -> Path:
        cfg_dir = spec.output_dir(data)  # ← CHANGE (no if-else)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        out_path = cfg_dir / f"{config_name}.yaml"

        if out_path.exists():
            if not self._confirm_overwrite(out_path):
                console.print("[red]Aborted – existing file left unchanged.[/]")
                return out_path

        with open(out_path, "w") as f:
            f.write(self._render_yaml_with_comments(data, component_type))
        return out_path

    def _write_config_file_sync(
        self,
        spec: ComponentSpec,
        component_type: str,
        config_name: str,
        data: Dict[str, Any],
        overwrite: bool = False,
    ) -> Path:
        """Write config file for sync operation without interactive prompts."""
        cfg_dir = spec.output_dir(data)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        out_path = cfg_dir / f"{config_name}.yaml"

        if out_path.exists() and not overwrite:
            return out_path

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self._render_yaml_with_comments(data, component_type))
        return out_path

    def _get_loss_class_path(self, loss_obj) -> Optional[str]:
        """Determine the class path for a registered loss object.
        
        Args:
            loss_obj: The loss object retrieved from the registry
            
        Returns:
            The class path string if determinable, None otherwise
        """
        # Handle adapted losses with original class
        if hasattr(loss_obj, "__original_cls__"):
            original_cls = loss_obj.__original_cls__
            return f"{original_cls.__module__}.{original_cls.__qualname__}"
        
        # Handle function adapters with original function
        if hasattr(loss_obj, "__original_fn__"):
            original_fn = loss_obj.__original_fn__
            return f"{original_fn.__module__}.{original_fn.__qualname__}"
        
        # Handle direct class registration
        if inspect.isclass(loss_obj):
            return f"{loss_obj.__module__}.{loss_obj.__qualname__}"
        
        # Handle instance objects
        if hasattr(loss_obj, "__class__"):
            cls = loss_obj.__class__
            return f"{cls.__module__}.{cls.__qualname__}"
        
        return None

    def _update_index_file(self, spec: ComponentSpec, config_name: str) -> None:
        """
        Updates the index file by appending the new component name to the defaults list.

        Args:
            spec: ComponentSpec containing index_file path information
            config_name: Name of the config to add to the index
        """
        if spec.index_file is None:
            return

        if callable(spec.index_file):
            index_path = spec.index_file(getattr(self, "_current_merged_config", {}))
        else:
            index_path = spec.index_file

        try:
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = yaml.safe_load(f) or {}
            else:
                index_data = {}

            if "defaults" not in index_data:
                index_data["defaults"] = []

            entry = f"{config_name}@{config_name}"

            if entry not in index_data["defaults"]:
                index_data["defaults"].append(entry)

                index_path.parent.mkdir(parents=True, exist_ok=True)

                with open(index_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        index_data, f, default_flow_style=False, sort_keys=False
                    )

                console.print(f"[green]Updated index file:[/] {index_path}")
            else:
                console.print(f"[yellow]Component already exists in index:[/] {entry}")

        except PermissionError:
            console.print(
                f"[red]Error: Permission denied writing to index file:[/] {index_path}"
            )
        except yaml.YAMLError as e:  # noqa: F841
            console.print(
                f"[yellow]Warning: Malformed YAML in index file {index_path}, recreating...[/]"
            )
            index_data = {"defaults": [f"{config_name}@{config_name}"]}
            try:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                with open(index_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        index_data, f, default_flow_style=False, sort_keys=False
                    )
                console.print(f"[green]Recreated index file:[/] {index_path}")
            except Exception as recreate_error:
                console.print(
                    f"[red]Error: Failed to recreate index file:[/] {recreate_error}"
                )
        except Exception as e:
            console.print(f"[red]Error updating index file:[/] {e}")

    def _announce_creation(self, component_type: str, path: Path):
        print(f"Generated {component_type} config ➜ {path}")

    def _default_for(self, type_hint, param: inspect.Parameter):
        # If the user provided an explicit default, sanitise it
        if param.default is not inspect.Parameter.empty:
            return self._make_yaml_safe(param.default)

        origin = get_origin(type_hint)
        if origin is list or type_hint is list:
            return []
        if origin is dict or type_hint is dict:
            return {}
        if origin is Union and type(None) in get_args(type_hint):
            return None
        if type_hint in (int, float):
            return 0
        if type_hint is str:
            return ""
        if type_hint is bool:
            return False
        return None

    def _make_yaml_safe(self, obj):
        """Return a YAML-serialisable representation of *obj*."""
        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return list(obj)
        if isinstance(obj, dict):
            return dict(obj)

        if inspect.isclass(obj) or callable(obj):
            module = getattr(obj, "__module__", "<unknown>")
            qual = getattr(obj, "__qualname__", repr(obj))
            return {"_object_": f"{module}.{qual}"}

        return {"_object_": repr(obj)}

    def _format_type_hint(self, th) -> str:
        """
        Recursively converts typing hints into human-readable strings.

        Examples
        --------
        >>> self._format_type_hint(Optional[int])
        'Optional[int]'

        >>> self._format_type_hint(Tuple[int, int])
        'Tuple[int, int]'

        >>> self._format_type_hint(Union[List[int], Dict[str, float]])
        'Union[List[int], Dict[str, float]]'
        """
        if th is Any:
            return "Any"
        if hasattr(th, "__name__") and get_origin(th) is None:
            return th.__name__

        origin = get_origin(th)
        if origin is None:  # might be ForwardRef, TypeVar, etc.
            if isinstance(th, ForwardRef):
                return th.__forward_arg__
            return str(th).replace("typing.", "")

        args = get_args(th)
        if origin is Union and type(None) in args and len(args) == 2:
            non_none = args[0] if args[1] is type(None) else args[1]
            return f"Optional[{self._format_type_hint(non_none)}]"

        inner = ", ".join(self._format_type_hint(a) for a in args)
        return f"{origin.__name__}[{inner}]"

    def _render_yaml_with_comments(
        self, data: Dict[str, Any], component_type: str
    ) -> str:
        """
        Recursively render YAML while appending `# Type: X` comments
        for parameters discovered via introspection.
        """
        param_meta = getattr(self, "_current_params", {})
        lines: list[str] = [
            "# Auto-generated configuration file",
            "# Modify parameters as needed",
            "",
        ]

        def write_dict(obj: Dict[str, Any], indent: int = 0, in_model_core=False):
            indent_str = "  " * indent
            for key, value in obj.items():
                comment = ""
                if (in_model_core or component_type != "model") and key in param_meta:
                    tp = param_meta[key]["type"]
                    req = not param_meta[key]["has_default"]
                    comment = f"  # Type: {tp}" + (" (required)" if req else "")

                if value is None:
                    lines.append(f"{indent_str}{key}: null{comment}")
                elif isinstance(value, bool):
                    lines.append(f"{indent_str}{key}: {str(value).lower()}{comment}")
                elif isinstance(value, (int, float)):
                    lines.append(f"{indent_str}{key}: {value}{comment}")
                elif isinstance(value, str):
                    lines.append(f'{indent_str}{key}: "{value}"{comment}')
                elif isinstance(value, list):
                    if value:
                        lines.append(f"{indent_str}{key}: {value}{comment}")
                    else:
                        lines.append(f"{indent_str}{key}: []{comment}")
                elif isinstance(value, dict):
                    lines.append(f"{indent_str}{key}:")
                    write_dict(
                        value,
                        indent + 1,
                        in_model_core
                        or (component_type == "model" and key == "model_core"),
                    )
                else:
                    dumped = yaml.safe_dump(value, default_flow_style=True).strip()
                    lines.append(f"{indent_str}{key}: {dumped}{comment}")

        write_dict(data)
        return "\n".join(lines)
