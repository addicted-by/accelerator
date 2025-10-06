import argparse
import shutil
import importlib
import inspect
from pathlib import Path
from typing import Type, Optional, List
from omegaconf import OmegaConf


def copy_configs(source: Path, dest: Path) -> None:
    """Copy config templates preserving directory structure"""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def get_class_fields(cls: Type) -> dict:
    """Extract constructor parameters from a class"""
    init = getattr(cls, '__init__', None)
    if not init:
        return {}
    
    signature = inspect.signature(init)
    return {
        k: (v.default if v.default is not inspect.Parameter.empty else None)
        for k, v in signature.parameters.items()
        if k != 'self'
    }


def interactive_config_setup(config_path: Path) -> None:
    """Guide user through config setup process"""
    from rich.prompt import Prompt
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.console import RenderableType

    class FilePicker:
        def __init__(self, start_path: str = "."):
            self.current_path = Path(start_path).resolve()
            self.selected = 0  # Default select first item
            self.console = Console()
            self.layout = Layout()

        def update_listing(self) -> List[str]:
            """Generate directory listing with navigation"""
            items = []
            if self.current_path.parent != self.current_path:
                items.append("../")
            
            try:
                for item in sorted(self.current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
                    if item.is_dir():
                        items.append(f"{item.name}/")
                    elif item.name.endswith('.py'):  # Only show Python files
                        items.append(item.name)
            except PermissionError:
                self.console.print("[red]Permission denied to access this directory[/red]")
                self.current_path = self.current_path.parent
                return self.update_listing()
                
            return items

        def render_listing(self, items: List[str]) -> RenderableType:
            """Render directory listing with highlighted selection"""
            lines = []
            for i, item in enumerate(items):
                if i == self.selected:
                    lines.append(f"[bold white on blue]> {item}[/bold white on blue]")
                else:
                    style = "bright_blue" if item.endswith("/") else "white"
                    lines.append(f"  [bold {style}]{item}[/bold {style}]")
            
            return Panel(
                "\n".join(lines),
                title=f"[yellow]{self.current_path}[/yellow]",
                title_align="left",
                border_style="bright_blue"
            )

        def run(self) -> Optional[Path]:
            """Run interactive file selection"""
            self.console.clear()
            
            while True:
                items = self.update_listing()
                if not items:
                    self.console.print("[red]No files found in this directory[/red]")
                    self.current_path = self.current_path.parent
                    continue
                    
                # Ensure selected index is valid
                if self.selected >= len(items):
                    self.selected = len(items) - 1
                
                # Render the file listing
                self.console.print(self.render_listing(items))
                self.console.print("[bold]Navigation:[/bold] [↑/↓] Move selection | [→/Enter] Select | [←] Go back | [Esc] Cancel")
                
                key = self.console.input("Action: ")
                
                if key.lower() in ("down", "j", "s"):
                    self.selected = min(self.selected + 1, len(items) - 1)
                elif key.lower() in ("up", "k", "w"):
                    self.selected = max(self.selected - 1, 0)
                elif key.lower() in ("right", "enter", "l", "d"):
                    if self.selected < len(items):
                        selected_name = items[self.selected].rstrip("/")
                        new_path = self.current_path / selected_name
                        if items[self.selected].endswith("/"):
                            self.current_path = new_path
                            self.selected = 0
                        else:
                            return new_path
                elif key.lower() in ("left", "h", "a") or key.lower() == "backspace":
                    if self.current_path.parent != self.current_path:
                        self.current_path = self.current_path.parent
                        self.selected = 0
                elif key.lower() in ("escape", "q", "quit", "exit"):
                    return None
                
                self.console.clear()
    
    console = Console()
    console.print("[bold green]Interactive Config Setup[/bold green]")
    console.print("Please select a Python file containing your model class:\n")
    
    try:
        base_cfg = OmegaConf.load(config_path)
    except Exception as e:
        console.print(f"[red]Error loading config file: {e}[/red]")
        return
    
    # File selection using interactive picker
    picker = FilePicker(str(Path.cwd()))
    selected_file = picker.run()
    
    if not selected_file:
        console.print("[yellow]No file selected, exiting![/yellow]")
        return

    if not selected_file.name.endswith('.py'):
        console.print(f"[red]Selected file {selected_file.name} is not a Python file![/red]")
        return
    
    # Class discovery
    try:
        spec = importlib.util.spec_from_file_location(selected_file.stem, selected_file)
        if spec is None or spec.loader is None:
            console.print(f"[red]Could not load module from {selected_file}[/red]")
            return
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        classes = [cls for _, cls in inspect.getmembers(module, inspect.isclass) 
                if cls.__module__ == module.__name__]
        
        if not classes:
            console.print(f"[red]No classes found in {selected_file.name}[/red]")
            return
        
        console.print("\n[bold]Available classes:[/bold]")
        for i, cls in enumerate(classes):
            console.print(f"[{i+1}] {cls.__name__}")
        
        class_choices = [str(i+1) for i in range(len(classes))]
        class_choice = int(Prompt.ask("\nSelect class number", choices=class_choices, default="1")) - 1
        selected_class = classes[class_choice]
        
        # Generate module path
        rel_path = selected_file.relative_to(Path.cwd())
        module_path = '.'.join(rel_path.with_suffix('').parts)
        
        model_fields = get_class_fields(selected_class)
        
        console.print(f"\n[bold]Updating config with class:[/bold] {selected_class.__name__}")
        console.print(f"[bold]Module path:[/bold] {module_path}")
        console.print(f"[bold]Fields:[/bold] {', '.join(model_fields.keys()) or 'None'}")
        
        try:
            OmegaConf.update(base_cfg, 'model.model_core', {
                '_target_': f'{module_path}.{selected_class.__name__}',
                **model_fields
            }, force=True)
            
            # Save updated config
            OmegaConf.save(base_cfg, config_path)
            console.print(f"\n[bold green]Config successfully updated and saved to:[/bold green] {config_path}")
        except Exception as e:
            console.print(f'[red]Error updating config: {e}[/red]')
    except Exception as e:
        console.print(f'[red]Error processing selected file: {e}[/red]')


def main() -> None:
    parser = argparse.ArgumentParser(description='Initialize accelerator config')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing configs')
    
    package_configs = Path(__file__).parent.parent / 'configs'
    dest_configs = Path.cwd() / 'configs'
    
    copy_configs(package_configs, dest_configs)
    print(f'Config templates copied to {dest_configs}')
    
    model_config = dest_configs / 'model' / 'default.yaml'
    interactive_config_setup(model_config)


if __name__ == '__main__':
    main()