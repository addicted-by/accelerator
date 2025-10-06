import shutil
from pathlib import Path

try:
    import accelerator

    _accelerator_available = True
except ImportError:
    _accelerator_available = False


class ProjectConfigurator:
    """Configure project by copying default configurations."""

    def __init__(self):
        if _accelerator_available:
            self.accelerator_package_path = Path(accelerator.__file__).parent
            self.source_configs_dir = self.accelerator_package_path / "configs"
            self.source_mlproject = self.accelerator_package_path / "MLproject"
        else:
            self.accelerator_package_path = None
            self.source_configs_dir = None
        self.target_configs_dir = Path.cwd() / "configs"
        self.target_mlproject = Path.cwd() / "MLproject"

    def configure(self, force: bool = False, backup: bool = False):
        """Copy configs directory from accelerator module to current directory.

        Args:
            force: Overwrite existing configs directory without prompting
            backup: Create backup of existing configs before overwriting
        """
        source_available = (
            _accelerator_available
            and self.source_configs_dir
            and self.source_configs_dir.exists()
        )

        if not source_available:
            print("Warning: Source configs directory not found")
            if not _accelerator_available:
                print("Accelerator package not found in Python path")
            elif self.source_configs_dir:
                print(f"Expected location: {self.source_configs_dir}")
            return

        if self.target_configs_dir.exists():
            if not force:
                response = input(
                    f"Configs directory already exists at {self.target_configs_dir}. Overwrite? (y/N): "
                )
                if response.lower() not in ["y", "yes"]:
                    print("Configuration cancelled.")
                    return

            if backup:
                self._backup_existing_configs()

            shutil.rmtree(self.target_configs_dir)

        print(
            f"Copying configs from {self.source_configs_dir} to {self.target_configs_dir}"
        )
        shutil.copytree(self.source_configs_dir, self.target_configs_dir)
        shutil.copyfile(self.source_mlproject, self.target_mlproject)
        print("✅ Configuration complete!")
        print(f"   Configs copied to: {self.target_configs_dir}")
        print(f"   Source: {self.source_configs_dir}")

        self._show_structure()

    def init(self, force: bool = False):
        """Initialize project with minimal configs (alias for configure)."""
        return self.configure(force=force, backup=True)

    def status(self):
        """Show configuration status and structure."""
        print("Configuration Status:")
        print("=" * 40)
        print(f"Current directory: {Path.cwd()}")
        print(f"Target configs: {self.target_configs_dir}")
        print(f"Accelerator available: {_accelerator_available}")
        if _accelerator_available:
            print(f"Source configs: {self.source_configs_dir}")
            print(
                f"Source exists: {self.source_configs_dir.exists() if self.source_configs_dir else False}"
            )
        print(f"Target exists: {self.target_configs_dir.exists()}")

        if self.target_configs_dir.exists():
            print("\nCurrent configs structure:")
            self._show_structure()
        else:
            print("\nNo configs directory found. Run 'configure' to set up.")

    def clean(self, confirm: bool = False):
        """Remove configs directory from current location.

        Args:
            confirm: Skip confirmation prompt
        """
        if not self.target_configs_dir.exists():
            print("No configs directory found to clean.")
            return

        if not confirm:
            response = input(
                f"Remove configs directory at {self.target_configs_dir}? (y/N): "
            )
            if response.lower() not in ["y", "yes"]:
                print("Clean cancelled.")
                return

        shutil.rmtree(self.target_configs_dir)
        print(f"✅ Removed configs directory: {self.target_configs_dir}")

    def _backup_existing_configs(self):
        """Create backup of existing configs directory."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path.cwd() / f"configs_backup_{timestamp}"

        print(f"Creating backup: {backup_dir}")
        shutil.copytree(self.target_configs_dir, backup_dir)

    def _create_minimal_configs(self):
        """Create minimal configs structure when source is not available."""
        print("Creating minimal configs structure...")

        # Create directory structure
        dirs_to_create = [
            "configs/model",
            "configs/optimizer",
            "configs/scheduler",
            "configs/callback",
            "configs/loss",
            "configs/transform",
            "configs/dataset",
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Create default model config
        default_model_config = Path("configs/model/default.yaml")
        with open(default_model_config, "w") as f:
            f.write("# Default model configuration template\n")
            f.write(
                "# This file is used as a base when generating new model configs\n\n"
            )
            f.write("model_core:\n")
            f.write(
                "  _target_: ''  # This will be filled with the actual model class path\n"
            )
            f.write("  # Model class parameters will be added below automatically\n\n")
            f.write("# Model wrapper configuration\n")
            f.write("collate_fn: ~\n")
            f.write("separate_fn: ~\n\n")
            f.write("# Gamma configuration\n")
            f.write("gamma:\n")
            f.write("  use_gamma: false\n\n")
            f.write("# Depth-to-Space configuration (pixel shuffle)\n")
            f.write("d2s: 0\n")

        # Create README
        readme_path = Path("configs/README.md")
        with open(readme_path, "w") as f:
            f.write("# Accelerator Configurations\n\n")
            f.write(
                "This directory contains configuration files for the accelerator framework.\n\n"
            )
            f.write("## Structure\n\n")
            f.write("- `model/` - Neural network model configurations\n")
            f.write("- `optimizer/` - Training optimizer configurations\n")
            f.write("- `scheduler/` - Learning rate scheduler configurations\n")
            f.write("- `callback/` - Training callback configurations\n")
            f.write("- `loss/` - Loss function configurations\n")
            f.write("- `transform/` - Data/tensor transform configurations\n")
            f.write("- `dataset/` - Dataset configurations\n\n")
            f.write("## Usage\n\n")
            f.write("Use the accelerator CLI to generate configurations:\n\n")
            f.write("```bash\n")
            f.write("python commands.py add model torch.nn.Linear linear\n")
            f.write("python commands.py add optimizer torch.optim.Adam adam\n")
            f.write("```\n")

        print("✅ Minimal configuration structure created!")

    def _show_structure(self):
        """Show the current configs directory structure."""

        def print_tree(directory, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return

            items = sorted(directory.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")

                if item.is_dir() and current_depth < max_depth - 1:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    print_tree(item, next_prefix, max_depth, current_depth + 1)

        if self.target_configs_dir.exists():
            print(f"\n{self.target_configs_dir.name}/")
            print_tree(self.target_configs_dir)

        print("\nTo add configurations, use:")
        print("  python commands.py add model <class_path> <name>")
        print("  python commands.py add optimizer <class_path> <name>")
