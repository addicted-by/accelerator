import fire

from .add import ComponentConfigGenerator
from .analyze import AnalyzeCLI
from .configure import ProjectConfigurator
from .experiment import Experiment


class CLI:
    """Accelerator CLI - Command Line Interface for managing configurations."""

    def __init__(self):
        self.add = ComponentConfigGenerator()
        self.config = ProjectConfigurator()
        self.experiment = Experiment()
        self.analyze = AnalyzeCLI()
        # Future basic commands - uncomment when ready:
        # self.remove = ComponentRemover()
        # self.list = ComponentLister()
        # self.update = ComponentUpdater()

    def version(self):
        """Show accelerator version."""
        return "accelerator CLI v1.0.0"

    def help(self):
        """Show available commands."""
        commands = {
            "add": "Generate configuration files for basic components (model, optimizer, scheduler, callback)",
            "config": "Create `configs` folder in the core project",
            "train": "Start training with specified config path",
            # 'remove': 'Remove configuration files',
            # 'list': 'List available configurations',
            # 'update': 'Update existing configurations'
        }

        print("Available commands:")
        for cmd, desc in commands.items():
            print(f"  {cmd:<12} {desc}")

        print("\nUsage examples:")
        print("  python -m accelerator.cli add model torch.nn.Linear linear")
        print("  python -m accelerator.cli add optimizer torch.optim.Adam adam")
        print("  python -m accelerator.cli add scheduler torch.optim.lr_scheduler.StepLR step_lr")
        print("  python -m accelerator.cli add --help")

        print("\nSupported shortcuts:")
        print("  python -m accelerator.cli add model <class_path> <n>")
        print("  python -m accelerator.cli add optimizer <class_path> <n>")
        print("  python -m accelerator.cli add scheduler <class_path> <n>")

        print("\nNote: callback, loss, transform, and dataset will have specialized commands")

        return commands


def main():
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
