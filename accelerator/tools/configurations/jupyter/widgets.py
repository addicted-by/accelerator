# accelerator/tools/configurations/jupyter/widgets.py
from pathlib import Path
from typing import Any, Dict, Tuple

import ipywidgets as w
from omegaconf import OmegaConf
import yaml
import copy

__all__ = ["ConfigSelector"]


# ── helpers ────────────────────────────────────────────────────────────────
def _widget_for(k: str, v: Any) -> w.Widget | None:
    style = {"description_width": "120px"}
    layout = w.Layout(width="65%")
    if isinstance(v, bool):
        return w.Checkbox(value=v, description=k, indent=False, style=style)
    if isinstance(v, int):
        return w.IntText(value=v, description=k, style=style, layout=layout)
    if isinstance(v, float):
        return w.FloatText(value=v, description=k, style=style, layout=layout)
    if isinstance(v, (str, Path)):
        return w.Text(value=str(v), description=k, style=style, layout=layout)
    if isinstance(v, (list, tuple)):
        return w.Textarea(
            value=yaml.safe_dump(v, default_flow_style=True).strip(),
            description=k,
            style=style,
            layout=w.Layout(width="75%", height="60px"),
        )
    if isinstance(v, dict):          # nested → handled recursively
        return None
    return w.Text(value=str(v), description=k, style=style, layout=layout)


def _val_from(wdg: w.Widget):
    if isinstance(wdg, w.Textarea):
        return yaml.safe_load(wdg.value or "null")
    return wdg.value


# ── main class ─────────────────────────────────────────────────────────────
class ConfigSelector(w.VBox):
    """
    Pick a component folder + YAML file, edit fields inline, then:

        cfg = selector.get_cfg("model")        # ΩmegaConf object
    """

    def __init__(self, cfg_root: str | Path):
        self.cfg_root = Path(cfg_root).expanduser().resolve()
        if not self.cfg_root.is_dir():
            raise ValueError(f"{self.cfg_root} is not a directory")

        self.folder_dd = w.Dropdown(
            options=[f.name for f in sorted(self.cfg_root.iterdir()) if f.is_dir()],
            description="Component:",
            layout=w.Layout(width="45%"),
            style={"description_width": "100px"},
        )
        self.file_dd = w.Dropdown(description="Config:", layout=w.Layout(width="55%"))
        self.form_box = w.VBox([])
        self.save_btn = w.Button(description="Save changes ��",
                                 button_style="success",
                                 disabled=True)
        self._saved_cfgs: Dict[Tuple[str, str], Dict] = {}
        self.msg_out = w.Output(layout=w.Layout(border="0"))

        super().__init__([
            self.folder_dd,
            self.file_dd,
            self.form_box,
            w.HBox([self.save_btn]),
            self.msg_out
        ])

        self.save_btn.on_click(self._on_save)
        self.folder_dd.observe(self._on_folder, names="value")
        self.file_dd.observe(self._on_file, names="value")
        self._on_folder({"new": self.folder_dd.value})

        self._cfg: Dict | None = None              # current editable dict
        self._cfg_path: Dict[str, Path] = {}       # component → path mapping
        self._widgets: Dict[Tuple[str, ...], w.Widget] = {}  # path tuple → widget


    # — notebook helpers —
    def display(self, component: str):
        """Show the dropdown & form for *component* (“model”, “datamodule”, …)."""
        self.folder_dd.value = component

    # — public API —
    def get_cfg(self, component: str):
        """Return ΩmegaConf of the currently edited YAML for *component*."""
        if self.folder_dd.value != component:
            raise RuntimeError(
                f"Component {component!r} is not active; call display('{component}') first."
            )
        if self._cfg is None:
            return None
        self._collect_values()
        return OmegaConf.create(self._cfg)

    __getitem__ = get_cfg          # allow selector["model"]

    # — event callbacks —
    def _on_folder(self, change):
        folder = self.cfg_root / change["new"]
        yaml_paths = sorted(folder.glob("*.yaml"))
        self.file_dd.options = [(p.stem, p) for p in yaml_paths]
        self.file_dd.value = None
        self.form_box.children = []
        self._cfg = None
        
        self.save_btn.disabled = True

    def _on_file(self, change):
        if change["new"] is None:
            self.form_box.children = []
            self._cfg = None
            self.save_btn.disabled = True
            return

        path = Path(change["new"])
        key = (self.folder_dd.value, str(path))

        # load from cache if we saved it earlier
        if key in self._saved_cfgs:
            self._cfg = copy.deepcopy(self._saved_cfgs[key])
        else:
            self._cfg = OmegaConf.to_container(
                OmegaConf.load(path), resolve=False
            )

        self._cfg_path[self.folder_dd.value] = path
        self.form_box.children = [self._build_form(self._cfg)]
        self.save_btn.disabled = False
        
    def _on_save(self, _):
        """Collect values from widgets and cache the edited YAML."""
        self._collect_values()
        key = (self.folder_dd.value, str(self._cfg_path[self.folder_dd.value]))
        self._saved_cfgs[key] = copy.deepcopy(self._cfg)

        self.msg_out.clear_output()
        with self.msg_out:
            print("✅ Changes saved for this session.")

    # — building / harvesting the form —
    def _build_form(self, d: dict, path: Tuple[str, ...] = ()):
        rows = []
        for k, v in d.items():
            wdg = _widget_for(k, v)
            if wdg is not None:
                wdg._cfg_path = (*path, k)         # stash key-path for later
                self._widgets[wdg._cfg_path] = wdg
                rows.append(wdg)
            else:                                  # nested dict → Accordion
                acc_child = self._build_form(v, (*path, k))
                acc = w.Accordion(children=[acc_child])
                acc.set_title(0, k)
                rows.append(acc)
        return w.VBox(rows)

    def _collect_values(self):
        for p, wdg in self._widgets.items():
            ptr = self._cfg
            *parents, leaf = p
            for key in parents:
                ptr = ptr[key]
            ptr[leaf] = _val_from(wdg)
