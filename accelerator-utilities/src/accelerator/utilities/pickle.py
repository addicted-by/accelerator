import pathlib
import pickle
from typing import Any


def _pkl_dump(path: pathlib.Path, obj: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✔  Saved → {path}")
