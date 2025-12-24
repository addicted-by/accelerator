import string
from typing import Any


def get_field_names(template: str) -> list[str]:
    fmt = string.Formatter()
    names = list()
    for _, field_name, _, _ in fmt.parse(template):
        if field_name is not None and field_name != "":
            names.append(field_name)
    return names


def render_filename(template: str, values: dict[str, Any]) -> str:
    needed = get_field_names(template)
    subset = {k: v for k, v in values.items() if k in needed}
    return template.format(**subset)
