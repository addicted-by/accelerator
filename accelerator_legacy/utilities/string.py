from typing import List, Dict, Any
import string

def get_field_names(template: str) -> List[str]:
    fmt = string.Formatter()
    names = list()
    for literal_text, field_name, format_spec, conversion in fmt.parse(template):
        if field_name is not None and field_name != "":
            names.append(field_name)
    return names

def render_filename(template: str, values: Dict[str, Any]) -> str:
    needed = get_field_names(template)
    subset = {k: v for k, v in values.items() if k in needed}
    return template.format(**subset)