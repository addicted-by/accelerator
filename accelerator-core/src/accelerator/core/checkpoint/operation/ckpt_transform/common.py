from accelerator.utilities.logging import get_logger

log = get_logger(__name__)


def add_item(key_name, value, ckpt_model_dict):
    log.info(f"\t\t{key_name:50} with shape {value.shape} - added new to checkpoint")
    ckpt_model_dict[key_name] = value


def remove_item(key_name, ckpt_model_dict):
    log.info(f"\t\t{key_name:50} - removed from checkpoint")
    ckpt_model_dict.pop(key_name)


def rename_item(name_old, name_new, ckpt_model_dict):
    log.info(f"\t\t{name_old:25} -> {name_new:21}  - renamed")
    value = ckpt_model_dict.pop(name_old)
    ckpt_model_dict[name_new] = value
