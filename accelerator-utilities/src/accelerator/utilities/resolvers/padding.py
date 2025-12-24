import omegaconf


def value_pad(x, pad_char="0", amount=3, direction="l"):
    s = str(x)
    if direction == "l":
        return s.rjust(amount, str(pad_char))
    elif direction == "r":
        return s.ljust(amount, str(pad_char))
    else:
        raise ValueError(f"Unknown direction '{direction}', expected 'l' or 'r'")


def zero_pad(x, amount=3, direction="l"):
    return value_pad(x, "0", amount=amount, direction=direction)


omegaconf.OmegaConf.register_new_resolver("value_pad", value_pad)
omegaconf.OmegaConf.register_new_resolver("zero_pad", zero_pad)
