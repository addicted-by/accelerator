def smape_lp(y_true, y_pred, p=2, eps=1e-7):
    return (y_true - y_pred).abs().pow(p) / ((2 ** (p - 1)) * (y_true.abs().pow(p) + y_pred.abs().pow(p) + eps))