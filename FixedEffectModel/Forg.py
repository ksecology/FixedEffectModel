def forg(x, prec=3):
    if prec == 3:
        # for 3 decimals
        if (abs(x) >= 1e3) or (abs(x) < 1e-3):
            return '%9.3g' % x
        else:
            return '%9.3f' % x
    elif prec == 4:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%10.4g' % x
        else:
            return '%10.4f' % x
    elif prec == 5:
        if (abs(x) >= 1e5) or (abs(x) < 1e-5):
            return '%10.5g' % x
        else:
            return '%10.5f' % x
    else:
        raise ValueError("`prec` argument must be either 3 or 4, not {prec}"
                         .format(prec=prec))