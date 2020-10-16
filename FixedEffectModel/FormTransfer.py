def form_transfer(form):
    """

    :param form: formula in string type. e.g. 'y~x1+x2|id+firm|id',dependent_variable~continuous_variable|fixed_effect|clusters
    :return: Lists of out_col, consist_col, category_col, cluster_col, respectively.
    """
    form = form.replace(' ', '')
    form = form.split('~')
    pos = 0
    out_col, consist_col, category_col, cluster_col = [], [], [], []
    for part in form:
        part = part.split('|')
        for p2 in part:
            pos += 1
            p2 = p2.split('+')
            if pos == 1:
                out_col = p2
            elif pos == 2:
                consist_col = p2
            elif pos == 3:
                category_col = p2
            elif pos == 4:
                cluster_col = p2
            else:
                raise NameError('Invalid formula, please refer to the right one')
    return out_col, consist_col, category_col, cluster_col
