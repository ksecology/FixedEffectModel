import re


def form_transfer(form):
    """

    :param form: formula in string type. e.g. 'y~x1+x2|id+firm|id',dependent_variable~continuous_variable|fixed_effect|clusters
    :return: Lists of out_col, consist_col, category_col, cluster_col, fake_x, iv_col respectively.
    """
    form = form.replace(' ', '')

    out_col, consist_col, category_col, cluster_col, fake_x, iv_col = [], [], [], [], [], []
    ivfinder = re.compile(r'[(](.*?)[)]', re.S)
    iv_expression = re.findall(ivfinder, form)
    if iv_expression:
        form = re.sub(r'[(](.*?)[)]', "", form)
        iv_expression = ''.join(iv_expression)
        iv_expression = iv_expression.split('~')

        fake_x = iv_expression[0].split('|')
        iv_col = iv_expression[1].split('+')
    form = form.split('~')
    pos = 0
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
            elif pos == 5:
                iv_col = iv_col
            else:
                raise NameError('Invalid formula, please refer to the right one')
                
    # if no input, replace 0 with null           
    if category_col[0] == '0':
        category_col = []
        
    if consist_col[0] == '0':
        consist_col = []
        
    return out_col, consist_col, category_col, cluster_col, fake_x, iv_col
