"""
This module is used to transform expression of relative fixed effects(e.g.'id_1 - id_2')
"""


def do_operation(alpha_df, exp):
    exp = exp.replace(' ', '')
    exp_list = list(exp)
    temp = ''
    behavior_temp = '+'
    result = 0
    i = 0
    length = len(exp_list)
    for item in exp_list:
        if is_operation(item):
            if behavior_temp == '+':
                result += alpha_df.loc[temp]
            else:
                result -= alpha_df.loc[temp]
            behavior_temp = item
            temp = ''
        else:
            temp += item

        if i == length - 1:
            if behavior_temp == '+':
                result += alpha_df.loc[temp]
            else:
                result -= alpha_df.loc[temp]

        i += 1
    return result


def is_operation(oper):
    if oper == '+' or oper == '-':
        return True
    else:
        return False
