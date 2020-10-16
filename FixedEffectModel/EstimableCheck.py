"""
This module is used to check whether a certain fixed effect if estimable. Through two projection starting on different
point, we expect to find out difference between two results of projections. If they have no difference, the fixed effect
is estimable.
"""
import numpy as np
import pandas as pd

from FixedEffectModel.Operation import do_operation


def is_estimable(data_df, b_x, category_col, formula, index_name):
    alpha_df = projection2df(data_df, b_x, category_col, index_name, startpoint=0)
    alpha_df1 = projection2df(data_df, b_x, category_col, index_name, startpoint=1)
    result = do_operation(alpha_df, formula)
    result1 = do_operation(alpha_df1, formula)
    if abs(result - result1) > 1e-5:
        return False
    else:
        return True


def projection2df(data_df, b_x, category_col, index_name, epsilon=1e-5, startpoint=0):
    category_unique_val = []
    length_list = []
    e = len(category_col)
    x_dim = 0
    for i in range(e):
        m = np.unique(data_df[category_col[i]].values)
        category_unique_val.append(m)
        length_list.append(m.shape[0])
        x_dim += m.shape[0]

    n = data_df.shape[0]
    max_iter = 2 * np.log(epsilon) / np.log(1 - 1 / (10 * min([x_dim, n])))
    # print('max_iter:', max_iter)
    fe = data_df[category_col].values
    d_i_index = np.zeros(e, dtype=np.int)
    if startpoint == 0:
        x = np.zeros(x_dim, dtype=np.float64)
    else:
        x = np.ones(x_dim, dtype=np.float64)

    x_cache = np.zeros(x_dim, dtype=np.float64)
    loop_index = np.zeros((n, e), dtype=np.int)
    update = 10
    iter_loops = int((max_iter // n) + 1)
    # print('max_whole_iter:', iter_loops)
    # index = np.zeros()
    for loop in range(iter_loops):
        if loop == 0:
            for i in range(n):
                d_ij = 0
                for j in range(e):
                    index = np.where(category_unique_val[j] == fe[i][j])
                    # print(index)
                    d_i_index[j] = d_ij + index[0][0]
                    d_ij += length_list[j]
                loop_index[i] = d_i_index
                update_value = (np.sum(x[d_i_index]) - b_x[i]) / e
                x[d_i_index] -= update_value
            x_cache = x.copy()
        else:
            prob_array = np.random.choice(n, n)
            for i in range(n):
                d_i_index = loop_index[prob_array[i]]
                update_value = (np.sum(x[d_i_index]) - b_x[prob_array[i]]) / e
                x[d_i_index] -= update_value
            update = np.linalg.norm(x - x_cache)
            # print(update)
            if update < epsilon:
                # print('not exceed max iteration')
                break
            x_cache = x.copy()
    result_df = pd.Series(x, index=index_name)
    return result_df



