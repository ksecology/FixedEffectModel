"""
This module is used to get standard error of each fixed effect. Two functions are concluded:
Function alpha_std0 is a traditional function to calculate std err.
Function alpha_std uses bootstrap method and allow 50 calculations at the same time, which is faster than alpha-std0.
Thus we recommend using alpha_std.
"""

import numpy as np
import statsmodels.api as sm
from multiprocessing import Pool
from FixedEffectModel.Bootstrap import bootstrap
from FixedEffectModel.DemeanDataframe import demean_dataframe
from FixedEffectModel.EstimableCheck import is_estimable, projection2df
from FixedEffectModel.Operation import do_operation


def alpha_std0(result, formula, sample_num=100):
    """

    :param result: result of model using function ols_high_d_category
    :param formula:  equation of relative effect of two fixed variables, like "id_1 - id_2"
    :param sample_num: number of samples
    :return: estimation of relative effect of two fixed variables and its standard error
    """

    data_df = result.data_df
    demean_df = result.demeaned_df
    coeff = result.params.values
    consist_col = result.consist_col
    old_x = result.old_x
    category_col = result.category_col
    out_col = result.out_col
    index_name = []
    e = len(category_col)
    for i in range(e):
        m = np.unique(data_df[category_col[i]].values)
        for l in m:
            name = category_col[i] + str(l)
            index_name.append(name)
    copy_list = old_x.copy()
    copy_list.extend(category_col)
    alpha = np.zeros(sample_num, dtype=np.float64)
    n = data_df.shape[0]
    new_df = data_df[copy_list].copy()
    y_pred = data_df[out_col[0]].values - demean_df['resid'].values
    y = data_df[out_col[0]]
    b_x = np.dot(coeff, data_df[old_x].values.T)
    ori_resid = y - b_x
    true_resid = ori_resid - demean_df['resid']
    true_alpha = projection2df(new_df, true_resid, category_col, index_name)
    demeaned_resid = demean_df['resid'].values
    final_result = do_operation(true_alpha, formula)
    ori_x = new_df[old_x].values.T
    # print(final_result)
    if not is_estimable(new_df, true_resid, category_col, formula, index_name):
        print('the function you defined is not estimable')
    else:
        for i in range(sample_num):
            sample_resid = np.random.choice(demeaned_resid, n)
            y_new = y_pred + sample_resid
            new_df['y_new'] = y_new
            demeaned_new = demean_dataframe(new_df, ['y_new'], category_col)
            model = sm.OLS(demeaned_new['y_new'], demean_df[old_x])
            result = model.fit()
            y = new_df['y_new'].values
            b_x = np.dot(result.params.values, ori_x)
            b_array = y - b_x
            pb_array = result.resid
            target_array = b_array - pb_array
            alpha_df = projection2df(new_df, target_array, category_col, index_name)
            result = do_operation(alpha_df, formula)
            alpha[i] = result
    return 'est:' + str(final_result), 'std:' + str(np.std(alpha))


def alpha_std(result, formula, sample_num=100):
    """

    :param result: result of model using function ols_high_d_category
    :param formula: equation of relative effect of two fixed variables, like "id_1 - id_2"
    :param sample_num: number of samples
    :return: estimation of relative effect of two fixed variables and its standard error
    """

    data_df = result.data_df
    demean_df = result.demeaned_df
    coeff = result.params.values
    consist_col = result.consist_col
    old_x = result.old_x
    category_col = result.category_col
    out_col = result.out_col

    index_name = []
    e = len(category_col)
    for i in range(e):
        m = np.unique(data_df[category_col[i]].values)
        for l in m:
            name = category_col[i] + '_' + str(l)
            index_name.append(name)
    copy_list = old_x.copy()
    copy_list.extend(category_col)
    alpha = np.zeros(sample_num, dtype=np.float64)
    n = data_df.shape[0]
    new_df = data_df[copy_list].copy()
    y_pred = data_df[out_col[0]].values - demean_df['resid'].values
    y = data_df[out_col[0]]
    b_x = np.dot(coeff, data_df[old_x].values.T)
    ori_resid = y - b_x
    true_resid = ori_resid - demean_df['resid']
    true_alpha = projection2df(new_df, true_resid, category_col, index_name)
    demeaned_resid = demean_df['resid'].values
    final_result = do_operation(true_alpha, formula)
    ori_x = new_df[old_x].values.T
    #     print(final_result)
    if not is_estimable(new_df, true_resid, category_col, formula, index_name):
        print('the function you defined is not estimable')
    else:
        print(formula)
        pool = Pool(processes=50)
        alpha_result = []
        for i in range(sample_num):
            alpha_result.append(pool.apply_async(bootstrap, args=(new_df, demeaned_resid, y_pred, n, category_col,
                                                                  demean_df, old_x, formula, index_name, i)))
        pool.close()
        pool.join()
        for i in range(len(alpha_result)):
            alpha[i] = alpha_result[i].get()
        return 'est:' + str(final_result), 'std:' + str(np.std(alpha))
