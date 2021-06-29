import os
import numpy as np
import statsmodels.api as sm
from .DemeanDataframe import demean_dataframe
from .EstimableCheck import projection2df
from .Operation import do_operation

#bootstrap package 2.0
def bootstrap(new, demeaned_resid, y_pred, n, category_col, demean_df, consist_col, formula, index_name, i):
    """

    :param new: dataframe with continuous variables and dependent variable
    :param demeaned_resid: dataframe of residuals obtained from model on demeaned dataframe
    :param y_pred: dataframe of the prediction of y
    :param n: size of original dataframe
    :param category_col: List of category variables
    :param demean_df: demeaned dataframe with relevant data
    :param consist_col: List of continuous variables
    :param formula: equation of relative effect of two fixed variables, like "id_1 - id_2"
    :param index_name: name of category
    :param i: index
    :return:
    """
    new_df = new.copy()
    LocalProcRandGen = np.random.RandomState()
    sample_resid = LocalProcRandGen.choice(demeaned_resid, n)
    y_new = y_pred + sample_resid
    name = 'y_new' + str(i)
    new_df[name] = y_new
    demeaned_new = demean_dataframe(new_df, [name], category_col)
    model = sm.OLS(demeaned_new[name], demean_df[consist_col])
    result = model.fit()
    y = new_df[name].values
    b_x = np.dot(result.params.values, new_df[consist_col].values.T)
    b_array = y - b_x
    pb_array = result.resid
    target_array = b_array - pb_array
    alpha_df = projection2df(new_df, target_array, category_col, index_name)
    result = do_operation(alpha_df, formula)
    # print('resid：',sample_resid)
    # print('Current process：', os.getpid())
    return result
