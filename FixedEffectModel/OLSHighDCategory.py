from statsmodels.compat import lrange
from statsmodels.iolib import SimpleTable
from FixedEffectModel.DemeanDataframe import demean_dataframe
from FixedEffectModel.FormTransfer import form_transfer
from FixedEffectModel.OLSFixed import OLSFixed
from FixedEffectModel.RobustErr import robust_err
from FixedEffectModel.ClusterErr import *
from FixedEffectModel.CalDf import cal_df
from FixedEffectModel.CalFullModel import cal_fullmodel
from FixedEffectModel.Forg import forg
from FixedEffectModel.WaldTest import waldtest
import statsmodels.api as sm
from scipy.stats import t
from scipy.stats import f
import time
import numpy as np
import pandas as pd


def ols_high_d_category(data_df, consist_input=None, out_input=None, category_input=None, cluster_input=[],
                        fake_x_input=[], iv_col_input=[],
                        formula=None, robust=False, c_method='cgm', psdef=True, epsilon=1e-8, max_iter=1e6, process=5):
    """

    :param fake_x_input: List of endogenous variables
    :param iv_col_input: List of instrument variables
    :param data_df: Dataframe of relevant data
    :param consist_input: List of continuous variables
    :param out_input: List of dependent variables(so far, only support one dependent variable)
    :param category_input: List of category variables(fixed effects)
    :param cluster_input: List of cluster variables
    :param formula: a string like 'y~x+x2|id+firm|id',dependent_variable~continuous_variable|fixed_effect|clusters
    :param robust: bool value of whether to get a robust variance
    :param c_method: method used to calculate multi-way clusters variance. Possible choices are:
            - 'cgm'
            - 'cgm2'
    :param psdef:if True, replace negative eigenvalue of variance matrix with 0 (only in multi-way clusters variance)
    :param epsilon: tolerance of the demean process
    :param max_iter: max iteration of the demean process
    :param process: number of process in multiprocessing(only in multi-way clusters variance calculating)
    :return:params,df,bse,tvalues,pvalues,rsquared,rsquared_adj,fvalue,f_pvalue,variance_matrix,fittedvalues,resid,summary
    """

    if (consist_input is None) & (formula is None):
        raise NameError('You have to input list of variables name or formula')
    elif consist_input is None:
        out_col, consist_col, category_col, cluster_col, fake_x, iv_col = form_transfer(formula)
        print('dependent variable(s):', out_col)
        print('continuous variables:', consist_col)
        print('category variables(fixed effects):', category_col)
        print('cluster variables:', cluster_col)
        if fake_x:
            print('endogenous variables:', fake_x)
            print('instruments:', iv_col)
    else:
        out_col, consist_col, category_col, cluster_col, fake_x, iv_col = out_input, consist_input, category_input, \
                                                                          cluster_input, fake_x_input, iv_col_input

    if category_col[0] == '0' or category_col == []:
        demeaned_df = data_df.copy()
        # const_consist = sm.add_constant(demeaned_df[consist_col])
        # # print(consist_col)
        # consist_col = ['const'] + consist_col
        # demeaned_df['const'] = const_consist['const']
        # print('Since the model does not have fixed effect, add an intercept.')
        rank = 0
    else:
        consist_var = []
        for i in consist_col:
            consist_var.append(i)
        for i in fake_x:
            consist_var.append(i)
        for i in iv_col:
            consist_var.append(i)
        consist_var.append(out_col[0])
        start = time.time()
        demeaned_df = demean_dataframe(data_df, consist_var, category_col, epsilon, max_iter)
        end = time.time()
        print('demean time:', forg((end - start), 4), 's')
        start = time.process_time()
        rank = cal_df(data_df, category_col)
        end = time.process_time()
        print('time used to calculate degree of freedom of category variables:', forg((end - start), 4), 's')
        print('degree of freedom of category variables:', rank)

    all_exo_x = consist_col + iv_col
    iv_model = []
    iv_result = []
    hat_fake_x = []
    if category_col[0] == '0' or category_col == []:
        const_all_exo_x = sm.add_constant(demeaned_df[all_exo_x])
        all_exo_x = ['iv_const'] + all_exo_x
        demeaned_df['iv_const'] = const_all_exo_x['const']
        print('Since the model does not have fixed effect, add an intercept during stage1 calculation.')

    for i in fake_x:
        model_iv = sm.OLS(demeaned_df[i], demeaned_df[all_exo_x])
        result_iv = model_iv.fit()
        iv_model.append(model_iv)
        iv_result.append(result_iv)
        iv_coeff = result_iv.params.values
        demeaned_df["hat_" + i] = np.dot(iv_coeff, demeaned_df[all_exo_x].values.T)
        hat_fake_x.append("hat_" + i)

    new_x = consist_col + hat_fake_x
    old_x = consist_col + fake_x

    if category_col[0] == '0' or category_col == []:
        const_new_x = sm.add_constant(demeaned_df[new_x])
        new_x = ['const'] + new_x
        old_x = ['const'] + old_x
        demeaned_df['const'] = const_new_x['const']

    model = sm.OLS(demeaned_df[out_col], demeaned_df[new_x])
    result = model.fit()
    coeff = result.params.values.reshape(len(new_x), 1)
    real_resid = demeaned_df[out_col] - np.dot(demeaned_df[old_x], coeff)
    demeaned_df['resid'] = real_resid

    n = demeaned_df.shape[0]
    k = len(new_x)
    f_result = OLSFixed()
    f_result.out_col = out_col
    f_result.consist_col = new_x
    f_result.old_x = old_x
    f_result.category_col = category_col
    f_result.data_df = data_df.copy()
    f_result.demeaned_df = demeaned_df
    f_result.params = result.params
    f_result.df = result.df_resid - rank

    if (len(cluster_col) == 0 or cluster_col[0] == '0') & (robust is False):
        std_error = result.bse * np.sqrt((n - k) / (n - k - rank))
        covariance_matrix = result.normalized_cov_params * result.scale * result.df_resid / f_result.df
    elif (len(cluster_col) == 0) & (robust is True):
        start = time.process_time()
        covariance_matrix = robust_err(demeaned_df, consist_col, n, k, rank)
        end = time.process_time()
        print('time used to calculate robust covariance matrix:', forg((end - start), 4), 's')
        std_error = np.sqrt(np.diag(covariance_matrix))
    else:
        if category_col[0] == '0' or category_col == []:
            nested = False
        else:
            start = time.process_time()
            nested = is_nested(demeaned_df, category_col, cluster_col, consist_col)
            end = time.process_time()
            print('category variable(s) is_nested in cluster variables:', nested)
            print('time used to define nested or not:', end - start)

        # if nested or c_method != 'cgm':
        #     f_result.df = min(min_clust(data_df, cluster_col) - 1, f_result.df)

        start = time.process_time()
        covariance_matrix = clustered_error(demeaned_df, new_x, cluster_col, n, k, rank, nested=nested,
                                            c_method=c_method, psdef=psdef)
        end = time.process_time()
        print('time used to calculate clustered covariance matrix:', forg((end - start), 4), 's')
        std_error = np.sqrt(np.diag(covariance_matrix))

    f_result.bse = std_error
    f_result.resid = demeaned_df['resid']
    f_result.variance_matrix = covariance_matrix
    f_result.tvalues = f_result.params / f_result.bse
    f_result.pvalues = pd.Series(2 * t.sf(np.abs(f_result.tvalues), f_result.df), index=list(result.params.index))
    proj_rss = sum(f_result.resid ** 2)
    proj_tss = sum(((demeaned_df[out_col] - demeaned_df[out_col].mean()) ** 2).values)[0]

    # proj_fvalue = (proj_tss - sum(result.resid**2)) * (len(data_df) - len(new_x) - rank) / (proj_rss * (rank + len(new_x) - 1))

    f_result.rsquared = 1-proj_rss/proj_tss
    f_result.rsquared_adj = 1 - (len(data_df) - 1) / (result.df_resid - rank) * (1 - f_result.rsquared)
    # tmp1 = np.linalg.solve(f_result.variance_matrix, np.mat(f_result.params).T)
    # tmp2 = np.dot(np.mat(f_result.params), tmp1)
    # f_result.fvalue = tmp2[0, 0] / result.df_model
    # f_result.fvalue = proj_fvalue
    w = waldtest(f_result.params, f_result.variance_matrix)
    f_result.fvalue = w/result.df_model

    if len(cluster_col) > 0 and c_method == 'cgm':
        f_result.f_pvalue = f.sf(f_result.fvalue, result.df_model,
                                 min(min_clust(data_df, cluster_col) - 1, f_result.df))
        f_result.f_df_proj = [result.df_model, (min(min_clust(data_df, cluster_col) - 1, f_result.df))]
    else:
        f_result.f_pvalue = f.sf(f_result.fvalue, result.df_model, f_result.df)
        f_result.f_df_proj = [result.df_model, f_result.df]

    # std err=diag( np.sqrt(result.normalized_cov_params*result.scale*result.df_resid/f_result.df) )
    f_result.fittedvalues = result.fittedvalues
    f_result.full_rsquared, f_result.full_rsquared_adj, f_result.full_fvalue, f_result.full_f_pvalue, f_result.f_df_full \
        = cal_fullmodel(data_df, out_col, new_x, rank, RSS=sum(f_result.resid ** 2), originRSS=sum(result.resid ** 2))
    f_result.nobs = result.nobs
    f_result.yname = out_col
    f_result.xname = new_x
    f_result.resid_std_err = np.sqrt(sum(f_result.resid ** 2) / (result.df_resid - rank))
    if len(cluster_col) == 0:
        f_result.cluster_method = 'no_cluster'
        if robust:
            f_result.Covariance_Type = 'robust'
        else:
            f_result.Covariance_Type = 'nonrobust'
    else:
        f_result.cluster_method = c_method
        f_result.Covariance_Type = 'clustered'
    return f_result  # , demeaned_df


def ols_high_d_category_multi_results(data_df, models, table_header):
    """
    This function is used to get multi results of multi models on one dataframe. During analyzing data with large data
    size and complicated, we usually have several model assumptions. By using this function, we can easily get the
    results comparison of the different models.

    :param data_df: Dataframe with relevant data
    :param models: List of models
    :param table_header: Title of summary table
    :return: summary table of results of the different models
    """
    results = []
    for model1 in models:
        results.append(ols_high_d_category(data_df,
                                           model1['consist_input'],
                                           model1['out_input'],
                                           model1['category_input'],
                                           model1['cluster_input'],
                                           model1['fake_x_input'],
                                           model1['iv_col_input'],
                                           formula=None,
                                           robust=False,
                                           c_method='cgm',
                                           epsilon=1e-5,
                                           max_iter=1e6))
    consist_name_list = [result.params.index.to_list() for result in results]
    consist_name_total = []
    consist_name_total.extend(consist_name_list[0])
    for i in consist_name_list[1:]:
        for j in i:
            if j not in consist_name_total:
                consist_name_total.append(j)
    index_name = []
    for name in consist_name_total:
        index_name.append(name)
        index_name.append('pvalue')
        index_name.append('std err')
    exog_len = lrange(len(results))
    lzip = []
    y_zip = []
    b_zip = np.zeros(5)
    table_content = []
    for name in consist_name_total:
        coeff_list = []
        pvalue_list = []
        std_list = []
        for i in range(len(results)):
            if name in consist_name_list[i]:
                coeff = "%#7.4g" % (results[i].params[name])
                pvalue = "%#8.2g" % (results[i].pvalues[name])
                std = "%#8.2f" % (results[i].bse[consist_name_list[i].index(name)])
                coeff_list.append(coeff)
                pvalue_list.append(pvalue)
                std_list.append(std)
            else:
                coeff = 'Nan'
                pvalue = 'Nan'
                std = 'Nan'
                coeff_list.append(coeff)
                pvalue_list.append(pvalue)
                std_list.append(std)
        table_content.append(tuple(coeff_list))
        table_content.append(tuple(pvalue_list))
        table_content.append(tuple(std_list))
    wtffff = dict(
        fmt='txt',
        # basic table formatting
        table_dec_above='=',
        table_dec_below='-',
        title_align='l',
        # basic row formatting
        row_pre='',
        row_post='',
        header_dec_below='-',
        row_dec_below=None,
        colwidths=None,
        colsep=' ',
        data_aligns="l",
        # data formats
        # data_fmt="%s",
        data_fmts=["%s"],
        # labeled alignments
        # stubs_align='l',
        stub_align='l',
        header_align='r',
        # labeled formats
        header_fmt='%s',
        stub_fmt='%s',
        header='%s',
        stub='%s',
        empty_cell='',
        empty='',
        missing='--',
    )
    a = SimpleTable(table_content,
                    table_header,
                    index_name,
                    title='multi',
                    txt_fmt=wtffff)
    print(a)
