from statsmodels.compat import lrange
from statsmodels.iolib import SimpleTable
from .DemeanDataframe import demean_dataframe,demeanonex
from .FormTransfer import form_transfer
from .OLSFixed import OLSFixed
from .RobustErr import robust_err
from .ClusterErr import *
from .CalDf import cal_df
from .CalFullModel import cal_fullmodel
from .Forg import forg
from .WaldTest import waldtest
from .ivtest import ivtest

import statsmodels.api as sm
from scipy.stats import t
from scipy.stats import f

import time
import numpy as np
import pandas as pd
import warnings

#11/24/2020 aa
from .GenCrossProd import gencrossprod_dataset
from .GenCrossProd import gencrossprod

#most updated version 0.0.3,example release second trial
def ols_high_d_category(data_df, 
                        consist_input = None, 
                        out_input = None, 
                        category_input = [], 
                        cluster_input = [],
                        fake_x_input = [], 
                        iv_col_input = [], 
                        treatment_input = None,
                        formula = None, 
                        robust = False, 
                        c_method = 'cgm', 
                        psdef = True, 
                        epsilon = 1e-8, 
                        max_iter = 1e6, 
                        process = 5,
                        noint = False,
                        **kwargs):
    """
    :param fake_x_input: List of endogenous variables
    :param iv_col_input: List of instrument variables
    :param data_df: Dataframe of relevant data
    :param consist_input: List of continuous variables
    :param out_input: List of dependent variables(so far, only support one dependent variable)
    :param category_input: List of category variables(fixed effects)
    :param cluster_input: List of cluster variables
    :param treatment_input: List of treatment input
    :param formula: a string like 'y~x+x2|id+firm|id',dependent_variable~continuous_variable|fixed_effect|clusters
    :param robust: bool value of whether to get a robust variance
    :param c_method: method used to calculate multi-way clusters variance. Possible choices are:
            - 'cgm'
            - 'cgm2'
    :param psdef:if True, replace negative eigenvalue of variance matrix with 0 (only in multi-way clusters variance)
    :param epsilon: tolerance of the demean process
    :param max_iter: max iteration of the demean process
    :param process: number of process in multiprocessing(only in multi-way clusters variance calculating)
    :param noint: force no intercept option
    :return:params,df,bse,tvalues,pvalues,rsquared,rsquared_adj,fvalue,f_pvalue,variance_matrix,fittedvalues,resid,summary
    **kwargs:some hidden option not supposed to be used by user
    """
    # initialize opt in kwargs
    no_print = False
    for key, value in kwargs.items():
        #print("{0} = {1}".format(key, value))
        if key == 'no_print':
            if value == True:
                no_print = True

    ###############################    Parse grammar and pre-processing data   ###########################################
    if noint is True:
        k0 = 0
    else:
        k0 = 1
            
    if (consist_input is None) & (formula is None):
        raise NameError('You have to input list of variables name or formula')
    elif consist_input is None:
        out_col, consist_col, category_col, cluster_col, fake_x, iv_col = form_transfer(formula)
        print('dependent variable(s):', out_col)
        print('independent(exogenous):', consist_col)
        print('category variables(fixed effects):', category_col)
        print('cluster variables:', cluster_col)
        if fake_x:
            print('endogenous variables:', fake_x)
            print('instruments:', iv_col)
    else:
        out_col, consist_col, category_col, cluster_col, fake_x, iv_col = out_input, consist_input, category_input, \
                                                                          cluster_input, fake_x_input, iv_col_input
               
    #11/24/2020: process cross product terms
    data_df = gencrossprod(data_df, consist_col)

    #11/24/2020: generate DID -- time dummies 
    # and its cross product with treatment, then update the 
    # exogenous variables list consist_col with the newly generated variables
    #03/24/2021: update DID grammar
    if (treatment_input != None):
        if len(category_col) != 2:
            raise NameError('The length of category_input should be 2, first input csid then tsid.')

        default_did_opt = {'treatment_col':'treatment',
                           'exp_date': None,
                           'effect': 'group'}

        for key in treatment_input:
            if key in default_did_opt:
                if no_print == False:
                    print("You've updated default treatment input: ", key)
                default_did_opt[key] = treatment_input[key]

        data_df = gencrossprod_dataset(data_df, out_col, consist_col, category_col, default_did_opt, no_print = no_print)

        treatment_col = default_did_opt['treatment_col']
        exp_date = default_did_opt['exp_date']
        did_effect = default_did_opt['effect']

        #did with group fixed effect
        if default_did_opt['effect']=='group':
            consist_col = consist_col + [str(treatment_col) + "*post_experiment"] + [default_did_opt['treatment_col']]
        #did with individual effect
        else:
            consist_col = consist_col + [str(treatment_col) + "*post_experiment"]
    else:
        default_did_opt = None



    #if OLS on level data:
    if (category_col == []):    
        demeaned_df = data_df.copy()
        if noint is False:
            demeaned_df['const'] = 1
        rank = 0        
    #if OLS on demean data:
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
        #2021/05/16: bug fix: did has two category_col input, but takes the second only if effect = group
        if (treatment_input != None) and (default_did_opt['effect']=='group'):
            demeaned_df = demean_dataframe(data_df, consist_var, [category_col[1]], epsilon, max_iter)
        else:
            demeaned_df = demean_dataframe(data_df, consist_var, category_col, epsilon, max_iter)

            # 11/27/2020
        if noint is False:
            for i in consist_var:
               demeaned_df[i] = demeaned_df[i].add(data_df[i].mean())
            
            demeaned_df['const'] = 1
        
        end = time.time()
        if no_print==False:
            print('demean time:', forg((end - start), 4), 's')
        start = time.process_time()
        rank = cal_df(data_df, category_col)
        end = time.process_time()
        if no_print==False:
            print('time used to calculate degree of freedom of category variables:', forg((end - start), 4), 's')
            print('degree of freedom of category variables:', rank)
            print(consist_col)
    all_exo_x = consist_col + iv_col
    iv_model = []
    iv_result = []
    hat_fake_x = []

    ###############################    First stage regression if IV   ###########################################        
    #if OLS on raw data:
    if noint is False:
        const_all_exo_x = sm.add_constant(demeaned_df[all_exo_x], has_constant='add')        
        all_exo_x = ['iv_const'] + all_exo_x
        demeaned_df['iv_const'] = const_all_exo_x['const']

        
    f_stat_first_stage = []   
    f_stat_first_stage_pval = []   
    
    for i in fake_x:        
        model_iv = sm.OLS(demeaned_df[i], demeaned_df[all_exo_x])
        result_iv = model_iv.fit()
        iv_model.append(model_iv)
        iv_result.append(result_iv)
        iv_coeff = result_iv.params.values
        demeaned_df["hat_" + i] = np.dot(iv_coeff, demeaned_df[all_exo_x].values.T)
        hat_fake_x.append("hat_" + i)
        
        f_stat_first_stage.append(result_iv.fvalue)
        f_stat_first_stage_pval.append(result_iv.f_pvalue)
        
    new_x = consist_col + hat_fake_x
    old_x = consist_col + fake_x

    #############           Second stage regression if IV/ Regular regression           ###################       
    #if OLS on raw data:
    if noint is False:
        const_new_x = sm.add_constant(demeaned_df[new_x])
        #demeaned_df['const'] = 1
        
        old_x = ['const'] + old_x
        new_x = ['const'] + new_x


    model = sm.OLS(demeaned_df[out_col].astype(float), demeaned_df[new_x].astype(float))
    result = model.fit()
    coeff = result.params.values.reshape(len(new_x), 1)


    real_resid = demeaned_df[out_col] - np.dot(demeaned_df[old_x], coeff)
    demeaned_df['resid'] = real_resid

    n = demeaned_df.shape[0]
    k = len(new_x)
    f_result = OLSFixed()
    f_result.out_col      = out_col
    f_result.consist_col  = new_x
    f_result.old_x        = old_x
    f_result.fake_x       = fake_x
    f_result.iv_col       = iv_col    
    f_result.category_col = category_col
    f_result.data_df = data_df.copy()
    f_result.demeaned_df = demeaned_df
    f_result.params = result.params
    f_result.df = result.df_resid - rank + k0
    f_result.treatment_input = treatment_input 
        
    
    if len(fake_x)>0:
        f_result.f_stat_first_stage = f_stat_first_stage
        f_result.f_stat_first_stage_pval = f_stat_first_stage_pval
    
        
    if (len(cluster_col) == 0 or cluster_col[0] == '0') & (robust is False): 
        if (len(category_col) == 0 ):
            std_error = result.bse * np.sqrt((n - k ) / (n - k - rank))#for pooled regression
        else:
            std_error = result.bse * np.sqrt((n - k ) / (n - k + k0 - rank))#for fe if k0=1 need to add it back
        covariance_matrix = result.normalized_cov_params * result.scale * result.df_resid / f_result.df
    elif (len(cluster_col) == 0 or cluster_col[0] == '0') & (robust is True):
        start = time.process_time()
        covariance_matrix = robust_err(demeaned_df, new_x, category_col,n, k, k0, rank)
        end = time.process_time()
        print('time used to calculate robust covariance matrix:', forg((end - start), 4), 's')
        std_error = np.sqrt(np.diag(covariance_matrix))
    else:
        if category_col == []:
            nested = False
        else:
            start = time.process_time()
            nested = is_nested(demeaned_df, category_col, cluster_col, consist_col)
            end = time.process_time()
            print('category variable(s) is_nested in cluster variables:', nested)
            print('time used to define nested or not:', forg((end - start), 4), 's')

        start = time.process_time()        
        covariance_matrix = clustered_error(demeaned_df, new_x, category_col, cluster_col, n, k, k0, rank, nested=nested,
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
    proj_rss = float("{:.8f}".format(proj_rss))#round up
    
    #calculate totoal sum squared of error
    if k0==0 and category_col==[]:
        #if OLS and no intercept, tss = sum(y^2)
        proj_tss = sum(((demeaned_df[out_col]) ** 2).values)[0]  
    else:
        proj_tss = sum(((demeaned_df[out_col] - demeaned_df[out_col].mean()) ** 2).values)[0]               
                
    proj_tss = float("{:.8f}".format(proj_tss))#round up
    if proj_tss>0:
        f_result.rsquared = 1 - proj_rss / proj_tss
    else:
        raise NameError('Total sum of square equal 0, program quit.')
    
    #calculate adjusted r2
    if category_col != []:   
        #for fixed effect, k0 should not affect adjusted r2
        f_result.rsquared_adj = 1 - (len(data_df) - 1) / (result.df_resid - rank + k0) * (1 - f_result.rsquared) 
    else:        
        f_result.rsquared_adj = 1 - (len(data_df) - k0) / (result.df_resid) * (1 - f_result.rsquared) 
    
    if k0 == 0:        
        w = waldtest(f_result.params, f_result.variance_matrix)        
    else:
        #get rid of constant in the vc matrix
        f_var_mat_noint = f_result.variance_matrix.copy()
        if type(f_var_mat_noint) == np.ndarray:
            f_var_mat_noint = np.delete(f_var_mat_noint, 0, 0)
            f_var_mat_noint = np.delete(f_var_mat_noint, 0, 1)
        else:
            f_var_mat_noint = f_var_mat_noint.drop('const',axis = 1)
            f_var_mat_noint = f_var_mat_noint.drop('const',axis = 0)
        
        #get rid of constant in the param column
        params_noint = f_result.params.drop('const',axis = 0)
        if category_col == []:
            w = waldtest(params_noint, (n - k ) / (n - k - rank)*f_var_mat_noint)
        else:
            w = waldtest(params_noint, f_var_mat_noint)
        
    #calculate f-statistics
    if result.df_model > 0:
        #if do pooled regression
        if category_col == []:
            #if do pooled regression, because doesn't account for const in f test, adjust dof
            scale_const = (n - k ) / (n - k + k0)
            f_result.fvalue =scale_const* w / result.df_model
        #if do fixed effect, just ignore    
        else:            
            f_result.fvalue = w / result.df_model
    else:    
        f_result.fvalue = 0
        
        
    if len(cluster_col) > 0 and cluster_col[0] != '0' and c_method == 'cgm':
        f_result.f_pvalue = f.sf(f_result.fvalue, result.df_model,
                                 min(min_clust(data_df, cluster_col) - 1, f_result.df))
        f_result.f_df_proj = [result.df_model, (min(min_clust(data_df, cluster_col) - 1, f_result.df))]
    else:
        f_result.f_pvalue = f.sf(f_result.fvalue, result.df_model, f_result.df)
        f_result.f_df_proj = [result.df_model, f_result.df]

    # std err=diag( np.sqrt(result.normalized_cov_params*result.scale*result.df_resid/f_result.df) )
    f_result.fittedvalues = result.fittedvalues
    
    # get full-model related statistics
    f_result.full_rsquared, f_result.full_rsquared_adj, f_result.full_fvalue, f_result.full_f_pvalue, f_result.f_df_full \
        = cal_fullmodel(data_df, out_col, new_x, category_col, rank, RSS=sum(f_result.resid ** 2), originRSS = sum(result.resid** 2))
        
    f_result.nobs = result.nobs
    f_result.yname = out_col
    f_result.xname = new_x
    f_result.resid_std_err = np.sqrt(sum(f_result.resid ** 2) / (result.df_resid - rank))
    if len(cluster_col) == 0 or cluster_col[0]=='0':
        f_result.cluster_method = 'no_cluster'
        if robust:
            f_result.Covariance_Type = 'robust'
        else:
            f_result.Covariance_Type = 'nonrobust'
    else:
        f_result.cluster_method = c_method
        f_result.Covariance_Type = 'clustered'

    # record some original input stuff for use of causal_engine
    f_result.consist_input = consist_input

    return f_result  

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

        consist_input=None 
        out_input=None 
        category_input=[] 
        cluster_input=[]                        
        fake_x_input=[] 
        iv_col_input=[] 
        treatment_input=[]                        
        formula=None     
        robust = False
        c_method = 'cgm'
        psdef = True
        epsilon = 1e-8
        max_iter = 1e6
        process = 5
        noint = False    

        if 'consist_input' in model1:
            consist_input = model1['consist_input'] 

        if model1['out_input'] != []:
            out_input = model1['out_input'] 

        if model1['category_input'] != []:
            category_input = model1['category_input'] 

        if 'cluster_input' in model1:        
            cluster_input = model1['cluster_input'] 

        if 'fake_x_input' in model1:        
            fake_x_input = model1['fake_x_input'] 

        if 'iv_col_input' in model1:                
            iv_col_input = model1['iv_col_input'] 

        if 'treatment_input' in model1:                
            treatment_input = model1['treatment_input'] 

        model1_robust = False      
        if 'robust' in model1: 
            model1_robust = True

        model1_c_method = 'cgm'    
        if 'c_method' in model1:     
            model1_c_method = 'cgm2'

        model1_noint = False        
        if 'noint' in model1:           
            model1_noint = True

        results.append(ols_high_d_category(data_df,
                                           consist_input,
                                           out_input,
                                           category_input,
                                           cluster_input,
                                           fake_x_input,
                                           iv_col_input,
                                           formula = None,
                                           robust = model1_robust,
                                           c_method = model1_c_method,
                                           epsilon=1e-5,
                                           max_iter=1e6,
                                           noint=model1_noint))   

        
    #    results.append(ols_high_d_category(data_df,
    #                                       model1['consist_input'],
    #                                       model1['out_input'],
    #                                       model1['category_input'],
    #                                       model1['cluster_input'],
    #                                       model1['fake_x_input'],
    #                                       model1['iv_col_input'],
    #                                       formula=None,
    #                                       robust=False,
    #                                       c_method='cgm',
    #                                       epsilon=1e-5,
    #                                       max_iter=1e6))
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
        index_name.append('std err')        
        index_name.append('pvalue')
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
                std = "%#8.4f" % (results[i].bse[consist_name_list[i].index(name)])                
                pvalue = "%#8.2g" % (results[i].pvalues[name])
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
        table_content.append(tuple(std_list))
        table_content.append(tuple(pvalue_list))
        
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
