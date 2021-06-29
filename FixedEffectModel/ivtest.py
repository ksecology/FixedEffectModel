from .Forg import forg
import pandas as pd
import numpy as np
import scipy as sp
from io import StringIO 
import warnings
from .TableFormat import gen_fmt, fmt_2
from statsmodels.iolib.table import SimpleTable
from statsmodels.compat.python import lrange, lmap, lzip
from scipy.stats import chi2
from scipy.stats import f
import statsmodels.api as sm

#2021
def ivtest(result):
    """
    """
    if result.iv_col == []:
        raise NameError('there is no iv')

    consist_col = result.consist_col
    iv_col      = result.iv_col
    out_col     = result.out_col
    fake_x      = result.fake_x
    old_x       = result.old_x

    for i in fake_x:
        if i in old_x:
            old_x.remove(i) 

    all_exo_x   = old_x + iv_col

    demeaned_df = result.demeaned_df #pass object so easier to reference
    z       = demeaned_df[iv_col].values
    y       = demeaned_df[out_col].values
    x_exog  = demeaned_df[old_x].values
    x_endog = demeaned_df[fake_x].values
    z_      = demeaned_df[all_exo_x].values

    nobs = result.demeaned_df.shape[0]
    k1 = len(old_x)
    k2 = len(iv_col)

    # x related stuff
    xpx_inv = np.linalg.inv(np.dot(x_exog.T, x_exog))
    px      = np.dot(np.dot(x_exog,xpx_inv),x_exog.T)
    m_x     = np.identity(nobs) - px
    y_proj  = np.dot(m_x, x_endog)
    z_proj  = np.dot(m_x, z)

    # z related stuff
    zpz_inv_proj = np.linalg.inv(np.dot(z_proj.T, z_proj))
    pz_proj      = np.dot(np.dot(z_proj,zpz_inv_proj),z_proj.T)

    zpz_full     = np.dot(z_.T, z_)
    zpz_inv_full = np.linalg.inv(zpz_full)
    pz_full      = np.dot(np.dot(z_,zpz_inv_full),z_.T)
    m_z_full     = np.identity(nobs) - pz_full
    sigma_vv     = np.dot(np.dot(x_endog.T,m_z_full),x_endog)/(nobs - k1 - k2)

    sigma_vv_inv_sqrt = np.linalg.inv(sp.linalg.sqrtm(sigma_vv))


    fstat_matrix_meat = np.dot(np.dot(y_proj.T, pz_proj),y_proj)
    fstat_matrix = np.dot(np.dot(sigma_vv_inv_sqrt.T, fstat_matrix_meat),sigma_vv_inv_sqrt)/k2

    cd_stat = min(np.linalg.eigvals(fstat_matrix))

    cd_stat = round(cd_stat,6)
    
    # critical value in stock and yogo 2005    
    tab_5_2 = u"""\
    k2_tab,0.1,0.15,0.2,0.25,0.1,0.15,0.2,0.25
    1,16.38,8.96,6.66,5.53,,,,
    2,19.93,11.59,8.75,7.25,7.03,4.58,3.95,3.63
    3,22.3,12.83,9.54,7.8,13.43,8.18,6.4,5.45
    4,24.58,13.96,10.26,8.31,16.87,9.93,7.54,6.28
    5,26.87,15.09,10.98,8.84,19.45,11.22,8.38,6.89
    6,29.18,16.23,11.72,9.38,21.68,12.33,9.1,7.42
    7,31.5,17.38,12.48,9.93,23.72,13.34,9.77,7.91
    8,33.84,18.54,13.24,10.5,25.64,14.31,10.41,8.39
    9,36.19,19.71,14.01,11.07,27.51,15.24,11.03,8.85
    10,38.54,20.88,14.78,11.65,29.32,16.16,11.65,9.31
    11,40.9,22.06,15.56,12.23,31.11,17.06,12.25,9.77
    12,43.27,23.24,16.35,12.82,32.88,17.95,12.86,10.22
    13,45.64,24.42,17.14,13.41,34.62,18.84,13.45,10.68
    14,48.01,25.61,17.93,14,36.36,19.72,14.05,11.13
    15,50.39,26.8,18.72,14.6,38.08,20.6,14.65,11.58
    16,52.77,27.99,19.51,15.19,39.8,21.48,15.24,12.03
    17,55.15,29.19,20.31,15.79,41.51,22.35,15.83,12.49
    18,57.53,30.38,21.1,16.39,43.22,23.22,16.42,12.94
    19,59.92,31.58,21.9,16.99,44.92,24.09,17.02,13.39
    20,62.3,32.77,22.7,17.6,46.62,24.96,17.61,13.84
    21,64.69,33.97,23.5,18.2,48.31,25.82,18.2,14.29
    22,67.07,35.17,24.3,18.8,50.01,26.69,18.79,14.74
    23,69.46,36.37,25.1,19.41,51.7,27.56,19.38,15.19
    24,71.85,37.57,25.9,20.01,53.39,28.42,19.97,15.64
    25,74.24,38.77,26.71,20.61,55.07,29.29,20.56,16.1
    26,76.62,39.97,27.51,21.22,56.76,30.15,21.15,16.55
    27,79.01,41.17,28.31,21.83,58.45,31.02,21.74,17
    28,81.4,42.37,29.12,22.43,60.13,31.88,22.33,17.45
    29,83.79,43.57,29.92,23.04,61.82,32.74,22.92,17.9
    30,86.17,44.78,30.72,23.65,63.51,33.61,23.51,18.35  """

    
    #not used for now. may add in future
    tab_5_1 = u"""\
    k2_tab,0.05,0.1,0.2,0.3,0.05,0.1,0.2,0.3,0.05,0.1,0.2,0.3
    3,13.91,9.08,6.46,5.39,,,,,,,,
    4,16.85,10.27,6.71,5.34,11.04,7.56,5.57,4.73,,,,
    5,18.37,10.83,6.77,5.25,13.97,8.78,5.91,4.79,9.53,6.61,4.99,4.3
    6,19.28,11.12,6.76,5.15,15.72,9.48,6.08,4.78,12.2,7.77,5.35,4.4
    7,19.86,11.29,6.73,5.07,16.88,9.92,6.16,4.76,13.95,8.5,5.56,4.44
    8,20.25,11.39,6.69,4.99,17.7,10.22,6.2,4.73,15.18,9.01,5.69,4.46
    9,20.53,11.46,6.65,4.92,18.3,10.43,6.22,4.69,16.1,9.37,5.78,4.46
    10,20.74,11.49,6.61,4.86,18.76,10.58,6.23,4.66,16.8,9.64,5.83,4.45
    11,20.9,11.51,6.56,4.8,19.12,10.69,6.23,4.62,17.35,9.85,5.87,4.44
    12,21.01,11.52,6.53,4.75,19.4,10.78,6.22,4.59,17.8,10.01,5.9,4.42
    13,21.1,11.52,6.49,4.71,19.64,10.84,6.21,4.56,18.17,10.14,5.92,4.41
    14,21.18,11.52,6.45,4.67,19.83,10.89,6.2,4.53,18.47,10.25,5.93,4.39
    15,21.23,11.51,6.42,4.63,19.98,10.93,6.19,4.5,18.73,10.33,5.94,4.37
    16,21.28,11.5,6.39,4.59,20.12,10.96,6.17,4.48,18.94,10.41,5.94,4.36
    17,21.31,11.49,6.36,4.56,20.23,10.99,6.16,4.45,19.13,10.47,5.94,4.34
    18,21.34,11.48,6.33,4.53,20.33,11,6.14,4.43,19.29,10.52,5.94,4.32
    19,21.36,11.46,6.31,4.51,20.41,11.02,6.13,4.41,19.44,10.56,5.94,4.31
    20,21.38,11.45,6.28,4.48,20.48,11.03,6.11,4.39,19.56,10.6,5.93,4.29
    21,21.39,11.44,6.26,4.46,20.54,11.04,6.1,4.37,19.67,10.63,5.93,4.28
    22,21.4,11.42,6.24,4.43,20.6,11.05,6.08,4.35,19.77,10.65,5.92,4.27
    23,21.41,11.41,6.22,4.41,20.65,11.05,6.07,4.33,19.86,10.68,5.92,4.25
    24,21.41,11.4,6.2,4.39,20.69,11.05,6.06,4.32,19.94,10.7,5.91,4.24
    25,21.42,11.38,6.18,4.37,20.73,11.06,6.05,4.3,20.01,10.71,5.9,4.23
    26,21.42,11.37,6.16,4.35,20.76,11.06,6.03,4.29,20.07,10.73,5.9,4.21
    27,21.42,11.36,6.14,4.34,20.79,11.06,6.02,4.27,20.13,10.74,5.89,4.2
    28,21.42,11.34,6.13,4.32,20.82,11.05,6.01,4.26,20.18,10.75,5.88,4.19
    29,21.42,11.33,6.11,4.31,20.84,11.05,6,4.24,20.23,10.76,5.88,4.18
    30,21.42,11.32,6.09,4.29,20.86,11.05,5.99,4.23,20.27,10.77,5.87,4.17
    """
    tab_5_1 = StringIO(tab_5_1)
    tab_5_2 = StringIO(tab_5_2)
    df_5_1 = pd.read_csv(tab_5_1)
    df_5_2 = pd.read_csv(tab_5_2)
    
    
    N = len(fake_x)    
    critical_val = []

    if N==1:
        stat_5p  = forg(df_5_2['0.1'].iloc[k2-1],4)
        stat_10p = forg(df_5_2['0.15'].iloc[k2-1],4)
        stat_20p = forg(df_5_2['0.2'].iloc[k2-1],4)
        stat_30p = forg(df_5_2['0.25'].iloc[k2-1],4)
        critical_val = [(stat_5p, stat_10p, stat_20p, stat_30p)]
    elif N==2:
        stat_5p  = forg(df_5_2['0.1.1'].iloc[k2-1],4)
        stat_10p = forg(df_5_2['0.15.1'].iloc[k2-1],4)
        stat_20p = forg(df_5_2['0.2.1'].iloc[k2-1],4)
        stat_30p = forg(df_5_2['0.25.1'].iloc[k2-1],4)
        
        critical_val = [(stat_5p, stat_10p, stat_20p, stat_30p)]
    else:
        warnings.warn("Critical values are not provided for number of endogenous variables greater than 3")
        critical_val = [(0,0,0,0)]

        
    #-------------------------------------------------------------#    
    #-------------------- over identification --------------------#
    #-------------------------------------------------------------# 
        
    if len(iv_col) <= len(fake_x):
        warnings.warn("There is no over identification, number of iv <= number of endogenous vars")
        sargan_stat = 0
        sargan_stat_p_val = 0
        b_stat = 0
        b_stat_p_val = 0
    else:
        resid = result.demeaned_df['resid'].values
        df_overid = len(all_exo_x)-len(consist_col) # number of overidentification constraints

        s_n_1 = np.dot(np.dot(resid,pz_full),resid.T)
        s_n_2 = np.dot(resid,resid.T)/nobs
        sargan_stat = round(s_n_1/s_n_2,6)
        sargan_stat_p_val =round(1 - chi2.cdf(sargan_stat, df_overid),6) 

        b_1 = s_n_1/df_overid
        b_2 = (np.dot(np.dot(resid,m_z_full),resid.T))/(nobs - len(all_exo_x))
        b_stat = round(b_1/b_2,6)
        b_stat_p_val = round(1 - chi2.cdf(b_stat, df_overid),6)
        #b_stat_p_val = f.sf(b_stat, df_overid, nobs - len(all_exo_x))

    #-------------------------------------------------------------#    
    #-------------------- endogeneity test    --------------------#
    #-------------------------------------------------------------# 
    uc = demeaned_df['resid'].values


    model_test = sm.OLS(demeaned_df[out_col], demeaned_df[old_x + fake_x])
    result_test = model_test.fit()

    #x_exog2 = demeaned_df[old_x+fake_x].values
    u_e = result_test.resid
    z_test = demeaned_df[old_x + iv_col + fake_x ].values
    zpz_test = np.dot(z_test.T,z_test)
    zpz_test_inv = np.linalg.inv(zpz_test)
    pz_test = np.dot(np.dot(z_test,zpz_test_inv),z_test.T)

    u_c = demeaned_df.resid.values
    #d1 = np.dot(np.dot(u_e.T,pz_test),u_e)

    d1 = np.dot(np.dot(u_e.T,pz_test),u_e)

    d2 = np.dot(np.dot(u_c.T,pz_full),u_c)
    d3 = np.dot(u_e.T,u_e)/nobs 
    durbin_stat = (d1-d2)/d3
    durbin_stat_p_val = round(1 - chi2.cdf(durbin_stat, len(fake_x)),6)
    
        

    #-------------------------------------------------------#    
    #-------------------- format output --------------------#
    #-------------------------------------------------------# 
    gen_title = 'Weak IV test with critical values based on 2SLS size'
    stat_header = None
    gen_stubs = ['Cragg-Donald Statistics:','number of instrumental variables:', 'number of endogenous variables:']
    cd_test_stat = [(cd_stat,),(k2,),(N,)]
    
    cd_tab = SimpleTable(cd_test_stat,
                         stat_header,
                         gen_stubs,
                         title = gen_title,
                         txt_fmt = gen_fmt)        
    
    
    wald_header = ['5%', '10%', '20%', '30%'] 
    wald_test_stat = critical_val

    
    tab_row_name = ['2SLS Size of nominal 5% Wald test']
    critical_val_tab = SimpleTable(wald_test_stat,
                                   wald_header,
                                   tab_row_name,
                                   title = None)  
    
    print(cd_tab)
    print(critical_val_tab)        
    print('H0: Instruments are weak')
    
    #---------------------------------------------------------#
    print()
    sargan = (forg(sargan_stat,4),forg(sargan_stat_p_val,4))
    Basmann = (forg(b_stat,4),forg(b_stat_p_val,4))

    gen_title2 = 'Over identification test - nonrobust'
    stat_header2 = ['test statistics', 'p values']
    gen_stubs2 = ['Sargan Statistics:','Basmann Statistics:']
    overid_stat = [sargan,Basmann]

    critical_val_tab = SimpleTable(overid_stat,
                                   stat_header2,
                                   gen_stubs2,
                                   title = gen_title2)  

    print(critical_val_tab)
    
    #---------------------------------------------------------#
    print()
    durbin = (forg(durbin_stat,4),forg(durbin_stat_p_val,4))

    gen_title3 = 'Tests of endogeneity'
    stat_header3 = ['test statistics', 'p values']
    gen_stubs3 = ['Durbin Statistics:']
    endog_stat = [durbin]

    durbin_tab = SimpleTable(endog_stat,
                                   stat_header3,
                                   gen_stubs3,
                                   title = gen_title3)  
    
    print(durbin_tab)
    print('H0: variables are exogenous')
    
    return