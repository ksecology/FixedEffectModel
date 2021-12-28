from scipy.stats import f


def cal_fullmodel(data_df, out_col, consist_col, category_col, rank, RSS, originRSS):
    """

    This function is used to calculate rsquared, rsquared_adj, fvalue, f_pvalue, and DoF of F-test for full model(
    data before demean process)

    """
    k0 = 0
    if ('const' in consist_col):
        k0 = 1
    
    if k0==0 and category_col==[]:
        TSS = sum(((data_df[out_col]) ** 2).values)[0]            
    else:
        TSS = sum(((data_df[out_col] - data_df[out_col].mean()) ** 2).values)[0]    

    RSS = float("{:.6f}".format(RSS))
    TSS = float("{:.6f}".format(TSS))
    originRSS = float("{:.6f}".format(originRSS))
        
    rsquared = 1 - RSS / TSS
    if category_col != []:
        rsquared_adj = 1 - len(data_df) / (len(data_df) - len(consist_col) - rank + k0) * (1 - rsquared)
    else:
        rsquared_adj = 1 - (len(data_df) - k0) / (len(data_df) - len(consist_col)) * (1 - rsquared)
    
    
    df_full_model = rank + len(consist_col) - k0
    
    if df_full_model > 0:
        fvalue = (TSS - originRSS) * (len(data_df) - len(consist_col) - rank) / (RSS * df_full_model)        
    else:
        fvalue = 0
    f_pvalue = f.sf(fvalue, (rank + len(consist_col) - k0), (len(data_df) - len(consist_col) - rank + k0))
    f_df = [(rank + len(consist_col) - k0), (len(data_df) - len(consist_col) - rank + k0)]
        
    return rsquared, rsquared_adj, fvalue, f_pvalue, f_df
