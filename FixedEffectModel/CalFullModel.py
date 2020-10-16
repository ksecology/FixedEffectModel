from scipy.stats import f


def cal_fullmodel(data_df, out_col, consist_col, rank, RSS):
    """

    This function is used to calculate rsquared, rsquared_adj, fvalue, f_pvalue, and DoF of F-test for full model(
    data before demean process)

    """
    TSS = sum(((data_df[out_col] - data_df[out_col].mean()) ** 2).values)[0]
    rsquared = 1 - RSS / TSS
    rsquared_adj = 1 - (len(data_df) - 1) / (len(data_df) - len(consist_col) - rank) * (1 - rsquared)
    fvalue = (TSS - RSS) * (len(data_df) - len(consist_col) - rank) / (RSS * (rank + len(consist_col) - 1))
    f_pvalue = f.sf(fvalue, (rank + len(consist_col) - 1), (len(data_df) - len(consist_col) - rank))
    f_df = [(rank + len(consist_col) - 1), (len(data_df) - len(consist_col) - rank)]
    return rsquared, rsquared_adj, fvalue, f_pvalue, f_df
