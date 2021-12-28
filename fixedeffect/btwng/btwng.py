# 2021/07/12 test
from fixedeffect.FormTransfer import form_transfer
from fixedeffect.GenCrossProd import gencrossprod
import statsmodels.api as sm

def btwng(data_df,
          consist_input = None,
          out_input = None,
          category_input = [],
          fake_x_input = [],
          iv_col_input = [],
          formula = None,
          robust = False,
          noint = False):

    cluster_input = []

    if noint is True:
        k0 = 0
    else:
        k0 = 1

    if (consist_input is None) & (formula is None):
        raise NameError('You have to input list of variables name or formula')
    elif consist_input is None:
        out_col, consist_col, category_col, cluster_col, fake_x, iv_col = form_transfer(formula)
        print('dependent variable(s):', out_col)
        print('independent(exogenous) variables:', consist_col)
        print('category variables:', category_col)
        if fake_x:
            print('endogenous variables:', fake_x)
            print('instruments:', iv_col)
    else:
        out_col, consist_col, category_col, cluster_col, fake_x, iv_col = out_input, consist_input, category_input, \
                                                                          cluster_input, fake_x_input, iv_col_input

    data_df = gencrossprod(data_df, consist_col)

    if (category_col == []):
        raise NameError('cannot perform between group estimation')
    else:
        mean_vars = out_col + consist_col + fake_x + iv_col
        df_mean = data_df.groupby(category_col[0])[mean_vars].transform('mean')

        if noint is False:
            df_mean['const'] = 1
            consist_col = consist_col + ['const']

        model_btwng = sm.OLS(df_mean[out_col], df_mean[consist_col])
        result_btwng = model_btwng.fit()

    return result_btwng


