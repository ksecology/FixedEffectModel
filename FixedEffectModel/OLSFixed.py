from FixedEffectModel.Forg import forg
import time
import pandas as pd
from statsmodels.iolib.tableformatting import (gen_fmt, fmt_2)
from statsmodels.iolib.table import SimpleTable
from itertools import zip_longest
from statsmodels.compat.python import lrange, lmap, lzip
from scipy.stats import t


class OLSFixed(object):
    def __init(self):
        self.params = None
        self.df = None
        self.bse = None
        self.tvalues = None
        self.pvalues = None
        self.summary = None
        self.covar_matrix = None
        self.fittedvalues = None
        self.resid = None
        self.rsquared = None
        self.rsquared_adj = None
        self.full_rsquared = None
        self.full_rsquared_adj = None
        self.fvalue = None
        self.f_pvalue = None
        self.full_fvalue = None
        self.full_f_pvalue = None
        self.variance_matrix = None
        self.fittedvalues = None
        self.resid = None
        self.nobs = None
        self.yname = None
        self.xname = None
        self.resid_std_err = None
        self.Covariance_Type = None
        self.cluster_method = None
        self.demeaned_df = None
        self.data_df = None
        self.general_table = None
        self.std_err_name = None
        self.consist_col = None
        self.old_x = None
        self.category_col = None
        self.out_col = None
        self.f_df_full = None
        self.f_df_proj = None

    def conf_int(self, conf=0.05):
        tmpdf = pd.DataFrame(columns=[0, 1], index=list(self.params.index))
        tmpdf[0] = self.params - t.ppf(1 - conf / 2, self.df) * self.bse
        tmpdf[1] = self.params + t.ppf(1 - conf / 2, self.df) * self.bse
        return tmpdf

    def summary(self, yname=None, xname=None, title=0, alpha=.05,
                returns='text', model_info=None):
        # General part of the summary table
        if title == 0:
            title = 'High Dimensional Fixed Effect Regression Results'

        if type(xname) == str: xname = [xname]
        if type(yname) == str: yname = [yname]
        if xname is not None and len(xname) != len(self.xname):
            # GH 2298
            raise ValueError('User supplied xnames must have the same number of '
                             'entries as the number of model parameters '
                             '({0})'.format(len(self.xname)))

        if yname is not None and len(yname) != len(self.yname):
            raise ValueError('User supplied ynames must have the same number of '
                             'entries as the number of model dependent variables '
                             '({0})'.format(len(self.yname)))
        if xname is None:
            xname = self.xname
        if yname is None:
            yname = self.yname

        time_now = time.localtime()
        time_of_day = [time.strftime("%H:%M:%S", time_now)]
        date = time.strftime("%a, %d %b %Y", time_now)
        nobs = int(self.nobs)
        df_model = self.df
        resid_std_err = forg(self.resid_std_err, 4)
        Covariance_Type = self.Covariance_Type
        cluster_method = self.cluster_method
        gen_left = [('Dep. Variable:', yname),
                    ('No. Observations:', [nobs]),  # TODO: What happens with multiple names?
                    ('DoF of residual:', [df_model]),
                    ('Residual std err:', [resid_std_err]),
                    ('Covariance Type:', [Covariance_Type]),
                    ('Cluster Method:', [cluster_method])
                    ]
        r_squared = forg(self.rsquared, 4)
        rsquared_adj = forg(self.rsquared_adj, 4)
        full_rsquared = forg(self.full_rsquared, 4)
        full_rsquared_adj = forg(self.full_rsquared_adj, 4)
        fvalue = forg(self.fvalue, 4)
        f_pvalue = forg(self.f_pvalue, 4)
        full_fvalue = forg(self.full_fvalue, 4)
        full_f_pvalue = forg(self.full_f_pvalue, 4)
        gen_right = [('R-squared(proj model):', [r_squared]),
                     ('Adj. R-squared(proj model):', [rsquared_adj]),
                     ('R-squared(full model):', [full_rsquared]),
                     ('Adj. R-squared(full model):', [full_rsquared_adj]),
                     ('F-statistic(proj model):', [fvalue]),
                     ('Prob (F-statistic (proj model)):', [f_pvalue]),
                     ('DoF of F-test (proj model):', [self.f_df_proj]),
                     ('F-statistic(full model):', [full_fvalue]),
                     ('Prob (F-statistic (full model)):', [full_f_pvalue]),
                     ('DoF of F-test (full model):', [self.f_df_full])
                     ]
        # pad both tables to equal number of rows
        if len(gen_right) < len(gen_left):
            # fill up with blank lines to same length
            gen_right += [(' ', ' ')] * (len(gen_left) - len(gen_right))
        elif len(gen_right) > len(gen_left):
            # fill up with blank lines to same length, just to keep it symmetric
            gen_left += [(' ', ' ')] * (len(gen_right) - len(gen_left))

        gen_stubs_left, gen_data_left = zip_longest(*gen_left)
        gen_title = title
        gen_header = None
        gen_table_left = SimpleTable(gen_data_left,
                                     gen_header,
                                     gen_stubs_left,
                                     title=gen_title,
                                     txt_fmt=gen_fmt
                                     )
        gen_stubs_right, gen_data_right = zip_longest(*gen_right)
        gen_table_right = SimpleTable(gen_data_right,
                                      gen_header,
                                      gen_stubs_right,
                                      title=gen_title,
                                      txt_fmt=gen_fmt
                                      )
        gen_table_left.extend_right(gen_table_right)
        self.general_table = gen_table_left

        # Parameters part of the summary table
        s_alp = alpha / 2
        c_alp = 1 - alpha / 2
        if Covariance_Type == 'nonrobust':
            self.std_err_name = 'nonrobust std err'
        elif Covariance_Type == 'robust':
            self.std_err_name = 'robust std err'
        elif Covariance_Type == 'clustered':
            self.std_err_name = 'cluster std err'
        else:
            self.std_err_name = 'std err'
        param_header = ['coef', self.std_err_name, 't', 'P>|t|', '[' + str(s_alp),
                        str(c_alp) + ']']  # alp + ' Conf. Interval'
        params_stubs = xname
        params = self.params.copy()
        conf_int = self.conf_int(alpha)
        std_err = self.bse.copy()
        exog_len = lrange(len(xname))
        tstat = self.tvalues.copy()
        prob_stat = self.pvalues.copy()
        for i in range(len(self.params)):
            params[i] = forg(self.params[i], 5)
            std_err[i] = forg(self.bse[i], 5)
            tstat[i] = forg(self.tvalues[i], 4)
            prob_stat[i] = forg(self.pvalues[i], 4)

        # Simpletable should be able to handle the formating
        params_data = lzip(["%#6.5f" % (params[i]) for i in exog_len],
                           ["%#6.5f" % (std_err[i]) for i in exog_len],
                           ["%#6.4f" % (tstat[i]) for i in exog_len],
                           ["%#6.4f" % (prob_stat[i]) for i in exog_len],
                           ["%#6.4f" % conf_int[0][i] for i in exog_len],
                           ["%#6.4f" % conf_int[1][i] for i in exog_len])
        self.parameter_table = SimpleTable(params_data,
                                           param_header,
                                           params_stubs,
                                           title=None,
                                           txt_fmt=fmt_2
                                           )
        print(self.general_table)
        print(self.parameter_table)
        return

    def to_excel(self, file=None):
        df_tmp = pd.DataFrame(columns=['coef', self.std_err_name, 't', 'p', 'conf_int_lower', 'conf_int_upper'],
                              index=self.xname)
        df_tmp.coef = self.params
        df_tmp[self.std_err_name] = self.bse
        df_tmp.t = self.tvalues
        df_tmp.p = self.pvalues
        df_tmp.conf_int_lower = self.conf_int()[0]
        df_tmp.conf_int_upper = self.conf_int()[1]
        df_tmp2 = pd.DataFrame(
            columns=['dep_variable', 'no_obs', 'df_model', 'resid_std_err', 'Covariance_Type', 'cluster_method',
                     'proj_Rsquared', 'proj_Rsquared_adj', 'full_Rsquared', 'full_Rsquared_adj',
                     'proj_fvalue', 'proj_f_pvalue', 'full_fvalue', 'full_f_pvalue'])
        df_tmp2.dep_variable = self.yname  # y不止一个怎么办
        df_tmp2.no_obs = self.nobs
        df_tmp2.df_model = self.df
        df_tmp2.resid_std_err = self.resid_std_err
        df_tmp2.Covariance_Type = self.Covariance_Type
        df_tmp2.cluster_method = self.cluster_method
        df_tmp2.proj_Rsquared = self.rsquared
        df_tmp2.proj_Rsquared_adj = self.rsquared_adj
        df_tmp2.full_Rsquared = self.full_rsquared
        df_tmp2.full_Rsquared_adj = self.full_rsquared_adj
        df_tmp2.proj_fvalue = self.fvalue
        df_tmp2.proj_f_pvalue = self.f_pvalue
        df_tmp2.full_fvalue = self.full_fvalue
        df_tmp2.full_f_pvalue = self.full_f_pvalue
        if file is None:
            file = 'output.xls'
        writer = pd.ExcelWriter(file)
        df_tmp.to_excel(writer, encoding='utf-8', sheet_name='params')
        df_tmp2.to_excel(writer, encoding='utf-8', sheet_name='general', index=False)
        writer.save()
