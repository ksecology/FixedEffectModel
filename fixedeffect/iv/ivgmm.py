from ..utils.DemeanDataframe import demean_dataframe
from ..utils.FormTransfer import form_transfer
from ..utils.CalDf import cal_df
from ..utils.CalFullModel import cal_fullmodel
from ..utils.WaldTest import waldtest
from ..utils.OLSFixed import OLSFixed
from ..utils.GenCrossProd import gencrossprod

import statsmodels.api as sm
from scipy.stats import t
from scipy.stats import f
from numpy.linalg import eigvalsh, inv, matrix_rank, pinv
import numpy as np
import pandas as pd

import warnings

class ivgmm:
    def __init__(self,
                 data_df,
                 dependent = None,
                 exog_x = None,
                 endog_x = [],
                 iv = [],
                 category=[],
                 cluster=[],
                 formula = None,
                 robust = False,
                 noint = False,
                 gmm2=False,
                 **kwargs
                 ):
        """
        :param data_df: Dataframe of relevant data
        :param dependent: List of dependent variables(so far, only support one dependent variable)
        :param exog_x: List of exogenous or right-hand-side variables (variable by time by entity).
        :param endog_x: List of endogenous variables
        :param iv: List of instrument variables
        :param category: List of category variables(fixed effects)
        :param cluster: List of cluster variables
        :param formula: a string like 'y~x+x2|id+firm|id',dependent_variable~continuous_variable|fixed_effect|clusters
        :param robust: bool value of whether to get a robust variance
        :param noint: force nointercept option
        :param gmm2: whether generate gmm1 or gmm2
        :return:params,df,bse,tvalues,pvalues,rsquared,rsquared_adj,fvalue,f_pvalue,variance_matrix,fittedvalues,resid,summary
        **kwargs:some hidden option not supposed to be used by user
        """
        if len(cluster)>1:
            raise NameError('IVGMM only support one-way cluster standard error')

        if (robust==True) and (len(cluster)==1):
            warnings.warn('Generate the cluster robust standard error and set robust=False')

        # grammar check
        if (exog_x is None) & (formula is None):
            raise NameError('You have to input list of variables name or formula')
        elif exog_x is None:
            dependent, exog_x, category_input, cluster_input, endog_x, iv = form_transfer(formula)
            print('dependent variable(s):', dependent)
            print('independent(exogenous):', exog_x)
            print('category variables(fixed effects):', category_input)
            print('cluster variables:', cluster_input)
            if endog_x:
                print('endogenous variables:', endog_x)
                print('instruments:', iv)
            else:
                raise NameError('You have to input endogenous variables for iv2sls')

        else:
            dependent, exog_x, category_input, cluster_input, endog_x, iv = dependent, exog_x, category,  \
                                                                              cluster, endog_x, iv

        # df preprocess
        data_df.fillna(0, inplace=True)
        data_df = gencrossprod(data_df, exog_x)
        orignal_exog_x = exog_x

        self.data_df = data_df
        self.dependent = dependent
        self.exog_x = exog_x
        self.endog_x = endog_x
        self.iv = iv
        self.category_input = category_input
        self.cluster_input = cluster_input
        self.formula = formula
        self.robust = robust
        self.noint = noint
        self.orignal_exog_x = orignal_exog_x
        self.gmm2 = gmm2

        self.is_clustered = False
        if len(cluster_input) > 0:
            if (cluster_input[0] != '0'):
                self.is_clustered = True


    def fit(self,
            epsilon = 1e-8,
            max_iter = 1e6):

        data_df = self.data_df
        demeaned_df = self.preprocess_data(epsilon, max_iter)

        iv_full = self.exog_x + self.iv
        x_full = self.exog_x + self.endog_x
        if self.noint is False:
            iv_full = ['const'] + iv_full
            x_full = ['const'] + x_full

        z = np.array(demeaned_df[iv_full].values)
        x = np.array(demeaned_df[x_full].values)
        y = np.array(demeaned_df[self.dependent].values)

        # calculates the first stage F statistic
        f_stat_first_stage, f_stat_first_stage_pval = self.compute_first_stage_f_stat(demeaned_df, data_df, z, iv_full)

        # estimate parameters for one-step GMM
        beta, v_beta, w2 = self.gmm1_estimation(z, x, y, iv_full, x_full, demeaned_df)

        if self.gmm2 == True:
            beta, v_beta = self.gmm2_estimation(z, x, y, iv_full, demeaned_df, w2)


        result, f_result = self.save_results(x_full, beta, v_beta, demeaned_df, y, x,
                                             f_stat_first_stage, f_stat_first_stage_pval)

        self.compute_summary_statistics(result, f_result, self.rank)

        return f_result

    def save_results(self, x_full, beta, v_beta, demeaned_df, y, x,
                     f_stat_first_stage, f_stat_first_stage_pval):

        se = np.sqrt(np.reshape(np.diag(v_beta), (len(v_beta), 1)))
        df = demeaned_df.shape[0] - len(x_full) - self.rank + self.k0

        result = pd.DataFrame()
        result['index'] = x_full
        result = result.set_index('index')
        result['params'] = beta
        result['bse'] = se
        result['tvalues'] = result.params / result.bse
        result['pvalues'] = pd.Series(2 * t.sf(np.abs(result.tvalues), df), index=list(result.params.index))

        # ------ initiate result object ------#
        demeaned_df['resid'] = y - x @ beta

        f_result = OLSFixed()
        f_result.model = 'ivgmm'
        f_result.dependent = self.dependent
        f_result.exog_x = self.exog_x
        f_result.endog_x = self.endog_x
        f_result.iv = self.iv
        f_result.category_input = self.category_input
        f_result.data_df = self.data_df
        f_result.demeaned_df = demeaned_df
        f_result.params = result['params']
        f_result.bse = result['bse']
        f_result.df = df
        f_result.variance_matrix = v_beta
        f_result.tvalues = result['tvalues']
        f_result.pvalues = result['pvalues']
        #f_result.fittedvalues = x @ beta
        f_result.x_second_stage = x_full

        f_result.f_stat_first_stage = f_stat_first_stage
        f_result.f_stat_first_stage_pval = f_stat_first_stage_pval
        f_result.orignal_exog_x = self.orignal_exog_x
        f_result.cluster = self.cluster_input

        if self.cov_type == 'Heteroskedastic-Robust Covariance':
            f_result.Covariance_Type = 'Heteroskedastic-Robust'
            f_result.cluster_method = 'no_cluster'
        elif self.cov_type == 'One-way Clustered Covariance':
            f_result.Covariance_Type = 'clustered'
            f_result.cluster_method = 'one-way'
        else:
            f_result.Covariance_Type = 'nonrobust'
            f_result.cluster_method = 'no_cluster'

        return result, f_result

    def preprocess_data(self, epsilon, max_iter):
        data_df   = self.data_df
        dependent = self.dependent
        exog_x    = self.exog_x
        endog_x   = self.endog_x
        iv        = self.iv
        noint     = self.noint
        category_input = self.category_input

        if noint is True:
            self.k0 = 0
        else:
            self.k0 = 1

        if (category_input == []):
            demeaned_df = data_df.copy()
            if noint is False:
                demeaned_df['const'] = 1
            self.rank = 0
        else:
            all_cols = []
            for i in exog_x:
                all_cols.append(i)
            for i in endog_x:
                all_cols.append(i)
            for i in iv:
                all_cols.append(i)
            all_cols.append(dependent[0])
            demeaned_df = demean_dataframe(data_df, all_cols, category_input, epsilon = epsilon, max_iter = max_iter)
            if noint is False:
                for i in all_cols:
                    demeaned_df[i] = demeaned_df[i].add(data_df[i].mean())
                demeaned_df['const'] = 1
            self.rank = cal_df(data_df, category_input)

        return demeaned_df

    def cov_cluster(self, demeaned_df, iv_full, z):
        id_list = np.unique(demeaned_df[self.cluster_input[0]].values)
        w = np.zeros([z.shape[1], z.shape[1]])
        for i in id_list:
            demeaned_df_sub = demeaned_df[demeaned_df[self.cluster_input[0]] == i]
            z_sub = demeaned_df_sub[iv_full].values
            res_sub = demeaned_df_sub['resid'].values.reshape(-1, 1)
            zres_sub_ = np.sum(z_sub * res_sub, axis=0).reshape(-1, 1)
            w = w + zres_sub_ @ zres_sub_.T

        return w

    def gmm1_estimation(self, z, x, y, iv_full, x_full, demeaned_df):
        qxz = x.T @ z
        qzz = z.T @ z
        qzx = z.T @ x
        pz  = z@(pinv(z.T @ z))

        beta = pinv(qxz@pinv(qzz)@qzx)@x.T @ pz@z.T @y
        resid = y - x@beta
        demeaned_df['resid'] = resid.flatten()
        zres = z * resid * resid
        w2 = zres.T @ z
        self.cov_type = 'Non-Robust Covariance'
        df = demeaned_df.shape[0] - len(x_full) - self.rank + self.k0

        if (self.robust == True) or (self.is_clustered == True):
            bread = pinv(qxz@pinv(qzz)@qzx)
            if self.robust==True:
                self.cov_type = 'Heteroskedastic-Robust Covariance'
            else:
                self.cov_type = 'One-way Clustered Covariance'
                w2 = self.cov_cluster(demeaned_df, iv_full, z)

            mid1 = qxz @ pinv(qzz)
            meat = mid1 @ w2 @ mid1.T
            v_beta = bread @ meat @ bread
        else:
            sigma2_hat = (resid.T @ resid) / df
            var_1 = pinv(x.T @ pz @ z.T @ x)
            v_beta = sigma2_hat * var_1

        return beta, v_beta, w2


    def gmm2_estimation(self, z, x, y, iv_full, demeaned_df, w2):
        qxz = x.T @ z
        qzx = z.T @ x
        qzy = z.T @ y

        xpx = qxz @ pinv(w2) @ qzx
        xpy = qxz @ pinv(w2) @ qzy
        beta = pinv(xpx) @ xpy

        if (self.robust == True) or (self.is_clustered == True):
            res2 = y - x @ beta
            bread = pinv(xpx)
            if self.robust == True:
                zres2 = z * res2 * res2
                w3 = zres2.T @ z
            else:
                demeaned_df['resid'] = res2
                w3 = self.cov_cluster(demeaned_df, iv_full, z)
            mid1 = qxz @ pinv(w2)
            meat = mid1 @ w3 @ mid1.T
            v_beta = bread @ meat @ bread
        else:
            mid1 = qxz @ pinv(w2)
            v_beta = pinv(mid1 @ qzx)
        return beta, v_beta


    def compute_first_stage_f_stat(self, demeaned_df, data_df, z, iv_full):
        iv_model = []
        iv_result = []
        f_stat_first_stage = []
        f_stat_first_stage_pval = []

        for i in self.endog_x:
            model_iv = sm.OLS(demeaned_df[i], z)
            result_iv = model_iv.fit()
            iv_model.append(model_iv)
            iv_result.append(result_iv)
            iv_coeff = result_iv.params.values
            demeaned_df["hat_" + i] = np.dot(iv_coeff, demeaned_df[iv_full].values.T)
            data_df["hat_" + i] = demeaned_df["hat_" + i]

            f_stat_first_stage.append(result_iv.fvalue)
            f_stat_first_stage_pval.append(result_iv.f_pvalue)

        return f_stat_first_stage, f_stat_first_stage_pval

    def compute_summary_statistics(self,
                                   result,
                                   f_result,
                                   rank):

        dependent = self.dependent
        category_input = self.category_input
        data_df = self.data_df
        exog_x = self.exog_x
        endog_x = self.endog_x
        iv= self.iv
        noint = self.noint

        iv_full = exog_x + iv
        x_full = exog_x + endog_x

        if noint is False:
            iv_full = ['const'] + iv_full
            x_full = ['const'] + x_full

        if self.noint is True:
            k0 = 0
        else:
            k0 = 1

        demeaned_df = f_result.demeaned_df
        n = demeaned_df.shape[0]
        k = len(x_full)
        f_result.resid = demeaned_df['resid']

        proj_rss = sum(f_result.resid ** 2)
        proj_rss = float("{:.8f}".format(proj_rss))  # round up

        # calculate total sum squared of error
        if k0 == 0:
            proj_tss = sum(((demeaned_df[dependent]) ** 2).values)[0]
        else:
            proj_tss = sum(((demeaned_df[dependent] - demeaned_df[dependent].mean()) ** 2).values)[0]

        proj_tss = float("{:.8f}".format(proj_tss))  # round up
        if proj_tss > 0:
            f_result.rsquared = 1 - proj_rss / proj_tss
        else:
            raise NameError('Total sum of square equal 0, program quit.')

        f_result.rsquared_adj = 1 - (len(data_df) - k0) / n * (1 - f_result.rsquared)

        if k0 == 0:
            w = waldtest(f_result.params, f_result.variance_matrix)
        else:
            # get rid of constant in the vc matrix
            f_var_mat_noint = f_result.variance_matrix.copy()
            if type(f_var_mat_noint) == np.ndarray:
                f_var_mat_noint = np.delete(f_var_mat_noint, 0, 0)
                f_var_mat_noint = np.delete(f_var_mat_noint, 0, 1)
            else:
                f_var_mat_noint = f_var_mat_noint.drop('const', axis=1)
                f_var_mat_noint = f_var_mat_noint.drop('const', axis=0)

            # get rid of constant in the param column
            params_noint = f_result.params.drop('const', axis=0)
            if category_input == []:
                w = waldtest(params_noint, (n - k) / (n - k + k0) * f_var_mat_noint)
            else:
                w = waldtest(params_noint, f_var_mat_noint)

        # calculate f-statistics
        df_model = k - k0
        if df_model > 0:
            # if do pooled regression
            if category_input == []:
                scale_const = (n - k) / (n - k + k0)
                f_result.fvalue = scale_const * w/df_model
            # if do fixed effect, just ignore
            else:
                f_result.fvalue = w / df_model
        else:
            f_result.fvalue = 0

        f_result.f_pvalue = f.sf(f_result.fvalue, df_model, f_result.df)
        f_result.f_df_proj = [df_model, f_result.df]

        # get full-model related statistics
        f_result.full_rsquared, f_result.full_rsquared_adj, f_result.full_fvalue, f_result.full_f_pvalue, f_result.f_df_full \
            = cal_fullmodel(data_df,
                            dependent,
                            x_full,
                            category_input,
                            rank,
                            RSS=sum(f_result.resid ** 2),
                            originRSS=sum(demeaned_df['resid'] ** 2))


        f_result.nobs = n
        f_result.yname = dependent
        f_result.xname = x_full
        f_result.resid_std_err = np.sqrt(sum(f_result.resid ** 2) / (f_result.df - rank))

        f_result.treatment_input = None

        return






