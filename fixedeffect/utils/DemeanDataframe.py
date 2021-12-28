import numpy as np
import pandas as pd
from pandas import (
    get_dummies,
)
from numpy.linalg import lstsq
import warnings

# before version 0.0.3, still use epsilon when demean
def demean_dataframe(df, consist_var, category_col, epsilon=1e-8, max_iter=1e6):
    """
    :param df: Dataframe
    :param consist_var: List of columns need centering on fixed effects
    :param category_col: List of fixed effects
    :param epsilon: Tolerance
    :param max_iter: Maximum iterations
    :return: Demeaned dataframe
    """
    n = df.shape[0]
    df_copy = df.copy()
    is_unbalance = False

    # if there's only one category variable, doesn't matter if balance or not.
    ## is_unbalance option is only used when there're two category variables
    if len(category_col)>1:
        n_cat = 1
        for cat in category_col:
            n_cat = n_cat * df[cat].nunique()
        if n_cat > df.shape[0]:
            warnings.warn('panel is unbalanced')
            is_unbalance = True
    
    #2020/12/23 when demean only once, no need to converge
    if len(category_col) == 1:
        cat = category_col[0]
        for consist in consist_var:            
            df_copy[consist] = df[consist] - df.groupby(cat)[consist].transform('mean')
    elif  len(category_col) == 2:
        df_copy = demean_dataframe_two_cat(df_copy, consist_var, category_col, is_unbalance)
    else:
        for consist in consist_var:
            mse = 10
            iter_count = 0
            demeans_cache = np.zeros(n, np.float64)
            while mse > epsilon:
                for cat in category_col:
                    if iter_count == 0:
                        df_copy[consist] = df[consist] - df.groupby(cat)[consist].transform('mean')
                    else:
                        df_copy[consist] = df_copy[consist] - df_copy.groupby(cat)[consist].transform('mean')
                iter_count += 1
                mse = np.linalg.norm(df_copy[consist].values - demeans_cache)
                demeans_cache = df_copy[consist].copy().values
                if iter_count > max_iter:
                    raise RuntimeWarning('Exceeds the maximum iteration counts, please recheck dataset')
                    break
    return df_copy

## 2021/12/16: to avoid convergence issue when panel is too unbalanced
# demean when category len equals 2
def demean_dataframe_two_cat(df_copy, consist_var, category_col, is_unbalance):
    """
    reference: Baltagi http://library.wbi.ac.id/repository/27.pdf page 176, equation (9.30)
    :param df_copy: Dataframe
    :param consist_var: List of columns need centering on fixed effects
    :param category_col: List of fixed effects
    :return: Demeaned dataframe
    """
    if is_unbalance:
        # first determine which is uid or the category that has the most items
        max_ncat = df_copy[category_col[0]].nunique()
        max_cat  = category_col[0]
        for cat in category_col:
            if df_copy[cat].nunique() >= max_ncat:
                max_ncat = df_copy[cat].nunique()
                max_cat = cat

        min_cat = category_col.copy()
        min_cat.remove(max_cat)
        min_cat = min_cat[0]

        df_copy.sort_values(by=[max_cat, min_cat], inplace=True)

        # demean on the first category variable, max_cat
        for consist in consist_var:
            df_copy[consist] = df_copy[consist] - df_copy.groupby(max_cat)[consist].transform('mean')

        dummies = get_dummies(df_copy[min_cat])  # time dummies
        dummies[max_cat] = df_copy[max_cat]
        dummies[min_cat] = df_copy[min_cat]
        dummies[max_cat] = dummies[max_cat].apply(str)
        dummies[min_cat] = dummies[min_cat].apply(str)
        dummies.set_index([max_cat, min_cat], inplace = True)

        group_mu = dummies.groupby(level=max_cat).transform("mean")
        out = dummies - group_mu  # q_delta_1 @ delta_2

        e = df_copy[consist_var].values
        d = out.values
        resid = e - d @ lstsq(d, e, rcond=None)[0]

        df_out = pd.DataFrame(data=resid, columns=consist_var)
        df_out[max_cat] = df_copy[max_cat]
        df_out[min_cat] = df_copy[min_cat]

    else: # balance
        for consist in consist_var:
            for cat in category_col:
                df_copy[consist] = df_copy[consist] - df_copy.groupby(cat)[consist].transform('mean')
        df_out = df_copy

    return df_out


def transform_mean(df_arr, cat_idx, unq_idx, consist_arr):
    """
    return mean of consist_arr for category with cat_idx as index in df.
    """
    out = np.zeros(consist_arr.shape)
    for i in range(len(unq_idx)):
        x = unq_idx[i]
        mask = np.where(df_arr[:, cat_idx] == x)
        out[mask] = consist_arr[mask].mean()
    return out


def center(consist_arr, df, category_col, ):
    """
    return new vec as the result of vec demeaned after all catrgory.
    note as one time demean.
    """
    vec = consist_arr.copy()
    df_arr = np.array(df)
    for cat in category_col:
        unq_idx = np.unique(df[cat])
        cat_idx = list(df.columns).index(cat)
        vec = vec - transform_mean(df_arr, cat_idx, unq_idx, vec)
    return vec


def demeanonex(df, consist, category_col, return_dict, epsilon=1e-8, max_iter=1e6):
    """
    return demeaned finished vector.
    """
    consist_arr = np.array(df[consist])
    vec = center(consist_arr, df, category_col)
    if len(category_col) == 1:
        print('only 1 demean iteration for', consist, 'since there is only one fixed effect')
        return_dict[consist] = vec.reshape(-1).tolist()
    else:
        iter_count = 1
        mse = 1
        while mse > epsilon:
            prev = vec.copy()
            vec = center(vec, df, category_col)
            nm = np.dot(prev.copy(), prev.copy())
            nm2 = np.dot((prev.copy() - vec.copy()), (prev.copy() - vec.copy()))
            ip = np.dot(prev.copy().T, (prev.copy() - vec.copy()))
            if nm2 > (nm * 1e-18):
                t = ip / nm2
                if t < 0.49:
                    print(t)
                    print('sorry,cannot converge')
                    break
                vec = (1 - t) * prev.copy() + t * vec.copy()
            iter_count = iter_count + 1
            mse = np.linalg.norm(prev.copy() - vec.copy())
            if iter_count > max_iter:
                raise RuntimeWarning('Exceeds the maximum iteration counts, please recheck dataset')
                break

        print(iter_count, 'demean iterations for', consist)
        print('mse', mse)
        return_dict[consist] = vec.reshape(-1).tolist()
