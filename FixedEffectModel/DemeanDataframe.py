import numpy as np
import statsmodels.api as sm

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
    
    #2020/12/23: 只做一次demean的时候，为了节约时间和内存，不比较mse和epsilon
    if len(category_col) == 1:
        cat = category_col[0]
        for consist in consist_var:            
            df_copy[consist] = df[consist] - df.groupby(cat)[consist].transform('mean')                                            
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
