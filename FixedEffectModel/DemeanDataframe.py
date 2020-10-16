import numpy as np


def demean_dataframe(df, consist_var, category_col, epsilon=1e-8, max_iter=100):
    n = df.shape[0]
    df_copy = df.copy()
    for consist in consist_var:
        mse = 10
        iter_count = 0
        demeaned_df = df.copy()
        demeans_cache = np.zeros(n, np.float64)
        while mse > epsilon:
            for cat in category_col:
                if iter_count == 0:
                    demeaned_df[consist] = df[consist] - df.groupby(cat)[consist].transform('mean')
                else:
                    demeaned_df[consist] = demeaned_df[consist] - demeaned_df.groupby(cat)[consist].transform('mean')
            iter_count += 1
            mse = np.linalg.norm(demeaned_df[consist].values - demeans_cache)
            demeans_cache = demeaned_df[consist].copy().values
            if iter_count > max_iter:
                raise RuntimeWarning('Exceeds the maximum iteration counts, please recheck dataset')
                break
        df_copy[[consist]] = demeaned_df[[consist]]
        # print(iter_count)
    return df_copy
