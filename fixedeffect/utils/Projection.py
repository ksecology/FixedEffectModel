import numpy as np


def projection(data_df, b_x, category_col, epsilon=1e-5):
    """

    :param data_df: Dataframe with relevant data
    :param b_x:
    :param category_col: List of category variables
    :param epsilon: tolerance of projection
    :return:
    """
    category_unique_val = []
    length_list = []
    e = len(category_col)
    x_dim = 0

    for i in range(e):
        m = np.unique(data_df[category_col[i]].values)
        category_unique_val.append(m)
        length_list.append(m.shape[0])
        x_dim += m.shape[0]
    threshold = epsilon * x_dim

    n = data_df.shape[0]
    max_iter = 2 * np.log(1e-5) / np.log(1 - 1 / (10 * min([x_dim, n])))
    if (max_iter // n) == 0:
        max_iter = 2 * np.log(1e-5) / np.log(1 - 1 / (1 * max([x_dim, n])))
    # print('max_iter:', max_iter)
    fe = data_df[category_col].values
    d_i_index = np.zeros(e, dtype=np.int)
    x = np.zeros(x_dim, dtype=np.float64)
    x_cache = np.zeros(x_dim, dtype=np.float64)

    loop_index = np.zeros((n, e), dtype=np.int)
    update = 10
    iter_loops = int((max_iter // n) + 1)
    #print('max_whole_iter:', iter_loops)
    for loop in range(iter_loops):
        if loop == 0:
            for i in range(n):
                d_ij = 0
                for j in range(e):
                    index = np.where(category_unique_val[j] == fe[i][j])
                    d_i_index[j] = d_ij + index[0][0]
                    d_ij += length_list[j]
                loop_index[i] = d_i_index
                update_value = (np.sum(x[d_i_index]) - b_x[i]) / e
                x[d_i_index] -= update_value
            x_cache = x.copy()
        else:
            prob_array = np.random.choice(n, n)
            for i in range(n):
                d_i_index = loop_index[prob_array[i]]
                update_value = (np.sum(x[d_i_index]) - b_x[prob_array[i]]) / e
                x[d_i_index] -= update_value
            update = np.linalg.norm(x - x_cache)
            #print(update)
            if update < threshold:
                # print('not exceed max iteration')
                break
            x_cache = x.copy()
    return x
