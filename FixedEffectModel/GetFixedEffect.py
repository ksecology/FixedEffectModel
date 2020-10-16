import numpy as np
import networkx as nx
import pandas as pd
from FixedEffectModel.Projection import projection


def getfe(result, epsilon=1e-8):
    """

    This function is used to get fixed effect.
    :param result: result of model after demean
    :param epsilon: tolerance for projection
    :return: dataframe of fixed effect for each category variable
    """
    data_df = result.data_df.copy()
    demean = result.demeaned_df.copy()
    coeff = result.params.values
    consist_col = result.consist_col
    category_col = result.category_col
    out_col = result.out_col

    y = data_df[out_col[0]].values
    b_x = np.dot(coeff, data_df[consist_col].values.T)
    # print('b_x shape:', b_x.shape)
    ori_resid = y - b_x
    true_resid = ori_resid - demean['resid'].values
    alpha_array = projection(data_df, true_resid, category_col, epsilon)

    index_name = []
    index_num = []
    e = len(category_col)
    for i in range(e):
        m = np.unique(data_df[category_col[i]].values)
        index_num.append(m.shape[0])
        for l in m:
            name = category_col[i] + str(l)
            index_name.append(name)

    dummy_shape = len(index_name)
    df_copy = data_df.copy()
    g = nx.Graph()
    e = len(category_col)
    edge_list = df_copy[category_col].values

    n = edge_list.shape[0]
    if e > 1:
        for i in range(n):
            g.add_edge(category_col[0] + str(edge_list[i][0]), category_col[1] + str(edge_list[i][1]))
    connected_component = [list(cc) for cc in list(nx.connected_components(g))]

    col_name = ['dummy_name', 'count']
    for i in range(e):
        count_df = data_df[category_col[i]].value_counts().sort_index()
        count_df = count_df.reset_index()
        count_df.columns = col_name
        count_df['dummy_name'] = category_col[i] + '_' + count_df['dummy_name'].apply(str)
        count_df['cat'] = category_col[i]
        if i == 0:
            dummy_df = count_df.copy()
        else:
            dummy_df = pd.concat([dummy_df, count_df])

    if e == 1:
        cc_list = np.ones(dummy_shape)
    else:
        cc_list = np.ones(dummy_shape, dtype=np.int)

        cc_num = len(connected_component)
        first_2_cat = category_col[0:2]
        first_2_cat_shape = 0
        for cat in first_2_cat:
            first_2_cat_shape += np.unique(data_df[cat].values).shape[0]
        for i in range(first_2_cat_shape):
            if cc_num == 1:
                cc_list[i] = 1
            else:
                for j in range(cc_num):
                    if dummy_df['dummy_name'].values[i] in connected_component[j]:
                        cc_list[i] = (j + 1)
        target = index_num[0] + index_num[1]
        for i in range(2, e):
            cc_list[target:target + index_num[i]] = np.ones(index_num[i], dtype=np.int) * (cc_num + i - 1)
            target += index_num[i]
    # print(cc_list)
    dummy_df['cc'] = cc_list
    group_num = np.unique(cc_list).shape[0]

    normalized_alpha = np.zeros_like(alpha_array, dtype=np.float64)
    for i in range(1, group_num + 1):
        group_alpha = alpha_array[cc_list == i]
        group_count = dummy_df['count'].values[cc_list == i]
        max_index = np.argmax(group_count)

        group_name = dummy_df['cat'].values[cc_list == i]
        max_name = group_name[max_index]
        # refence here
        ref_value = group_alpha[max_index]
        normalized_alpha[cc_list == i] = np.where(group_name == max_name, group_alpha - ref_value,
                                                  group_alpha + ref_value)
    dummy_df['effect'] = normalized_alpha
    return dummy_df
