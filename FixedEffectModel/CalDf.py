import networkx as nx
import numpy as np


def cal_df(data_df, category_col):
    """
    This function returns the degree of freedom of category variables(DoF). When there are category variables( fixed
    effects), part of degree of freedom of the model will lose during the demean process. When there is only one
    fixed effect, the loss of DoF is the level of this category. When there are more than one fixed effects,
    we need to calculate their connected components. Even if there are more than two fixed effects, only the first
    two will be used.

    :param data_df: Data with relevant variables
    :param category_col: List of category variables(fixed effect)
    :return: the degree of freedom of category variables
    """
    e = 0
    for i in category_col:
        e += np.unique(data_df[i].values).shape[0]
    if len(category_col) >= 2:
        g = nx.Graph()
        edge_list = data_df[category_col].values.tolist()
        for ll in edge_list:
            g.add_edge('fix1_' + str(ll[0]), 'fix2_' + str(ll[1]))
        df = e - (len(list(nx.connected_components(g))) + len(category_col) - 2)
    else:
        df = e
    return df
