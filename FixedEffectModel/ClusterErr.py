from itertools import combinations
import warnings
import numpy as np


def clustered_error(demean, consist_col, cluster_col, n, k, rank, nested=False, c_method='cgm', psdef=True):
    """

    This function is used to calculate clustered variance matrix based on equation (x'px)^-1 * middle *(x'px)^-1.
    The status of nested, choice of c_method and status of psdef may affect calculation of middle through adjustment of
    degrees of freedom or different calculation method. Thus, they could affect clustered variance matrix and the
    regressors standard error.
    In specific, if there is any category variable nested in cluster variables, the scale of DoF  will be
    (n - 1) / (n - k - 1). If using cgm as c_method, the scale of DoF for each cluster(with G clusters) will be
    G / (G - 1). If using cgm2 as c_method, the scale of DoF to all clusters will be G_MIN / (G_MIN - 1), where G_MIN is
    the minimum number of cluster for each kind of cluster. Last but not least, a bool value psdef is default to be True,
    which means when calculating multi-way clusters, negative eigenvalue of the covariance matrix will be set to zero.
    If psdef is set False, clustered variance matrix will be output without check for negative eigen value.

    :param demean: demeaned dataframe with relevant data
    :param consist_col: List of continuous variables
    :param cluster_col: List of clusters
    :param n: data size
    :param k: number of continuous variables
    :param rank: degree of freedom of category variables(fixed effect)
    :param nested: bool value of whether category variables(fixed effect) is nested in clusters
    :param c_method: method for calculating multi-way clustered stand error. Possible choices are:
            - 'cgm'
            - 'cgm2'
    :param psdef: bool value of whether set negative eigenvalue of multi-clustered variance matrix into zero
    :return: clustered variance matrix
    """
    if len(cluster_col) == 1 and c_method == 'cgm2':
        raise NameError('cgm2 must be applied to multi-clusters')
    beta_list = []
    xpx = np.dot(demean[consist_col].values.T, demean[consist_col].values)
    xpx_inv = np.linalg.inv(xpx)
    # rank = cal_df(demean,category_col)
    # print('rank:',rank)
    # print(xpx_inv)
    demeaned_df = demean.copy()
    # n = demeaned_df.shape[0]
    G_array = np.array([])
    # print('n:', n)
    # k = len(category_col)
    # print('k:', k)
    if nested:
        scale_df = (n - 1) / (n - k - 1)
    else:
        scale_df = (n - 1) / (n - k - rank)

    if len(cluster_col) == 1:
        G = np.unique(demeaned_df[cluster_col].values).shape[0]
        print('G:', G)
        middle = middle_term(demeaned_df, consist_col, cluster_col)
        m = np.dot(xpx_inv, middle)
        beta = np.dot(m, xpx_inv)
        scale = scale_df * G / (G - 1)
        beta = scale * beta
        print(beta)
        beta_list.append(beta)
    else:
        if c_method == 'cgm':
            for cluster in cluster_col:
                middle = middle_term(demeaned_df, consist_col, [cluster])
                G = np.unique(demeaned_df[cluster].values).shape[0]
                # print('G:',G)
                # print('middle:',middle)
                m = np.dot(xpx_inv, middle)
                beta = np.dot(m, xpx_inv)
                scale = scale_df * G / (G - 1)
                beta = scale * beta
                beta_list.append(beta)
            for j in range(2, len(cluster_col) + 1):
                for combine_name in list(combinations(cluster_col, j)):
                    name_list = [e for e in combine_name]
                    # print(name_list)
                    # print(j)
                    new_col_name = ''
                    for i in name_list:
                        new_col_name = new_col_name + '_' + i
                    demeaned_df[new_col_name] = demeaned_df[name_list[0]].apply(str)
                    # print('col_name:', new_col_name)
                    for i in range(1, len(name_list)):
                        # print('col_name:', new_col_name)
                        demeaned_df[new_col_name] = demeaned_df[new_col_name] + '_' + demeaned_df[name_list[i]].apply(
                            str)
                    middle = np.power(-1, j - 1) * middle_term(demeaned_df, consist_col, [new_col_name])
                    # print(middle)
                    m = np.dot(xpx_inv, middle)
                    beta = np.dot(m, xpx_inv)
                    G = np.unique(demeaned_df[new_col_name].values).shape[0]
                    scale = scale_df * G / (G - 1)
                    # print('g:', G)
                    beta = scale * beta
                    beta_list.append(beta)

        elif c_method == 'cgm2':
            for cluster in cluster_col:
                middle = middle_term(demeaned_df, consist_col, [cluster])
                G = np.unique(demeaned_df[cluster].values).shape[0]
                G_array = np.append(G_array, G)
                print('G:', G)
                m = np.dot(xpx_inv, middle)
                beta = np.dot(m, xpx_inv)
                beta_list.append(beta)
            for j in range(2, len(cluster_col) + 1):
                for combine_name in list(combinations(cluster_col, j)):
                    name_list = [e for e in combine_name]
                    new_col_name = ''
                    for i in name_list:
                        new_col_name = new_col_name + '_' + i
                    demeaned_df[new_col_name] = demeaned_df[name_list[0]].apply(str)
                    for i in range(1, len(name_list)):
                        demeaned_df[new_col_name] = demeaned_df[new_col_name] + '_' + demeaned_df[name_list[i]].apply(
                            str)
                    middle = np.power(-1, j - 1) * middle_term(demeaned_df, consist_col, [new_col_name])
                    m = np.dot(xpx_inv, middle)
                    beta = np.dot(m, xpx_inv)
                    G = np.unique(demeaned_df[new_col_name].values).shape[0]
                    G_array = np.append(G_array, G)
                    beta_list.append(beta)
    # print(G_array)
    m = np.zeros((k, k))
    if c_method == 'cgm':
        for i in beta_list:
            m += i
    elif c_method == 'cgm2':
        for i in beta_list:
            G_MIN = np.min(G_array)
            scale = scale_df * G_MIN / (G_MIN - 1)
            m += i * scale

    if psdef is True and len(cluster_col) > 1:
        m_eigen_value, m_eigen_vector = np.linalg.eig(m)
        m_new_eigen_value = np.maximum(m_eigen_value, 0)
        if (m_eigen_value != m_new_eigen_value).any():
            warnings.warn('Negative eigenvalues set to zero in multi-way clustered variance matrix.')
        m_new = np.dot(np.dot(m_eigen_vector, np.diag(m_new_eigen_value)), m_eigen_vector.T)
        # print('covariance matrix:')
        # print(m_new)
        return m_new
    else:
        # print('covariance matrix:')
        # print(m)
        return m


def is_nested(data_df, category_col, cluster_col, consist_col):
    """

    This function is used to find out whether category variables(fixed effect) is nested in clusters. 'Nested' means
    that once two observation belongs to one category in category variable(e.g. city), it is certain that it belongs to
    same cluster if the cluster variable is province/state/country etc.

    :param data_df: Date with relevant variables
    :param category_col: List of category variables(fixed effect)
    :param cluster_col: List of cluster variables
    :param consist_col: List of continuous variables
    :return: bool value of whether category variables(fixed effect) is nested in clusters
    """
    if len(set(cluster_col) & set(category_col)) > 0:
        return True
    else:
        for cat in category_col:
            flag = True
            unique_val = np.unique(data_df[cat].values).shape[0]
            for cluster in cluster_col:
                count_num = data_df.groupby([cat, cluster]).count()[consist_col[0]]
                if len(count_num) > unique_val:
                    flag = False
                else:
                    flag = True
                    break
            if flag:
                break
        return flag


def middle_term(demean, consist_col, cluster_col):
    """

    This function is used to calculate the middle term during calculating covariance matrix.
    :param demean: Demeaned dataframe with relevant data
    :param consist_col: List of continuous variables
    :param cluster_col: List of cluster variables
    :return: the middle part sum(u_i*u_j*X_i'*X_j) when calculating clustered variance matrix
    """
    consist_copy = consist_col[:]
    consist_copy.append('resid')
    trash = demean.copy()
    consist_num = len(consist_col)
    beta = np.zeros((consist_num, consist_num))
    consist_cache = []
    trash[consist_col] = trash[consist_col].values * trash[['resid']].values
    for i in range(consist_num):
        cache = trash.groupby(cluster_col[0])[[consist_col[i]]].sum().values
        consist_cache.append(cache)
    for i in range(consist_num):
        for j in range(consist_num):
            beta[i][j] = np.sum(consist_cache[i] * consist_cache[j])
    return beta


def min_clust(data_df, cluster_col):
    """

    :param data_df: Dataframe with relevant data
    :param cluster_col: List of cluster variables
    :return: the minimum number of cluster in all kinds of clusters.
            Used when using 'cgm2' to calculate multi-way clustered variance.
    """
    min_c = np.unique(data_df[cluster_col[0]]).shape[0]
    for i in cluster_col:
        min_c = min(min_c, np.unique(data_df[i]).shape[0])
    return min_c



