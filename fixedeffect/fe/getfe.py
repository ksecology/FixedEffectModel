import numpy as np
import networkx as nx
import pandas as pd
from fixedeffect.utils.Projection import projection
from scipy.stats import t

def getfe(result,
          epsilon=1e-8,
          normalize = False,
          category_input = []):
    """
    This function is used to get fixed effect.
    :param result: result of model after demean
    :param epsilon: tolerance for projection
    :return: dataframe of fixed effect for each category variable
    """


    if (category_input==[]) and (result.category_input==[]):
        raise NameError('no category_input to compute fixedeffect')

    data_df = result.data_df.copy()
    demean = result.demeaned_df.copy()
    coeff = result.params.values
    exog_x = result.exog_x

    if result.endog_x:
        old_x = exog_x + result.endog_x
        if 'const' in result.x_second_stage:
            old_x = ['const'] + old_x
    else:
        old_x = exog_x

    category_col = result.category_input
    treatment_input = result.treatment_input
    
    #if no category_input,default all category_col
    if category_input == []:
        category_col_specified = category_col
        if treatment_input:
            if treatment_input['effect']=='group':
                category_col_specified = [category_col[1]]
    else: #calculate user specify fe
        category_col_specified = category_input


    out_col = result.dependent
    df = result.df
    vc = result.variance_matrix

    #loop through old_x to see if const is included, if not add
    k0 = 0
    if any('const' in s for s in old_x):
        k0 = 1
        data_df['const'] = 1            

    #-------------------------- begin if normalize is true --------------------------#
    #-------------------------- match result of R package flm  ----------------------#
    if normalize is True: 
        y = data_df[out_col[0]].values
        b_x = np.dot(coeff, data_df[old_x].values.T)
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
        col_cat_value = ['cat_vale']
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
        
        #finally output subset if user specify category_input option
        if category_input != []:
            output_dummy_df = pd.DataFrame([]) 
            for cat_specified in category_col_specified:
                filtered_df  = dummy_df[(dummy_df['cat'] == cat_specified)]
                output_dummy_df = pd.concat([output_dummy_df, filtered_df])
        else:
            output_dummy_df = dummy_df
    # -------------------------- finish if normalize is true --------------------------#
    else:
        # -------------------------- begin if normalize is false --------------------------#
        # --------------------- match R package plm and Greene's textbook formula ---------#
        demean_y = demean[out_col[0]].values
        demean_b_x = np.dot(coeff, demean[old_x].values.T)
        ori_resid = demean_y - demean_b_x
        sigma_e2 = (ori_resid ** 2).sum() / df

        c_i_name_all = []
        c_i_all = []
        se_c_i_all = []
        for cat in category_col_specified:
            mean_df = data_df.groupby(cat)[old_x + out_col].transform('mean')
            mean_df[cat] = data_df[cat]
            group_id = pd.DataFrame(data_df[cat].value_counts().sort_index())
            group_mean = mean_df.groupby(cat).first()
            x_group_mean = pd.DataFrame([])
            # this is to process duplicate independent variable
            for xvar in old_x:
                if isinstance(group_mean[xvar], pd.Series) == True:
                    a = group_mean[xvar]
                else:
                    a = group_mean[xvar].iloc[:, 0].to_frame()
                x_group_mean = pd.concat((x_group_mean, a), axis=1)
            y_group_mean = group_mean[out_col[0]]

            # add dummy variables name
            c_i_name = pd.DataFrame([])
            group_name = group_id.columns.tolist()
            c_i_name['dummy_name'] = group_id.index
            c_i_name['dummy_name'] = group_name[0] + c_i_name['dummy_name'].astype(str)

            x_mean_b = np.dot(coeff, x_group_mean.values.T)
            c_i = y_group_mean.values - x_mean_b

            # xe = np.dot(x_group_mean.values.astype(np.float32), vc.values.astype(np.float32))
            xe = np.dot(x_group_mean.values.astype(np.float32), vc.astype(np.float32))
            ngroup = xe.shape[0]

            xex = np.zeros((ngroup, 1))
            B = x_group_mean.values.astype(np.float32).T
            for i in range(xex.shape[0]):
                xex[i, :] = xe[i, :].dot(B[:, i])
            se_c_i = np.zeros((ngroup, 1))
            for i in range(xex.shape[0]):
                se_c_i[i, 0] = np.sqrt(xex[i, 0] + sigma_e2 / group_id[cat].iloc[i])

            # xe = np.dot(x_group_mean.values, vc)
            # xex = np.dot(xe, x_group_mean.values.T)
            # v_c_i = np.zeros((group_id.shape[0],group_id.shape[0]))
            # for i in range(xex.shape[1]):
            #    v_c_i[:,i] = xex[:,i]+(sigma_e2/group_id[cat])
            # se_c_i = np.sqrt(np.diag(v_c_i))
            c_i_all = np.append(c_i_all, c_i)
            se_c_i_all = np.append(se_c_i_all, se_c_i)
            c_i_name_all = np.append(c_i_name_all, c_i_name)


        dummy_df = pd.DataFrame([])    
        dummy_df['dummy_name'] = c_i_name_all
        dummy_df['effect'] = c_i_all
        dummy_df['s.e'] = se_c_i_all
        dummy_df['t-value'] = c_i_all/se_c_i_all
        dummy_df['P>|t|'] = pd.Series(2 * t.sf(np.abs(dummy_df['t-value']), df))
        
        output_dummy_df = dummy_df

    return output_dummy_df
