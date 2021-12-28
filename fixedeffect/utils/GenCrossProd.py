import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from .DemeanDataframe import demean_dataframe,demeanonex

#this function generates cross product terms in consist variables
# that user requires
def gencrossprod(data_df, consist_col):
    """
    :data_df:  Dataframe of relevant data
    :consist_col: Lists of consist_col
    :status: if 0, generate cross product for exogeneous variables in consist cols
             if 1, generate dummies for time variable and its cross product with treatment
    """
    pos = 0
    str3 = '*'
    for p3 in consist_col:
        pos += 1
        if str3 in p3:
            bool_crossprod = True
            p4 = p3.split('*')  
            cross_prod_all_cols = data_df[p4[0]]#initialize the product result to the 1st column
            for p5 in p4[1:]:            
                cross_prod_all_cols = cross_prod_all_cols*data_df[p5]
            
            data_df["p3"]= cross_prod_all_cols  
            #change name of the column        
            new = p3        
            data_df = data_df.rename(columns={'p3':new})
        else:
            bool_crossprod = False                
     
    return data_df

# this function only generate cross product terms in consist variables
def gencrossprod_dataset(data_df,
                         out_col,
                         consist_col,
                         category_col,
                         treatment,
                         exp_date,
                         did_effect,
                         no_print = False,
                         figsize = (2,1),
                         fontsize = 15):

    csid = category_col[0]
    tsid = category_col[1]
    treatment_col = treatment[0]

    if did_effect=='treatment':
        warnings.warn('You are doing DID with group effect where group is exp or base')
    else:
        warnings.warn('You are doing DID with individual effect')

    #generate post experiment dummy
    data_df["post_experiment"] = data_df[tsid] >= exp_date
    data_df["post_experiment"] = data_df["post_experiment"] * 1

    #generate treatment*post_experiment
    data_df[str(treatment_col)+"*post_experiment"] = data_df["post_experiment"]*data_df[treatment_col]
    tsid_items = list(data_df[tsid].unique())
    tsid_items.sort()

    #generate the time dummies
    time_dummies = pd.get_dummies(data_df[tsid])
    if exp_date in tsid_items:
        pre_treatment_date = tsid_items[tsid_items.index(exp_date) - 1]
    else:
        tsid_items_ = tsid_items + [exp_date]
        tsid_items_.sort()
        pre_treatment_date = tsid_items_[tsid_items_.index(exp_date) - 1]

    time_dummies = time_dummies.drop(pre_treatment_date,axis = 1)
    time_dummies = time_dummies.sort_index(axis = 1)

    #generate the time dummies and its interaction with treatment_group_dummy
    time_dummies_interactions = pd.DataFrame([])
    for name in time_dummies.columns:
        time_dummies_interactions[str(name) + "*treatment"] = time_dummies[name]*data_df[treatment_col]
           
    if did_effect=='group':
        X = pd.concat([data_df[consist_col], time_dummies, time_dummies_interactions, data_df[treatment_col]], axis = 1)
        X = sm.add_constant(X)        
    else:
        #plot based on csid instead of treatment_col        
        df_plot_id = pd.concat([data_df, time_dummies,time_dummies_interactions], axis = 1)
        interaction_list = time_dummies_interactions.columns.tolist()
        time_dummies_list = time_dummies.columns.tolist()
        #obtain the list of variables you want to demean on 
        demean_list = interaction_list + time_dummies_list + out_col + consist_col
        
        #demean on csid, equal to including csid dummies  
        demeaned_df = demean_dataframe(df_plot_id, demean_list, [csid])
        #X doesn't include constant here due to multicolinearity
        X = pd.concat([demeaned_df[interaction_list + time_dummies_list + consist_col]], axis = 1)  
         
        
    y = data_df[out_col]
    model_plot = sm.OLS(y,X).fit()
    
    xvalue = list(time_dummies)
    x = range(len(xvalue))
    param, se = [], []
    param.extend(list(model_plot.params[list(time_dummies_interactions)]))
    se.extend(list(model_plot.bse[list(time_dummies_interactions)]))

    if no_print==False:
        plt.figure(figsize=figsize)
        #plt.title('parallel check of ' + out_col[0])

        ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
        ax.set_title('parallel check of ' + out_col[0])

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
        ax.errorbar(xvalue, param, se, marker='^', capsize=5, elinewidth=2, markeredgewidth=2)

        plt.xticks(rotation=60)

        plt.grid(True)
        plt.show()
        plt.close()
    
    #newvar_list = str(treatment_col)+"*post_experiment"
    
    return data_df
