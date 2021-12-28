import numpy as np
import pandas as pd
import random

def gen_data(N, T, beta, ate, exp_date):
    
    """

    :N: number of cross sections
    :T: number of time periods
    :beta: true beta
    :ate: true ate
    :exp_date: experiment date
    
    """    
    if exp_date > T:
        raise NameError('exp_date must be smaller than the number of time periods')
        
    if len(beta) <= 1:
        raise NameError('length of array beta must be greater than or equal to 2')
    
    k = len(beta)
    id_list = np.linspace(1, N, num = N)
    time_list = np.linspace(1, T, num = T)

    np.random.seed(0)
    error    = np.random.normal(0, 1, N*T)
    t_effect = np.random.normal(ate, 1, N*T)

    c_i = np.random.normal(0, 1, N)
    a_t = np.random.normal(0, 1, T)
    x1 = np.random.normal(0, 1, N*T*(k-1))
    x1.shape = (N*T, (k-1))
    const = np.ones(N*T)
    const.shape = (N*T, 1)
    x = np.concatenate((const, x1), axis=1)


    j_N = np.ones(N)
    time_full = np.kron(j_N,time_list)
    a_t_full = np.kron(j_N, a_t)
    id_full = np.repeat(id_list, T, axis=0)
    c_i_full = np.repeat(c_i, T, axis=0)
    treatment_group = np.random.choice(id_list,round(N/4), replace=False)


    df = pd.DataFrame(data = x)
    for name in df.columns:
        new_name = "x_"+str(name)
        df = df.rename(columns={name: new_name})  

    beta = np.reshape(beta, (-1, k))    
    df['xb'] = np.dot(x,beta.T)

    df['id']   = id_full
    df['time'] = time_full
    df['c_i']  = c_i_full
    df['a_t']  = a_t_full
    df['error']  = error

    df['post'] = (df['time']>=exp_date) * 1
    df['treatment'] = df['id'].apply(lambda x: 1 if x in treatment_group else 0)
    
    
    df['y'] = df['xb'] + ate*df['treatment']*df['post'] + df['c_i'] + df['a_t'] + df['error']
    return df


def gen_panel_data(N, T, beta, ate, exp_date):
    
    """

    :N: number of cross sections
    :T: number of time periods
    :beta: true beta
    :ate: true ate
    :exp_date: experiment date
    
    """    
    if exp_date > T:
        raise NameError('exp_date must be smaller than the number of time periods')
        
    if len(beta) <= 1:
        raise NameError('length of array beta must be greater than or equal to 2')
    
    k = len(beta)
    id_list = np.linspace(1, N, num = N)
    time_list = np.linspace(1, T, num = T)

    np.random.seed(0)
    error    = np.random.normal(0, 1, N*T)
    t_effect = np.random.normal(ate, 1, N*T)

    c_i = np.random.normal(0, 1, N)
    a_t = np.random.normal(0, 1, T)
    x1 = np.random.normal(0, 1, N*T*(k-1))
    x1.shape = (N*T, (k-1))
    const = np.ones(N*T)
    const.shape = (N*T, 1)
    x = np.concatenate((const, x1), axis=1)


    j_N = np.ones(N)
    time_full = np.kron(j_N,time_list)
    a_t_full = np.kron(j_N, a_t)
    id_full = np.repeat(id_list, T, axis=0)
    c_i_full = np.repeat(c_i, T, axis=0)
    treatment_group = np.random.choice(id_list,round(N/4), replace=False)


    df = pd.DataFrame(data = x)
    for name in df.columns:
        new_name = "x_"+str(name)
        df = df.rename(columns={name: new_name})  

    beta = np.reshape(beta, (-1, k))    
    df['xb'] = np.dot(x,beta.T)

    df['id']   = id_full
    df['time'] = time_full
    df['c_i']  = c_i_full
    df['a_t']  = a_t_full
    df['error']  = error

    df['post'] = (df['time']>=exp_date) * 1
    df['treatment'] = df['id'].apply(lambda x: 1 if x in treatment_group else 0)
    
    
    df['y'] = df['xb'] + ate*df['treatment']*df['post'] + df['c_i'] + df['a_t'] + df['error']
    return df

def gen_data_causal_engine(N = 50,
                           T = 5,
                           beta = [-3,1],
                           alpha = 0.9,
                           ate = 1,
                           exp_date = 3,
                           b_unbalance = False,
                           b_dynamic = False,
                           b_twoway = False,
                           unbalance_frac = 0.1):

    """
    :N: number of cross sections
    :T: number of time periods
    :beta: true beta
    :alpha: coefficient for lag_y, works only when b_dynamic=True
    :ate: true ate
    :exp_date: experiment date
    :b_unbalance: if panel is unbalanced, i.e, if contains missing
    :b_dynamic: if dgp is dynamic, i.e, if lag y in dgp
    """
    if exp_date > T:
        raise NameError('exp_date must be smaller than the number of time periods')

    if len(beta) <= 1:
        raise NameError('length of array beta must be greater than or equal to 2')


    k = len(beta)
    id_list = np.linspace(1, N, num=N)
    time_list = np.linspace(1, T, num=T)

    np.random.seed(0)
    error = np.random.normal(0, 1, N * T)
    t_effect = np.random.normal(ate, 1, N * T)

    c_i = np.random.normal(0, 1, N)
    a_t = np.random.normal(0, 1, T)
    x1 = np.random.normal(0, 1, N * T * (k - 1))
    x1.shape = (N * T, (k - 1))
    const = np.ones(N * T)
    const.shape = (N * T, 1)
    x = np.concatenate((const, x1), axis=1)

    j_N = np.ones(N)
    j_T = np.ones(T)

    time_full = np.kron(j_N, time_list)
    a_t_full = np.kron(j_N, a_t)
    id_full = np.repeat(id_list, T, axis=0)
    c_i_full = np.repeat(c_i, T, axis=0)
    treatment_group = np.random.choice(id_list, round(N / 4), replace=False)

    rand_start_time = random.choices(time_list, k=len(id_list))
    start_time_full = np.kron(rand_start_time,j_T)

    df = pd.DataFrame(data=x)
    for name in df.columns:
        new_name = "x_" + str(name)
        df = df.rename(columns={name: new_name})

    beta = np.reshape(beta, (-1, k))
    df['xb'] = np.dot(x, beta.T)

    df['id'] = id_full
    df['time'] = time_full
    df['time_numercial'] = time_full
    df['c_i'] = c_i_full
    df['a_t'] = a_t_full
    df['error'] = error

    df['post'] = (df['time'] >= exp_date) * 1

    df['exp_rand_start'] = start_time_full
    df['post_rand'] = (df['time'] >= df['exp_rand_start']) * 1

    df['treatment'] = df['id'].apply(lambda x: 1 if x in treatment_group else 0)
    df['treatment_did'] = df['treatment']* df['post']
    df['treatment_rand'] = df['treatment']* df['post_rand']

    if b_twoway==True:
        df['y'] = df['xb'] + ate * df['treatment'] * df['post'] + df['c_i'] + df['a_t'] + df['error']
    else:
        df['y'] = df['xb'] + ate * df['treatment'] * df['post'] + df['c_i'] + df['error']

    if b_dynamic:
        df_y = pd.DataFrame()
        for i_index, i in enumerate(id_list):
            y_i = np.zeros(T)
            df_sub = df[df['id'] == i]
            xb = df_sub['xb'].tolist()
            e_i = df_sub['error'].tolist()
            for tt, j in enumerate(time_list):
                if tt == 0:
                    y_i[0] = 0
                #elif tt == 1:
                #    y_i[tt] = xb[tt - 1]
                else:
                    y_i[tt] = alpha * y_i[tt - 1] + xb[tt] + e_i[tt] + c_i[i_index]

            df_y = df_y.append(pd.DataFrame(y_i))

        df_y = df_y.reset_index()
        df['y'] = df_y[0]
        df['lag_1'] = df.groupby('id')['y'].shift(1)
        df = df.dropna()


    df2 = df.copy()
    for tt in time_list:
        if tt < 10:
            time_str = "date_0" + str(int(tt))
        else:
            time_str = "date_" + str(int(tt))
        df2 = df2.replace({'time': tt}, time_str)

    if b_unbalance:
        np.random.seed(10)
        drop_indices = np.random.choice(df.index, N * T - np.round(N * T * unbalance_frac, 0).astype(int), replace=False)
        df2 = df2.drop(drop_indices)

    return df2
