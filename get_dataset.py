import os
import urllib
import numpy as np
import pandas as pd

            
def gen_synthetic_data(n = 6000, seed=0):
    
    t0 = np.random.get_state()
    np.random.seed(seed)
    
    p0 = 0.1
    p1 = 1 - p0

    sigX_small = 1
    sigX_large = 3
    beta_eo = [sigX_large/2, sigX_large/2]

    A = np.random.binomial(1,p1,n).T
    X = np.random.randn(n,2)
    X[A==0,0] = X[A==0,0] * sigX_small
    X[A==0,1] = X[A==0,1] * sigX_large
    X[A==1,0] = X[A==1,0] * sigX_large
    X[A==1,1] = X[A==1,1] * sigX_small

    beta0 = [0,sigX_large]
    beta1 = [sigX_large,0]

    Y = np.random.randn(n)
    Y[A==0] = Y[A==0] + np.dot(X[A==0],beta0)
    Y[A==1] = Y[A==1] + np.dot(X[A==1],beta1)

    x_axis_0 = np.dot(X[A==0],beta0)
    x_axis_1 = np.dot(X[A==1],beta1)
    
    Y_eo = np.dot(X,beta_eo)
    
    np.random.set_state(t0)
    
    return X, A, Y, x_axis_0, x_axis_1, Y_eo
            
def read_meps_data(base_path):
    
    df = pd.read_csv(base_path + 'meps_21_reg_fix.csv')
    column_names = df.columns
    response_name = "UTILIZATION_reg"
    column_names = column_names[column_names!=response_name]
    column_names = column_names[column_names!="Unnamed: 0"]

    col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
               'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
               'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
               'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
               'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
               'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
               'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
               'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
               'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
               'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
               'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
               'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
               'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
               'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
               'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
               'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
               'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
               'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
               'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
               'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
               'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
               'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
               'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
               'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
               'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
               'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
               'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

    
    Y = df[response_name].values
    Y = np.log(1 + Y - min(Y))
    A = df['RACE'].values
    X = df[col_names[:-1]].values # drop race
    
    return X, A, Y
            
    
            
def read_crimes_data(base_path):
    threshold_a = 0.1
    label='ViolentCrimesPerPop'
    sensitive_attribute='racepctblack'

    if not os.path.isfile(base_path + "communities.data"):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", base_path + "communities.data")
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            base_path + "communities.names")

    # create names
    names = []
    with open(base_path + 'communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv(base_path + 'communities.data', names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)

    y = data[label].values
    to_drop += [label]

    z = (data[sensitive_attribute].values <= threshold_a).astype(float)
    #z = data[sensitive_attribute].values

    to_drop += [sensitive_attribute]

    data.drop(to_drop + [label], axis=1, inplace=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    x = np.array(data.values)
    
    return x, z, y

def get_nursery(base_path):
    
    df = pd.read_csv(base_path + 'nursery_processed.csv')
    
    protected = "finance"
    vars_to_drop = [protected, "class"]
    target = ["class"]
    X = df.drop(vars_to_drop, axis = 1).values
    target = df[target]
    Y = pd.DataFrame(target).values
    Y = Y.squeeze()
    
    A = df[protected].values
    A = np.array(A>0).astype(float)
    A = A.squeeze()
    
    return X, A, Y

def get_german(base_path):

    all_data = pd.read_csv(base_path + 'german_processed.csv')
    
    protected = "statussex"
    label = "credithistory"
    vars_to_drop = [protected, label]
 
    all_data[ all_data['credithistory'] == 0 ] = 1
    
    X = all_data.drop(vars_to_drop, axis=1).values
    A = all_data[protected].values
    Y = all_data[label].values - 1
    
    return X, A, Y
            
def get_train_test_data(base_path, dataset, seed):
    if dataset == "meps":
        X_, A_, Y_ = read_meps_data(base_path)
        n_train = int(Y_.shape[0]*0.6)
        n_train = n_train - n_train%2
        n_cal = int( (Y_.shape[0]-n_train) / 2)
        
    elif dataset == "crimes":
        X_, A_, Y_ = read_crimes_data(base_path)
        n_train = int(Y_.shape[0]*0.6)
        n_train = n_train - n_train%2
        n_cal = int( (Y_.shape[0]-n_train) / 2)        

    elif dataset == "nursery":
        X_, A_, Y_ = get_nursery(base_path)
        n_train = int(Y_.shape[0]*0.6) #0.8
        n_train = n_train - n_train%2
        n_cal = int( (Y_.shape[0]-n_train) / 2)            

        
    t0 = np.random.get_state()
    np.random.seed(seed)
    all_inds = np.random.permutation(Y_.shape[0])
    np.random.set_state(t0)
    
    inds_train = all_inds[:n_train]
    inds_cal = all_inds[n_train:n_train+n_cal]
    inds_test = all_inds[n_train+n_cal:]

    X = X_[inds_train]
    A = A_[inds_train]
    Y = Y_[inds_train]
    
    X_cal = X_[inds_cal]
    A_cal = A_[inds_cal]
    Y_cal = Y_[inds_cal]
    
    X_test = X_[inds_test]
    A_test = A_[inds_test]
    Y_test = Y_[inds_test]

    return X, A, Y, X_cal, A_cal, Y_cal, X_test, A_test, Y_test