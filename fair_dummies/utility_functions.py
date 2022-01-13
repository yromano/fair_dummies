import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import csr_matrix

# Estimate the distribusion of P{A|Y}
def density_estimation(Y, A, Y_test=[]):

    bandwidth = np.sqrt( max(np.median(np.abs(Y)), 0.01))

    kde_0 = KernelDensity(kernel='linear', bandwidth=bandwidth).fit(Y[A==0][:, np.newaxis])
    kde_1 = KernelDensity(kernel='linear', bandwidth=bandwidth).fit(Y[A==1][:, np.newaxis])

    log_dens_0 = np.exp(np.squeeze(kde_0.score_samples(Y[:, np.newaxis])))
    log_dens_1 = np.exp(np.squeeze(kde_1.score_samples(Y[:, np.newaxis])))
    p_0 = np.sum(A==0) / A.shape[0]
    p_1 = 1 - p_0

    # p(A=1|y) = p(y|A=1)p(A=1) / (p(y|A=1)p(A=1) + p(y|A=0)p(A=0))
    p_success = (log_dens_1*p_1) / (log_dens_1*p_1 + log_dens_0*p_0 + 1e-10)

    p_success_test = []
    if len(Y_test)>0:
        log_dens_0_test = np.exp(np.squeeze(kde_0.score_samples(Y_test[:, np.newaxis])))
        log_dens_1_test = np.exp(np.squeeze(kde_1.score_samples(Y_test[:, np.newaxis])))
        p_success_test = (log_dens_1_test*p_1) / (log_dens_1_test*p_1 + log_dens_0_test*p_0 + 1e-10)

    return p_success, p_success_test


# Define linear model
class linear_model(torch.nn.Module):
    def __init__(self,
                 in_shape=1,
                 out_shape=2):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.build_model()

    def build_model(self):
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.out_shape, bias=True),
        )

    def forward(self, x):
        return torch.squeeze(self.base_model(x))

# Define deep neural net model for classification
class deep_model(torch.nn.Module):
    def __init__(self,
                 in_shape=1,
                 out_shape=1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64
        self.dropout = 0.5
        self.out_shape = out_shape
        self.build_model()

    def build_model(self):
        self.base_model = nn.Sequential(
                nn.Linear(self.in_shape, self.dim_h, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim_h, self.out_shape, bias=True),
        )

    def forward(self, x):
        return torch.squeeze(self.base_model(x))


# Define deep model for regression
class deep_reg_model(torch.nn.Module):
    def __init__(self,
                 in_shape=1,
                 out_shape=1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64 #in_shape*10
        self.out_shape = out_shape
        self.build_model()

    def build_model(self):
        self.base_model = nn.Sequential(
                nn.Linear(self.in_shape, self.dim_h, bias=True),
                nn.ReLU(),
                nn.Linear(self.dim_h, self.out_shape, bias=True),
        )

    def forward(self, x):
        return torch.squeeze(self.base_model(x))

# Define deep regression model, used by the fair dummies test
class deep_proba_model(torch.nn.Module):
    def __init__(self,
                 in_shape=1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64 #in_shape*10
        self.dropout = 0.5
        self.out_shape = 1
        self.build_model()

    def build_model(self):
        self.base_model = nn.Sequential(
                nn.Linear(self.in_shape, self.dim_h, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim_h, self.out_shape, bias=True),
                nn.Sigmoid(),
        )

    def forward(self, x):
        return torch.squeeze(self.base_model(x))

def calc_accuracy(outputs,Y): #Care outputs are going to be in dimension 2
    max_vals, max_indices = torch.max(outputs,1)
    acc = (max_indices == Y).sum().detach().cpu().numpy()/max_indices.size()[0]
    return acc

def compute_acc(Yhat,Y):
    _, predicted = torch.max(Yhat, 1)
    total = Y.size(0)
    correct = (predicted == Y).sum().item()
    acc = correct/total
    return acc

def compute_acc_numpy(Yhat,Y):
    Yhat = torch.from_numpy(Yhat)
    Y = torch.from_numpy(Y)

    return compute_acc(Yhat,Y)

def pytorch_standard_scaler(x):
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x

# fit a neural netwok on a given data, used by the fair dummies test
class GeneralLearner:

    def __init__(self,
                 lr,
                 epochs,
                 cost_func,
                 in_shape,
                 batch_size,
                 model_type,
                 out_shape = 1):

        # input dim
        self.in_shape = in_shape

        # output dim
        self.out_shape = out_shape

        # Data normalization
        self.X_scaler = StandardScaler()

        # learning rate
        self.lr = lr

        # number of epochs
        self.epochs = epochs

        # cost to minimize
        self.cost_func = cost_func

        # define a predictive model
        self.model_type = model_type
        if self.model_type == "deep_proba":
            self.model = deep_proba_model(in_shape=in_shape)
        elif self.model_type == "deep_regression":
            self.model = deep_model(in_shape=in_shape, out_shape=self.out_shape)
        else:
            raise

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        # minibatch size
        self.batch_size = batch_size

    # fit a model by sweeping over all data points
    def internal_epoch(self,X_,Y_):

        # shuffle data
        shuffle_idx = np.arange(X_.shape[0])
        np.random.shuffle(shuffle_idx)
        X = X_.clone()[shuffle_idx]
        Y = Y_.clone()[shuffle_idx]

        # fit pred func
        self.model.train()

        batch_size = self.batch_size
        epoch_losses = []

        for idx in range(0, X.shape[0], batch_size):
            self.optimizer.zero_grad()

            batch_x = X[idx : min(idx + batch_size, X.shape[0]),:]
            batch_y = Y[idx : min(idx + batch_size, Y.shape[0])]

            # utility loss
            batch_Yhat = self.model(batch_x)
            loss = self.cost_func(batch_Yhat,batch_y)

            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.cpu().detach().numpy())

        epoch_loss = np.mean(epoch_losses)
        return epoch_loss

    def run_epochs(self,X,Y):

        for epoch in range(self.epochs):
            epoch_loss = self.internal_epoch(X,Y)

    # fit a model on training data
    def fit(self,X,Y):

        self.X_scaler.fit(X)

        Xp = torch.from_numpy(self.X_scaler.transform(X)).float()
        Yp = torch.from_numpy(Y).float()

        # evaluate at init
        self.model.eval()
        Yhat = self.model(Xp)

        print('Init Loss = ' + str(self.cost_func(Yhat, Yp).detach().numpy()))


        self.model.train()
        self.run_epochs(Xp,Yp)

        # evaluate
        self.model.eval()
        Yhat = self.model(Xp)

        print('Final Loss = ' + str(self.cost_func(Yhat, Yp).detach().numpy()))


    def predict(self,X):

        self.model.eval()

        Xp = torch.from_numpy(self.X_scaler.transform(X)).float()
        Yhat = self.model(Xp)
        Yhat = Yhat.detach().numpy()

        return Yhat


# run the fair dummies test
# Yhat_cal, A_cal, Y_cal: are used to fit a model that formulates the test statistics
# Yhat, A, Y: variables in which we test whether Yhat is indpendent on A given Y
def fair_dummies_test_regression(Yhat_cal,
                                 A_cal,
                                 Y_cal,
                                 Yhat,
                                 A,
                                 Y,
                                 num_reps=1,
                                 num_p_val_rep=1000,
                                 reg_func_name="Net",
                                 lr=0.1,
                                 return_vec=False):

    p_success, dummy = density_estimation(np.concatenate((Y_cal,Y),0),np.concatenate((A_cal,A),0))
    p_success = p_success[Y_cal.shape[0]:]

    out_shape = 1
    if len(Yhat.shape) > 1:
        out_shape = Yhat.shape[1]


    Y_cal = Y_cal[:,np.newaxis]
    Y = Y[:,np.newaxis]

    test_i = []
    for i in range(num_reps):

        # fit regressor
        if reg_func_name == "RF":
            regr = RandomForestRegressor(n_estimators = 10)
        elif reg_func_name == "Net":
            regr = GeneralLearner(lr=lr,
                                  epochs=200,
                                  cost_func=nn.MSELoss(),
                                  in_shape=2,
                                  batch_size=128,
                                  model_type="deep_regression",
                                  out_shape=out_shape)

        features_cal = np.concatenate((A_cal[:,np.newaxis],Y_cal),1)
        regr.fit(features_cal, Yhat_cal)

        # compute error on holdout points
        features_orig = np.concatenate((A[:,np.newaxis], Y),1)
        output_orig = regr.predict(features_orig)
        est_orig_err = np.mean((Yhat - output_orig)**2)

        # generate A and compare
        est_fake_err = np.zeros(num_p_val_rep)
        for inter_p_value in range(num_p_val_rep):
            random_array = np.random.uniform(low=0.0, high=1.0, size=A.shape)
            A_tilde = (random_array < p_success).astype(float)

            features_fake = np.concatenate((A_tilde[:,np.newaxis],Y),1)
            output_fake = regr.predict(features_fake)
            est_fake_err[inter_p_value] = np.mean((Yhat - output_fake)**2)

        p_val = 1.0/(num_p_val_rep+1) * (1 + sum(est_orig_err >= est_fake_err))

        test_i.append(p_val)


    print("Fair dummies test (regression score), p-value:", np.mean(test_i)) # should be uniform under ind.

    out = test_i[0]
    if return_vec:
        out = test_i

    return out


def classification_score(y_hat,y):
    assert(y <= len(y_hat))
    prob = y_hat[int(y)]
    return prob

# run the fair dummies test
# Yhat_cal, A_cal, Y_cal: are used to fit a model that formulates the test statistics
# Yhat, A, Y: variables in which we test whether Yhat is indpendent on A given Y
def fair_dummies_test_classification(Yhat_cal,
                                     A_cal,
                                     Y_cal,
                                     Yhat,
                                     A,
                                     Y,
                                     num_reps=10,
                                     num_p_val_rep=1000,
                                     reg_func_name="Net"):

    p_success, dummy = density_estimation(np.concatenate((Y_cal,Y),0),np.concatenate((A_cal,A),0))
    p_success = p_success[Y_cal.shape[0]:]

    Yhat_cal_score = np.array([classification_score(Yhat_cal[i],Y_cal[i]) for i in range(Yhat_cal.shape[0])], dtype = float)
    Yhat_score = np.array([classification_score(Yhat[i],Y[i]) for i in range(Y.shape[0])], dtype = float)

    def get_dummies(labels):
        num_datapoints = len(labels)
        row_ind = np.arange(num_datapoints)
        return csr_matrix((np.ones(num_datapoints), (row_ind, labels)), dtype=float).todense()

    Y_cal = get_dummies(Y_cal)
    Y = get_dummies(Y)

    test_i = []
    err_func = nn.BCELoss()
    for i in range(num_reps):

        features_cal = np.concatenate((A_cal[:,np.newaxis],Y_cal),1)

        # fit regressor
        if reg_func_name == "RF":
            regr = RandomForestRegressor(n_estimators = 10)
        elif reg_func_name == "Net":
            regr = GeneralLearner(lr=0.1,
                                  epochs=200,
                                  cost_func=nn.BCELoss(),
                                  in_shape=features_cal.shape[1],
                                  batch_size=128,
                                  model_type="deep_proba")

        regr.fit(features_cal, Yhat_cal_score)

        # compute error on holdout points
        features_orig = np.concatenate((A[:,np.newaxis], Y),1)
        output_orig = regr.predict(features_orig)

        if reg_func_name == "RF":
            est_orig_err = np.mean((Yhat_score - output_orig)**2)
        elif reg_func_name == "Net":
            est_orig_err = err_func(torch.from_numpy(output_orig).float(),
                                    torch.from_numpy(Yhat_score).float()).detach().cpu().numpy()

        # generate A and compare
        est_fake_err = np.zeros(num_p_val_rep)
        for inter_p_value in range(num_p_val_rep):
            random_array = np.random.uniform(low=0.0, high=1.0, size=A.shape)
            A_tilde = (random_array < p_success).astype(float)

            features_fake = np.concatenate((A_tilde[:,np.newaxis],Y),1)
            output_fake = regr.predict(features_fake)

            if reg_func_name == "RF":
                est_fake_err[inter_p_value] = np.mean((Yhat_score - output_fake)**2)
            elif reg_func_name == "Net":
                est_fake_err[inter_p_value] = err_func(torch.from_numpy(output_fake).float(),
                                                       torch.from_numpy(Yhat_score).float()).detach().cpu().numpy()


        p_val = 1.0/(num_p_val_rep+1) * (1 + sum(est_orig_err >= est_fake_err))

        test_i.append(p_val)


    print("Fair dummies test (classification score), p-value:", np.mean(test_i)) # should be uniform under ind.

    return test_i[0]
