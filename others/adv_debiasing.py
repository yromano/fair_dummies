
# Code is taken from https://github.com/equialgo/fairness-in-ml
# Original implementation is modified to handle regression and multi-class 
# classification problems

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


from fair_dummies.utility_functions import linear_model
from fair_dummies.utility_functions import deep_model, deep_reg_model



class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()

class Adversary(nn.Module):

    def __init__(self, n_sensitive, n_y, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_y, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


def pretrain_adversary(adv, clf, data_loader, optimizer, criterion, lambdas):
    for x, y, z in data_loader:
        p_y = clf(x).detach()
        adv.zero_grad()
        if len(p_y.size())==1:
            p_y = p_y.unsqueeze(dim=1)
        in_adv = torch.cat((p_y,y),1)
        p_z = adv(in_adv)
        loss = (criterion(p_z, z) * lambdas).mean()
        loss.backward()
        optimizer.step()
    return adv


def train(clf, adv, data_loader, clf_criterion, adv_criterion,
          clf_optimizer, adv_optimizer, lambdas):

    # Train adversary
    for x, y, z in data_loader:
        p_y = clf(x)
        if len(p_y.size())==1:
            p_y = p_y.unsqueeze(dim=1)
        adv.zero_grad()
        in_adv = torch.cat((p_y,y),1)
        p_z = adv(in_adv)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        loss_adv.backward()
        adv_optimizer.step()

    # Train predictor on single batch
    for x, y, z in data_loader:
        p_y = clf(x)
        if len(p_y.size())==1:
            p_y = p_y.unsqueeze(dim=1)
        in_adv = torch.cat((p_y,y),1)
        p_z = adv(in_adv)
        clf.zero_grad()
        p_z = adv(in_adv)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        clf_loss = (1.0-lambdas)*clf_criterion(p_y, y.squeeze().long()) - (adv_criterion(adv(in_adv), z) * lambdas).mean()
        clf_loss.backward()
        clf_optimizer.step()
        break

    return clf, adv


def train_regressor(clf, adv, data_loader, clf_criterion, adv_criterion,
                    clf_optimizer, adv_optimizer, lambdas):

    # Train adversary
    for x, y, z in data_loader:
        p_y = clf(x)
        if len(p_y.size())==1:
            p_y = p_y.unsqueeze(dim=1)
        adv.zero_grad()
        in_adv = torch.cat((p_y,y),1)
        p_z = adv(in_adv)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        loss_adv.backward()
        adv_optimizer.step()

    # Train predictor on single batch
    for x, y, z in data_loader:
        p_y = clf(x)
        if len(p_y.size())==1:
            p_y = p_y.unsqueeze(dim=1)
        in_adv = torch.cat((p_y,y),1)
        p_z = adv(in_adv)
        clf.zero_grad()
        p_z = adv(in_adv)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        clf_loss = (1.0-lambdas)*clf_criterion(p_y.squeeze(), y.squeeze()) - (adv_criterion(adv(in_adv), z) * lambdas).mean()
        clf_loss.backward()
        clf_optimizer.step()
        break

    return clf, adv


def pretrain_classifier(clf, data_loader, optimizer, criterion):
    for x, y, _ in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        loss = criterion(p_y, y.squeeze().long())
        loss.backward()
        optimizer.step()
    return clf

def pretrain_regressor(clf, data_loader, optimizer, criterion):
    for x, y, _ in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        loss = criterion(p_y.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
    return clf

class AdvDebiasingClassLearner:

    def __init__(self,
                 lr,
                 N_CLF_EPOCHS,
                 N_ADV_EPOCHS,
                 N_EPOCH_COMBINED,
                 cost_pred,
                 in_shape,
                 batch_size,
                 model_type,
                 num_classes,
                 lambda_vec):

        self.lr = lr
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.num_classes = num_classes

        self.model_type = model_type
        if self.model_type == "deep_model":
            self.clf = deep_model(in_shape=in_shape, out_shape=num_classes)
        elif self.model_type == "linear_model":
            self.clf = linear_model(in_shape=in_shape, out_shape=num_classes)
        else:
            raise

        self.clf_criterion = cost_pred
        self.clf_optimizer = optim.Adam(self.clf.parameters(),lr=self.lr)

        self.N_CLF_EPOCHS = N_CLF_EPOCHS

        self.lambdas = torch.Tensor([lambda_vec])

        self.adv = Adversary(n_sensitive=1,n_y=num_classes+1)
        self.adv_criterion = nn.BCELoss(reduce=False)
        self.adv_optimizer = optim.Adam(self.adv.parameters(),lr=self.lr)

        self.N_ADV_EPOCHS = N_ADV_EPOCHS

        self.N_EPOCH_COMBINED = N_EPOCH_COMBINED

        self.scaler = StandardScaler()
        self.scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df),
                                                        columns=df.columns,
                                                        index=df.index)




    def fit(self,X,Y):

        # The features are X[:,1:]
        X_train = pd.DataFrame(data=X[:,1:])
        y_train = pd.DataFrame(data=Y)
        Z_train = pd.DataFrame(data=X[:,0])

        self.scaler.fit(X_train)

        X_train = X_train.pipe(self.scale_df, self.scaler)

        train_data = PandasDataSet(X_train, y_train, Z_train)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for epoch in range(self.N_CLF_EPOCHS):
            self.clf = pretrain_classifier(self.clf,
                                          train_loader,
                                          self.clf_optimizer,
                                          self.clf_criterion)



        for epoch in range(self.N_ADV_EPOCHS):
            pretrain_adversary(self.adv,
                               self.clf,
                               train_loader,
                               self.adv_optimizer,
                               self.adv_criterion,
                               self.lambdas)


        for epoch in range(1, self.N_EPOCH_COMBINED):

            self.clf, self.adv = train(self.clf,
                                       self.adv,
                                       train_loader,
                                       self.clf_criterion,
                                       self.adv_criterion,
                                       self.clf_optimizer,
                                       self.adv_optimizer,
                                       self.lambdas)

    def predict(self,X):

        X = X[:,1:]

        X_test = pd.DataFrame(data=X)
        X_test = X_test.pipe(self.scale_df, self.scaler)

        test_data = PandasDataSet(X_test)

        with torch.no_grad():
            Yhat = self.clf(test_data.tensors[0])

        sm = nn.Softmax(dim=1)
        Yhat = sm(Yhat)
        Yhat = Yhat.detach().numpy()

        return Yhat



class AdvDebiasingRegLearner:

    def __init__(self,
                 lr,
                 N_CLF_EPOCHS,
                 N_ADV_EPOCHS,
                 N_EPOCH_COMBINED,
                 cost_pred,
                 in_shape,
                 batch_size,
                 model_type,
                 out_shape,
                 lambda_vec):

        self.lr = lr
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.model_type = model_type
        if self.model_type == "deep_model":
            self.clf = deep_reg_model(in_shape=in_shape, out_shape=out_shape)
        elif self.model_type == "linear_model":
            self.clf = linear_model(in_shape=in_shape, out_shape=out_shape)
        else:
            raise

        self.clf_criterion = cost_pred
        self.clf_optimizer = optim.Adam(self.clf.parameters(),lr=self.lr)

        self.N_CLF_EPOCHS = N_CLF_EPOCHS

        self.lambdas = torch.Tensor([lambda_vec])

        self.adv = Adversary(n_sensitive=1,n_y=out_shape+1)
        self.adv_criterion = nn.BCELoss(reduce=False)
        self.adv_optimizer = optim.Adam(self.adv.parameters(),lr=self.lr)

        self.N_ADV_EPOCHS = N_ADV_EPOCHS

        self.N_EPOCH_COMBINED = N_EPOCH_COMBINED

        self.scaler = StandardScaler()
        self.scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df),
                                                        columns=df.columns,
                                                        index=df.index)


    def fit(self,X,Y):

        # The features are X[:,1:]

        X_train = pd.DataFrame(data=X[:,1:])
        y_train = pd.DataFrame(data=Y)
        Z_train = pd.DataFrame(data=X[:,0])

        self.scaler.fit(X_train)

        X_train = X_train.pipe(self.scale_df, self.scaler)

        train_data = PandasDataSet(X_train, y_train, Z_train)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for epoch in range(self.N_CLF_EPOCHS):
            self.clf = pretrain_regressor(self.clf,
                                          train_loader,
                                          self.clf_optimizer,
                                          self.clf_criterion)



        for epoch in range(self.N_ADV_EPOCHS):
            pretrain_adversary(self.adv,
                               self.clf,
                               train_loader,
                               self.adv_optimizer,
                               self.adv_criterion,
                               self.lambdas)


        for epoch in range(1, self.N_EPOCH_COMBINED):

            self.clf, self.adv = train_regressor(self.clf,
                                                 self.adv,
                                                 train_loader,
                                                 self.clf_criterion,
                                                 self.adv_criterion,
                                                 self.clf_optimizer,
                                                 self.adv_optimizer,
                                                 self.lambdas)

    def predict(self,X):

        X = X[:,1:]
        self.clf.eval()

        X_test = pd.DataFrame(data=X)
        X_test = X_test.pipe(self.scale_df, self.scaler)
        test_data = PandasDataSet(X_test)

        with torch.no_grad():
            Yhat = self.clf(test_data.tensors[0]).squeeze().detach().numpy()

        if self.out_shape==1:
            out = Yhat
        else:
            out = 0*Yhat
            out[:,0] = np.min(Yhat,axis=1)
            out[:,1] = np.max(Yhat,axis=1)

        return out
