
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from fair_dummies import utility_functions
from fair_dummies.utility_functions import linear_model
from fair_dummies.utility_functions import deep_model, deep_reg_model

def covariance_diff_biased(Z, W, scale=1.0):

    # Center X,Xk
    mZ  = Z  - torch.mean(Z,0,keepdim=True)
    mW = W - torch.mean(W,0,keepdim=True)
    # Compute covariance matrices
    SZZ = torch.mm(torch.t(mZ),mZ)/mZ.shape[0]
    SWW  = torch.mm(torch.t(W),mW)/mW.shape[0]

    # Compute loss
    T  = (SZZ-SWW).pow(2).sum() / scale
    return T

class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()

# defining discriminator class (for regression)
class reg_discriminator(nn.Module):

    def __init__(self, inp, out=1):

        super(reg_discriminator, self).__init__()
        self.net = nn.Sequential(
                                 nn.Linear(inp,10*inp),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(10*inp,out),
                                 nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.net(x)
        return x

# defining discriminator class (for classification)
class class_discriminator(nn.Module):

    def __init__(self, out_dim, n_y, n_hidden=32):
        super(class_discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_y, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, out_dim),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


def pretrain_adversary_fast_loader(dis, model, x, y, a, at, optimizer, criterion, lambdas):
    yhat = model(x).detach()
    dis.zero_grad()
    if len(yhat.size())==1:
        yhat = yhat.unsqueeze(dim=1)
    real = torch.cat((yhat,at,y),1)
    fake = torch.cat((yhat,a,y),1)
    in_dis = torch.cat((real, fake), 0)
    out_dis = dis(in_dis)
    labels = torch.cat((torch.ones(real.shape[0],1), torch.zeros(fake.shape[0],1)), 0)
    loss = (criterion(out_dis, labels) * lambdas).mean()
    loss.backward()
    optimizer.step()
    return dis

def pretrain_adversary(dis, model, data_loader, optimizer, criterion, lambdas):
    for x, y, a, at in data_loader:
        dis = pretrain_adversary_fast_loader(dis,
                                             model,
                                             x,
                                             y,
                                             a,
                                             at,
                                             optimizer,
                                             criterion,
                                             lambdas)
    return dis



def pretrain_classifier(model, data_loader, optimizer, criterion):
    for x, y, _, _ in data_loader:
        model.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y.squeeze().long())
        loss.backward()
        optimizer.step()
    return model

def pretrain_regressor(model, data_loader, optimizer, criterion):
    for x, y, _, _ in data_loader:
        model.zero_grad()
        yhat = model(x)
        loss = criterion(yhat.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
    return model

def pretrain_regressor_fast_loader(model, x, y, optimizer, criterion):
    model.zero_grad()
    yhat = model(x)
    loss = criterion(yhat.squeeze(), y.squeeze())
    loss.backward()
    optimizer.step()
    return model


def train_classifier(model, dis, data_loader, pred_loss, dis_loss,
                     clf_optimizer, adv_optimizer, lambdas, second_moment_scaling,
                     dis_steps, loss_steps, num_classes):

    # Train adversary
    for i in range(dis_steps):
        for x, y, a, at in data_loader:
            yhat = model(x)
            dis.zero_grad()
            if len(yhat.size())==1:
                yhat = yhat.unsqueeze(dim=1)
            real = torch.cat((yhat,at,y),1)
            fake = torch.cat((yhat,a,y),1)
            in_dis = torch.cat((real, fake), 0)
            out_dis = dis(in_dis)
            labels = torch.cat((torch.ones(real.shape[0],1), torch.zeros(fake.shape[0],1)), 0)
            loss_adv = (dis_loss(out_dis, labels) * lambdas).mean()
            loss_adv.backward()
            adv_optimizer.step()

    # Train predictor
    for i in range(loss_steps):
        for x, y, a, at in data_loader:
            yhat = model(x)
            if len(yhat.size())==1:
                yhat = yhat.unsqueeze(dim=1)

            y_one_hot = torch.zeros(len(y), num_classes).scatter_(1, y.long(), 1.)
            fake_one_hot = torch.cat((yhat,a,y_one_hot),1)
            real_one_hot = torch.cat((yhat,at,y_one_hot),1)

            loss_second_moment = covariance_diff_biased(fake_one_hot, real_one_hot)

            fake = torch.cat((yhat,a,y),1)
            real = torch.cat((yhat,at,y),1)

            in_dis = torch.cat((real, fake), 0)
            out_dis = dis(in_dis)
            model.zero_grad()
            out_dis = dis(in_dis)
            labels = torch.cat((torch.zeros(real.shape[0],1), torch.ones(fake.shape[0],1)), 0)
            loss_adv = (dis_loss(out_dis, labels) * lambdas).mean()
            clf_loss = (1.0-lambdas)*pred_loss(yhat, y.squeeze().long())
            clf_loss += (dis_loss(dis(in_dis), labels) * lambdas).mean()
            clf_loss += lambdas*second_moment_scaling*loss_second_moment
            clf_loss.backward()
            clf_optimizer.step()

            break

    return model, dis


def inner_train_adversary_regression(model, dis, x, y, a, at, pred_loss, dis_loss,
                                     clf_optimizer, adv_optimizer, lambdas,
                                     second_moment_scaling, dis_steps, loss_steps):
    yhat = model(x)
    dis.zero_grad()
    if len(yhat.size())==1:
        yhat = yhat.unsqueeze(dim=1)
    real = torch.cat((yhat,at,y),1)
    fake = torch.cat((yhat,a,y),1)
    in_dis = torch.cat((real, fake), 0)
    out_dis = dis(in_dis)
    labels = torch.cat((torch.ones(real.shape[0],1), torch.zeros(fake.shape[0],1)), 0)
    loss_adv = (dis_loss(out_dis, labels) * lambdas).mean()
    loss_adv.backward()
    adv_optimizer.step()
    return dis

def inner_train_model_regression(model, dis, x, y, a, at, pred_loss, dis_loss,
                                 clf_optimizer, adv_optimizer, lambdas, second_moment_scaling,
                                 dis_steps, loss_steps):
    yhat = model(x)
    if len(yhat.size())==1:
        yhat = yhat.unsqueeze(dim=1)

    fake = torch.cat((yhat,a,y),1)
    real = torch.cat((yhat,at,y),1)

    loss_second_moment = covariance_diff_biased(fake, real)

    in_dis = torch.cat((real, fake), 0)
    out_dis = dis(in_dis)
    model.zero_grad()
    out_dis = dis(in_dis)
    labels = torch.cat((torch.zeros(real.shape[0],1), torch.ones(fake.shape[0],1)), 0)
    loss_adv = (dis_loss(out_dis, labels) * lambdas).mean()
    clf_loss = (1.0-lambdas)*pred_loss(yhat.squeeze(), y.squeeze())
    clf_loss += (dis_loss(dis(in_dis), labels) * lambdas).mean()
    clf_loss += lambdas*second_moment_scaling*loss_second_moment
    clf_loss.backward()
    clf_optimizer.step()
    return model

def train_regressor_fast_loader(model, dis, x, y, a, at, pred_loss, dis_loss,
                                clf_optimizer, adv_optimizer, lambdas, second_moment_scaling,
                                dis_steps, loss_steps):

    # Train adversary
    for i in range(dis_steps):
        dis = inner_train_adversary_regression(model, dis, x, y, a, at,
                                               pred_loss, dis_loss,
                                               clf_optimizer, adv_optimizer,
                                               lambdas, second_moment_scaling,
                                               dis_steps, loss_steps)

    # Train predictor
    for i in range(loss_steps):
        model = inner_train_model_regression(model, dis, x, y, a, at, pred_loss,
                                             dis_loss, clf_optimizer,
                                             adv_optimizer, lambdas,
                                             second_moment_scaling,
                                             dis_steps, loss_steps)

    return model, dis


def train_regressor(model, dis, data_loader, pred_loss, dis_loss,
                    clf_optimizer, adv_optimizer, lambdas, second_moment_scaling,
                    dis_steps, loss_steps):

    # Train adversary
    for i in range(dis_steps):
        for x, y, a, at in data_loader:
            dis = inner_train_adversary_regression(model, dis, x, y, a, at,
                                                   pred_loss, dis_loss,
                                                   clf_optimizer, adv_optimizer,
                                                   lambdas, second_moment_scaling,
                                                   dis_steps, loss_steps)

    # Train predictor
    for i in range(loss_steps):
        for x, y, a, at in data_loader:
            model = inner_train_model_regression(model, dis, x, y, a, at, pred_loss,
                                                 dis_loss, clf_optimizer,
                                                 adv_optimizer, lambdas,
                                                 second_moment_scaling,
                                                 dis_steps, loss_steps)

    return model, dis

class EquiClassLearner:

    def __init__(self,
                 lr,
                 pretrain_pred_epochs,
                 pretrain_dis_epochs,
                 epochs,
                 loss_steps,
                 dis_steps,
                 cost_pred,
                 in_shape,
                 batch_size,
                 model_type,
                 lambda_vec,
                 second_moment_scaling,
                 num_classes):

        self.lr = lr
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.num_classes = num_classes

        self.model_type = model_type
        if self.model_type == "deep_model":
            self.model = deep_model(in_shape=in_shape, out_shape=num_classes)
        elif self.model_type == "linear_model":
            self.model = linear_model(in_shape=in_shape, out_shape=num_classes)
        else:
            raise

        self.pred_loss = cost_pred
        self.clf_optimizer = optim.Adam(self.model.parameters(),lr=self.lr)

        self.pretrain_pred_epochs = pretrain_pred_epochs

        self.lambdas = torch.Tensor([lambda_vec])
        self.second_moment_scaling = torch.Tensor([second_moment_scaling])

        self.dis = class_discriminator(out_dim=1,n_y=num_classes+1+1)
        self.dis_loss = nn.BCELoss(reduce=False)
        self.adv_optimizer = optim.Adam(self.dis.parameters(),lr=self.lr)

        self.pretrain_dis_epochs = pretrain_dis_epochs

        self.epochs = epochs
        self.loss_steps = loss_steps
        self.dis_steps = dis_steps

        self.scaler = StandardScaler()
        self.scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df),
                                                        columns=df.columns,
                                                        index=df.index)


    def fit(self,X,Y):

        # The features are X[:,1:]
        X_train = pd.DataFrame(data=X[:,1:])
        y_train = pd.DataFrame(data=Y)
        orig_Z = X[:,0]
        Z_train = pd.DataFrame(data=orig_Z)
        p_success, dummy = utility_functions.density_estimation(Y, orig_Z)

        self.scaler.fit(X_train)
        X_train = X_train.pipe(self.scale_df, self.scaler)

        for epoch in range(self.pretrain_pred_epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_Z.shape)
            Z_tilde = (random_array < p_success).astype(float)
            Zt_train = pd.DataFrame(data=Z_tilde)
            train_data = PandasDataSet(X_train, y_train, Z_train, Zt_train)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

            self.model = pretrain_classifier(self.model,
                                           train_loader,
                                           self.clf_optimizer,
                                           self.pred_loss)



        for epoch in range(self.pretrain_dis_epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_Z.shape)
            Z_tilde = (random_array < p_success).astype(float)
            Zt_train = pd.DataFrame(data=Z_tilde)
            train_data = PandasDataSet(X_train, y_train, Z_train, Zt_train)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

            pretrain_adversary(self.dis,
                               self.model,
                               train_loader,
                               self.adv_optimizer,
                               self.dis_loss,
                               self.lambdas)


        for epoch in range(1, self.epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_Z.shape)
            Z_tilde = (random_array < p_success).astype(float)
            Zt_train = pd.DataFrame(data=Z_tilde)
            train_data = PandasDataSet(X_train, y_train, Z_train, Zt_train)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

            self.model, self.dis = train_classifier(self.model,
                                                    self.dis,
                                                    train_loader,
                                                    self.pred_loss,
                                                    self.dis_loss,
                                                    self.clf_optimizer,
                                                    self.adv_optimizer,
                                                    self.lambdas,
                                                    self.second_moment_scaling,
                                                    self.dis_steps,
                                                    self.loss_steps,
                                                    self.num_classes)

    def predict(self,X):
        X = X[:,1:]
        X_test = pd.DataFrame(data=X)
        X_test = X_test.pipe(self.scale_df, self.scaler)

        test_data = PandasDataSet(X_test)

        with torch.no_grad():
            Yhat = self.model(test_data.tensors[0])

        sm = nn.Softmax(dim=1)
        Yhat = sm(Yhat)
        Yhat = Yhat.detach().numpy()

        return Yhat

class EquiRegLearner:

    def __init__(self,
                 lr,
                 pretrain_pred_epochs,
                 pretrain_dis_epochs,
                 epochs,
                 loss_steps,
                 dis_steps,
                 cost_pred,
                 in_shape,
                 batch_size,
                 model_type,
                 lambda_vec,
                 second_moment_scaling,
                 out_shape,
                 use_standardscaler=True):

        self.lr = lr
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.use_standardscaler = use_standardscaler

        self.model_type = model_type
        if self.model_type == "deep_model":
            self.model = deep_reg_model(in_shape=in_shape, out_shape=out_shape)
        elif self.model_type == "linear_model":
            self.model = linear_model(in_shape=in_shape, out_shape=out_shape)
        else:
            raise

        self.pred_loss = cost_pred
        self.clf_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.pretrain_pred_epochs = pretrain_pred_epochs

        self.lambdas = torch.Tensor([lambda_vec])
        self.second_moment_scaling = torch.Tensor([second_moment_scaling])

        self.dis = reg_discriminator(out_shape+1+1)
        self.dis_loss = nn.BCELoss(reduce=False)
        self.adv_optimizer = torch.optim.SGD(self.dis.parameters(),
                                             lr=self.lr,
                                             momentum=0.9)
        self.pretrain_dis_epochs = pretrain_dis_epochs

        self.epochs = epochs
        self.loss_steps = loss_steps
        self.dis_steps = dis_steps

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_z = StandardScaler()
        self.scaler_zt = StandardScaler()

        self.scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df),
                                                        columns=df.columns,
                                                        index=df.index)


    def fit(self,X,Y):

        if self.batch_size>=X.shape[0]:
            fast_loader=True
        else:
            fast_loader=False

        # The features are X[:,1:]
        X_train = pd.DataFrame(data=X[:,1:])
        y_train = pd.DataFrame(data=Y)
        orig_Z = X[:,0]
        Z_train = pd.DataFrame(data=orig_Z)
        p_success, dummy = utility_functions.density_estimation(Y, orig_Z)

        if self.use_standardscaler:
            self.scaler_x.fit(X_train)
            X_train = X_train.pipe(self.scale_df, self.scaler_x)

            self.scaler_z.fit(Z_train)
            Z_train = Z_train.pipe(self.scale_df, self.scaler_z)

            if self.out_shape==1:
                self.scaler_y.fit(y_train)
                y_train = y_train.pipe(self.scale_df, self.scaler_y)

        x = torch.from_numpy(X_train.values).float()
        y = torch.from_numpy(y_train.values).float()
        a = torch.from_numpy(Z_train.values).float()

        for epoch in range(self.pretrain_pred_epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_Z.shape)
            Z_tilde = (random_array < p_success).astype(float)
            Zt_train = pd.DataFrame(data=Z_tilde)
            if self.use_standardscaler:
                self.scaler_zt.fit(Zt_train)
                Zt_train = Zt_train.pipe(self.scale_df, self.scaler_zt)
            train_data = PandasDataSet(X_train, y_train, Z_train, Zt_train)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
            if fast_loader:
                self.model = pretrain_regressor_fast_loader(self.model,
                                                          x,
                                                          y,
                                                          self.clf_optimizer,
                                                          self.pred_loss)
            else:
                self.model = pretrain_regressor(self.model,
                                              train_loader,
                                              self.clf_optimizer,
                                              self.pred_loss)



        for epoch in range(self.pretrain_dis_epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_Z.shape)
            Z_tilde = (random_array < p_success).astype(float)
            Zt_train = pd.DataFrame(data=Z_tilde)
            if self.use_standardscaler:
                self.scaler_zt.fit(Zt_train)
                Zt_train = Zt_train.pipe(self.scale_df, self.scaler_zt)
            train_data = PandasDataSet(X_train, y_train, Z_train, Zt_train)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
            if fast_loader:
                pretrain_adversary_fast_loader(self.dis,
                                               self.model,
                                               x,
                                               y,
                                               a,
                                               torch.from_numpy(Zt_train.values).float(),
                                               self.adv_optimizer,
                                               self.dis_loss,
                                               self.lambdas)
            else:
                pretrain_adversary(self.dis,
                                   self.model,
                                   train_loader,
                                   self.adv_optimizer,
                                   self.dis_loss,
                                   self.lambdas)


        for epoch in range(1, self.epochs):
            random_array = np.random.uniform(low=0.0, high=1.0, size=orig_Z.shape)
            Z_tilde = (random_array < p_success).astype(float)
            Zt_train = pd.DataFrame(data=Z_tilde)
            if self.use_standardscaler:
                self.scaler_zt.fit(Zt_train)
                Zt_train = Zt_train.pipe(self.scale_df, self.scaler_zt)
            train_data = PandasDataSet(X_train, y_train, Z_train, Zt_train)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
            if fast_loader:
                self.model, self.dis = train_regressor_fast_loader(self.model,
                                                                 self.dis,
                                                                 x,
                                                                 y,
                                                                 a,
                                                                 torch.from_numpy(Zt_train.values).float(),
                                                                 self.pred_loss,
                                                                 self.dis_loss,
                                                                 self.clf_optimizer,
                                                                 self.adv_optimizer,
                                                                 self.lambdas,
                                                                 self.second_moment_scaling,
                                                                 self.dis_steps,
                                                                 self.loss_steps)
            else:
                self.model, self.dis = train_regressor(self.model,
                                                     self.dis,
                                                     train_loader,
                                                     self.pred_loss,
                                                     self.dis_loss,
                                                     self.clf_optimizer,
                                                     self.adv_optimizer,
                                                     self.lambdas,
                                                     self.second_moment_scaling,
                                                     self.dis_steps,
                                                     self.loss_steps)

    def predict(self,X):
        X = X[:,1:]
        X_test = pd.DataFrame(data=X)

        if self.use_standardscaler:
            X_test = X_test.pipe(self.scale_df, self.scaler_x)

        test_data = PandasDataSet(X_test)

        with torch.no_grad():
            Yhat = self.model(test_data.tensors[0]).squeeze().detach().numpy()

        if self.out_shape==1 and self.use_standardscaler:
            out = self.scaler_y.inverse_transform(Yhat.reshape(-1, 1)).squeeze()
        elif self.out_shape==1:
            out = Yhat.squeeze()
        else:
            out = 0*Yhat
            out[:,0] = np.min(Yhat,axis=1)
            out[:,1] = np.max(Yhat,axis=1)

        return out
