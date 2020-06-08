# Run a regression experiment
# implements model fitting with equalized odds and demonstrates
# how to use equlized coverage for unbiased uncertainty estimation

# We use the CQR package, provided in https://github.com/yromano/cqr
# We also rely on the nonconformist package, available at
# https://github.com/donlnz/nonconformist

import os
import sys

print(os.getcwd())

sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/others/third_party/fairness_aware_learning')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/others/third_party/cqr')))

base_path = os.getcwd() + '/data/'

import torch
import random
import get_dataset
import numpy as np
import pandas as pd
from fair_dummies import utility_functions
from fair_dummies import fair_dummies_learning

from others import adv_debiasing
from others import continuous_fairness


from cqr.torch_models import AllQuantileLoss
from cqr.helper import compute_coverage_per_sample, compute_coverage_len

from nonconformist.nc import RegressorNc
from nonconformist.cp import IcpRegressor
from nonconformist.nc import AbsErrorErrFunc
from nonconformist.nc import QuantileRegErrFunc
from nonconformist.base import RegressorAdapter


pd.set_option('precision', 4)


def run_experiment(cur_test_method,
                   cur_dataset_name,
                   cur_batch_size,
                   cur_lr_loss,
                   cur_lr_dis,
                   cur_loss_steps,
                   cur_dis_steps,
                   cur_mu_val,
                   cur_epochs,
                   cur_model_type,
                   cur_regression_type,
                   cur_random_state,
                   cur_second_scale,
                   num_experiments):

    method = cur_test_method

    seed = cur_random_state
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = cur_dataset_name

    batch_size = cur_batch_size

    # step size to minimize loss
    lr_loss = cur_lr_loss

    # step size used to fit GAN's classifier
    lr_dis = cur_lr_dis

    # inner epochs to fit loss
    loss_steps = cur_loss_steps

    # inner epochs to fit GAN's classifier
    dis_steps = cur_dis_steps

    # total number of epochs
    epochs = cur_epochs

    # utility loss
    if cur_regression_type == "mreg":
        cost_pred = torch.nn.MSELoss()
        out_shape = 1
    else:
        raise

    model_type = cur_model_type

    metric = "equalized_odds"

    print(dataset)
    print(method)
    sys.stdout.flush()


    avg_length_0 = np.zeros(num_experiments)
    avg_length_1 = np.zeros(num_experiments)

    avg_coverage_0 = np.zeros(num_experiments)
    avg_coverage_1 = np.zeros(num_experiments)

    avg_p_val = np.zeros(num_experiments)
    mse = np.zeros(num_experiments)


    for i in range(num_experiments):

        # Split into train and test
        X, A, Y, X_cal, A_cal, Y_cal, X_test, A_test, Y_test = get_dataset.get_train_test_data(base_path, dataset, seed+i)
        in_shape = X.shape[1]

        print("n train = " + str(X.shape[0]) + " p = " + str(X.shape[1]))
        print("n calibration = " + str(X_cal.shape[0]))
        print("n test = " + str(X_test.shape[0]))

        sys.stdout.flush()

        if method == "AdversarialDebiasing":

            class RegAdapter(RegressorAdapter):
                def __init__(self, model=None,fit_params=None, params=None):
                    super(RegAdapter, self).__init__(model,fit_params)
                    # Instantiate model
                    self.learner = adv_debiasing.AdvDebiasingRegLearner(lr=lr_loss,
                                                                        N_CLF_EPOCHS=loss_steps,
                                                                        N_ADV_EPOCHS=dis_steps,
                                                                        N_EPOCH_COMBINED=epochs,
                                                                        cost_pred=cost_pred,
                                                                        in_shape=in_shape,
                                                                        batch_size=batch_size,
                                                                        model_type=model_type,
                                                                        out_shape=out_shape,
                                                                        lambda_vec=cur_mu_val)

                def fit(self, x, y):
                    self.learner.fit(x, y)

                def predict(self, x):
                    return self.learner.predict(x)

        elif method=='FairDummies':

            class RegAdapter(RegressorAdapter):
                def __init__(self, model=None,fit_params=None, params=None):
                    super(RegAdapter, self).__init__(model,fit_params)
                    # Instantiate model
                    self.learner = fair_dummies_learning.EquiRegLearner(lr=lr_loss,
                                                                     pretrain_pred_epochs=0,
                                                                     pretrain_dis_epochs=0,
                                                                     epochs=epochs,
                                                                     loss_steps=loss_steps,
                                                                     dis_steps=dis_steps,
                                                                     cost_pred=cost_pred,
                                                                     in_shape=in_shape,
                                                                     batch_size=batch_size,
                                                                     model_type=model_type,
                                                                     lambda_vec=cur_mu_val,
                                                                     second_moment_scaling=cur_second_scale,
                                                                     out_shape=out_shape)

                def fit(self, x, y):
                    self.learner.fit(x, y)

                def predict(self, x):
                    return self.learner.predict(x)


        elif method == 'HGR':

            class RegAdapter(RegressorAdapter):
                def __init__(self, model=None,fit_params=None, params=None):
                    super(RegAdapter, self).__init__(model,fit_params)
                    # Instantiate model

                    self.learner = continuous_fairness.HGR_Reg_Learner(lr=lr_loss,
                                                                       epochs=epochs,
                                                                       mu=cur_mu_val,
                                                                       cost_pred=cost_pred,
                                                                       in_shape=in_shape,
                                                                       out_shape=out_shape,
                                                                       batch_size=batch_size,
                                                                       model_type=model_type)

                def fit(self, x, y):
                    self.learner.fit(x, y)

                def predict(self, x):
                    return self.learner.predict(x)

        elif method=='Baseline':

            class RegAdapter(RegressorAdapter):
                def __init__(self, model=None,fit_params=None, params=None):
                    super(RegAdapter, self).__init__(model,fit_params)
                    # Instantiate model
                    self.learner = fair_dummies_learning.EquiRegLearner(lr=lr_loss,
                                                                     pretrain_pred_epochs=epochs,
                                                                     pretrain_dis_epochs=0,
                                                                     epochs=0,
                                                                     loss_steps=0,
                                                                     dis_steps=0,
                                                                     cost_pred=cost_pred,
                                                                     in_shape=in_shape,
                                                                     batch_size=batch_size,
                                                                     model_type=model_type,
                                                                     lambda_vec=0,
                                                                     second_moment_scaling=0,
                                                                     out_shape=out_shape)

                def fit(self, x, y):
                    self.learner.fit(x, y)

                def predict(self, x):
                    return self.learner.predict(x)

        fairness_reg = RegAdapter(model=None)

        if cur_regression_type == "mreg":
            nc = RegressorNc(fairness_reg, AbsErrorErrFunc())
        else:
            raise

        # function that extracts the group identifier
        def condition(x, y=None):
            return int(x[0][0]>0)


        icp = IcpRegressor(nc,condition=condition)

        input_data_train = np.concatenate((A[:,np.newaxis],X),1)
        icp.fit(input_data_train, Y)

        input_data_cal = np.concatenate((A_cal[:,np.newaxis],X_cal),1)
        icp.calibrate(input_data_cal, Y_cal)

        input_data_test = np.concatenate((A_test[:,np.newaxis],X_test),1)
        Yhat_test = icp.predict(input_data_test, significance=0.1)

        # compute and print average coverage and average length
        coverage_sample, length_sample = compute_coverage_per_sample(Y_test,
                                                                     Yhat_test[:,0],
                                                                     Yhat_test[:,1],
                                                                     0.1,
                                                                     method,
                                                                     input_data_test,
                                                                     condition)

        avg_coverage, avg_length = compute_coverage_len(Y_test, Yhat_test[:,0], Yhat_test[:,1])
        avg_length_0[i] = np.mean(length_sample[0])
        avg_coverage_0[i] = np.mean(coverage_sample[0])
        avg_length_1[i] = np.mean(length_sample[1])
        avg_coverage_1[i] = np.mean(coverage_sample[1])

        Yhat_out_cal = fairness_reg.learner.predict(input_data_cal)
        Yhat_out_test = fairness_reg.learner.predict(input_data_test)

        if out_shape==1:
            mse[i] = np.mean((Yhat_out_test-Y_test)**2)
            MSE_trivial = np.mean((np.mean(Y_test)-Y_test)**2)
            print("MSE = " + str(mse[i]) + "MSE Trivial = " + str(MSE_trivial))

        p_val = utility_functions.fair_dummies_test_regression(Yhat_out_cal,
                                                               A_cal,
                                                               Y_cal,
                                                               Yhat_out_test,
                                                               A_test,
                                                               Y_test,
                                                               num_reps = 1,
                                                               num_p_val_rep=1000,
                                                               reg_func_name="Net")

        avg_p_val[i] = p_val

        print("experiment = " + str(i+1))

#        if out_shape==2:
#            init_coverage, init_length = compute_coverage_len(Y_test, Yhat_out_test[:,0], Yhat_out_test[:,1])
#            print("Init Coverage = " + str(init_coverage))
#            print("Init Length = " + str(init_length))

        print("Coverage 0 = " + str(avg_coverage_0[i]))
        print("Coverage 1 = " + str(avg_coverage_1[i]))

        print("Length 0 = " + str(avg_length_0[i]))
        print("Length 1 = " + str(avg_length_1[i]))
        print("MSE = " + str(mse[i]))


        print("p_val = " + str(p_val))
        sys.stdout.flush()


        outdir = './results/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        out_name = outdir + 'results.csv'

        full_name = cur_test_method + "_" + cur_model_type + "_" + cur_regression_type
        df = pd.DataFrame({'method'    : [cur_test_method],
                           'dataset'   : [cur_dataset_name],
                           'batch_size': [cur_batch_size],
                           'lr_loss'   : [cur_lr_loss],
                           'lr_dis'    : [cur_lr_dis],
                           'loss_steps': [cur_loss_steps],
                           'dis_steps' : [cur_dis_steps],
                           'mu_val'    : [cur_mu_val],
                           'epochs'    : [cur_epochs],
                           'random_state' : [seed+i],
                           'model_type'   : [cur_model_type],
                           'metric'       : [metric],
                           'cur_second_scale' : [cur_second_scale],
                           'regression_type'  : [cur_regression_type],
                           'avg_length'        : [avg_length],
                           'avg_coverage'        : [avg_coverage],
                           'avg_length_0'      : [avg_length_0[i]],
                           'avg_length_1'      : [avg_length_1[i]],
                           'mse'               : [mse[i]],
                           'avg_coverage_0'    : [avg_coverage_0[i]],
                           'avg_coverage_1'    : [avg_coverage_1[i]],
                           'p_val'           : [p_val],
                           'full_name'       : [full_name]
                           })

        if os.path.isfile(out_name):
            df2 = pd.read_csv(out_name)
            df = pd.concat([df2, df], ignore_index=True)

        df.to_csv(out_name, index=False)

        print(full_name)
        print("Num experiments %02d | Avg MSE = %.4f | Avg Length 0 = %.4f | Avg Length 1 = %.4f | Avg Coverage 0 = %.4f | Avg Coverage 1 = %.4f | Avg p_val = %.4f | min p_val = %.4f" %
              (i+1, np.mean(mse[:i+1]), np.mean(avg_length_0[:i+1]), np.mean(avg_length_1[:i+1]),
                    np.mean(avg_coverage_0[:i+1]), np.mean(avg_coverage_1[:i+1]),
                    np.mean(avg_p_val[:i+1]), np.min(avg_p_val[:i+1])))
        print("======== Done =========")
        sys.stdout.flush()
