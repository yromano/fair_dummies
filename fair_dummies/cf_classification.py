# Run a classification experiment
# implements model fitting with equalized odds and demonstrates
# how to use equlized coverage for unbiased uncertainty estimation

# We rely on the nonconformist package and CQR package, available at
# https://github.com/donlnz/nonconformist
# https://github.com/yromano/cqr

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/others/third_party/fairness_aware_learning')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/others/third_party/cqr')))

base_path = os.getcwd() + '/data/'

import torch
import random
import get_dataset
import numpy as np
import pandas as pd
from others import adv_debiasing
from others import continuous_fairness
from fair_dummies import utility_functions
from fair_dummies import fair_dummies_learning

from nonconformist.nc import ClassifierNc
from nonconformist.cp import IcpClassifier
from nonconformist.base import ClassifierAdapter
from nonconformist.nc import InverseProbabilityErrFunc


pd.set_option('precision', 4)

# Get length
def get_length(Y,predicted_C):
    length = sum(predicted_C)
    return length

# Get coverage
def get_coverage(Y,predicted_C):
    coverage = int( predicted_C[int(Y)] == True )
    return coverage

# Get coverage and length
def get_stat(Y,predicted_C):
    coverage = int( predicted_C[int(Y)] == True )
    length = sum(predicted_C)
    return coverage, length

def class_compute_coverage_len(y_test, y_set):

    results = [get_stat(y_test[test],y_set[test]) for test in range(len(y_test))]
    results = list(zip(*results))
    coverage = pd.DataFrame([row for row in results[0]])
    length = pd.DataFrame([r for r in results[1]])

    return coverage.mean().values[0], length.mean().values[0]

def class_compute_coverage_per_sample(y_test,y_set,significance,x_test=None,condition=None):

    if condition is not None:

        category_map = np.array([condition((x_test[i, :], y_test[i])) for i in range(y_test.size)])
        categories = np.unique(category_map)

        coverage = np.empty(len(categories), dtype=np.object)
        length = np.empty(len(categories), dtype=np.object)

        cnt = 0

        for cond in categories:
            tmp = np.arange(len(y_test))
            idx = tmp[category_map == cond]

            coverage[cnt] = [get_coverage(y_test[idx[test]],y_set[idx[test],:]) for test in range(len(idx))]

            coverage_avg = np.sum( coverage[cnt] ) / len(y_test[idx]) * 100
            print("Group %d : Percentage in the range (expecting %.2f): %f" % (cond, 100 - significance*100, coverage_avg))
            sys.stdout.flush()

            length[cnt] = [get_length(y_test[idx[test]],y_set[idx[test]]) for test in range(len(idx))]
            print("Group %d : Average length: %f" % (cond, np.mean(length[cnt])))
            sys.stdout.flush()
            cnt = cnt + 1
    else:
        raise


    return coverage, length

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

    # step size used to fit bianry classifier (discriminator)
    lr_dis = cur_lr_dis

    # inner epochs to fit loss
    loss_steps = cur_loss_steps

    # inner epochs to fit binary classifier (discriminator)
    dis_steps = cur_dis_steps

    # total number of epochs
    epochs = cur_epochs

    # utility loss
    cost_pred = torch.nn.CrossEntropyLoss()

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
    pred_error = np.zeros(num_experiments)

    for i in range(num_experiments):

        # Split into train and test
        X, A, Y, X_cal, A_cal, Y_cal, X_test, A_test, Y_test = get_dataset.get_train_test_data(base_path, dataset, seed+i)
        in_shape = X.shape[1]
        num_classes = len(np.unique(Y))


        print("n train = " + str(X.shape[0]) + " p = " + str(X.shape[1]))
        print("n calibration = " + str(X_cal.shape[0]))
        print("n test = " + str(X_test.shape[0]))

        sys.stdout.flush()

        if method == "AdversarialDebiasing":

            class ClassAdapter(ClassifierAdapter):
                def __init__(self, model=None,fit_params=None, params=None):
                    super(ClassAdapter, self).__init__(model,fit_params)
                    # Instantiate model
                    self.learner = adv_debiasing.AdvDebiasingClassLearner(lr_loss,
                                                                          loss_steps,
                                                                          dis_steps,
                                                                          epochs,
                                                                          cost_pred,
                                                                          in_shape,
                                                                          batch_size,
                                                                          model_type,
                                                                          num_classes,
                                                                          cur_mu_val)

                def fit(self, x, y):
                    self.learner.fit(x, y)

                def predict(self, x):
                    return self.learner.predict(x)

        elif method == "FairDummies":

            class ClassAdapter(ClassifierAdapter):
                def __init__(self, model=None,fit_params=None, params=None):
                    super(ClassAdapter, self).__init__(model,fit_params)
                    # Instantiate model
                    self.learner = fair_dummies_learning.EquiClassLearner(lr=lr_loss,
                                                                       pretrain_pred_epochs=loss_steps,
                                                                       pretrain_dis_epochs=dis_steps,
                                                                       epochs=epochs,
                                                                       loss_steps=1,
                                                                       dis_steps=1,
                                                                       cost_pred=cost_pred,
                                                                       in_shape=in_shape,
                                                                       batch_size=batch_size,
                                                                       model_type=model_type,
                                                                       lambda_vec=cur_mu_val,
                                                                       second_moment_scaling=cur_second_scale,
                                                                       num_classes=num_classes)

                def fit(self, x, y):
                    self.learner.fit(x, y)

                def predict(self, x):
                    return self.learner.predict(x)

        elif method == "HGR":

            class ClassAdapter(ClassifierAdapter):
                def __init__(self, model=None,fit_params=None, params=None):
                    super(ClassAdapter, self).__init__(model,fit_params)
                    # Instantiate model
                    self.learner = continuous_fairness.HGR_Class_Learner(lr = lr_loss,
                                                                         epochs = epochs,
                                                                         mu=cur_mu_val,
                                                                         cost_pred=cost_pred,
                                                                         in_shape=in_shape,
                                                                         out_shape=num_classes,
                                                                         batch_size=batch_size,
                                                                         model_type=model_type)

                def fit(self, x, y):
                    self.learner.fit(x, y)

                def predict(self, x):
                    return self.learner.predict(x)

        elif method == "Baseline":

            class ClassAdapter(ClassifierAdapter):
                def __init__(self, model=None,fit_params=None, params=None):
                    super(ClassAdapter, self).__init__(model,fit_params)
                    # Instantiate model
                    self.learner = fair_dummies_learning.EquiClassLearner(lr=lr_loss,
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
                                                                       num_classes=num_classes)

                def fit(self, x, y):
                    self.learner.fit(x, y)

                def predict(self, x):
                    return self.learner.predict(x)

        fairness_class = ClassAdapter(model=None)


        nc = ClassifierNc(fairness_class, InverseProbabilityErrFunc())

        # function that extracts the group identifier
        def condition(x, y=None):
            return int(x[0][0]>0)

        icp = IcpClassifier(nc,condition=condition)

        input_data_train = np.concatenate((A[:,np.newaxis],X),1)
        icp.fit(input_data_train, Y)

        input_data_cal = np.concatenate((A_cal[:,np.newaxis],X_cal),1)
        icp.calibrate(input_data_cal, Y_cal)

        input_data_test = np.concatenate((A_test[:,np.newaxis],X_test),1)
        Yhat_test = icp.predict(input_data_test, significance=0.1)

        avg_coverage, avg_length = class_compute_coverage_len(Y_test, Yhat_test)
        coverage_sample, length_sample = class_compute_coverage_per_sample(Y_test,
                                                                           Yhat_test,
                                                                           0.1,
                                                                           input_data_test,
                                                                           condition)

        avg_length_0[i] = np.mean(length_sample[0])
        avg_coverage_0[i] = np.mean(coverage_sample[0])
        avg_length_1[i] = np.mean(length_sample[1])
        avg_coverage_1[i] = np.mean(coverage_sample[1])


        Yhat_out_cal = fairness_class.learner.predict(input_data_cal)
        Yhat_out_test = fairness_class.learner.predict(input_data_test)

        p_val = utility_functions.fair_dummies_test_classification(Yhat_out_cal,
                                                                   A_cal,
                                                                   Y_cal,
                                                                   Yhat_out_test,
                                                                   A_test,
                                                                   Y_test,
                                                                   num_reps=1,
                                                                   num_p_val_rep=1000,
                                                                   reg_func_name="Net")
        avg_p_val[i] = p_val

        pred_error[i] = 1.0-utility_functions.compute_acc_numpy(Yhat_out_test, Y_test)

        print("experiment = " + str(i+1))
        print("Coverage 0 = " + str(avg_coverage_0[i]))
        print("Coverage 1 = " + str(avg_coverage_1[i]))
        print("Length 0 = " + str(avg_length_0[i]))
        print("Length 1 = " + str(avg_length_1[i]))
        print("Prediction Error = " + str(pred_error[i]))


        print("p_val = " + str(p_val))

        sys.stdout.flush()


        outdir = './results/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        out_name = outdir + 'results.csv'

        full_name = cur_test_method + "_" + cur_model_type
        df = pd.DataFrame({'method'    : [cur_test_method],
                           'dataset'   : [cur_dataset_name],
                           'batch_size': [cur_batch_size],
                           'lr_loss'   : [cur_lr_loss],
                           'lr_dis'    : [cur_lr_dis],
                           'loss_steps': [cur_loss_steps],
                           'dis_steps' : [cur_dis_steps],
                           'mu_val'    : [cur_mu_val],
                           'epochs'    : [cur_epochs],
                           'second_scale' : [cur_second_scale],
                           'random_state' : [seed+i],
                           'model_type'   : [cur_model_type],
                           'metric'       : [metric],
                           'avg_length'        : [avg_length],
                           'avg_coverage'        : [avg_coverage],
                           'avg_length_0'      : [avg_length_0[i]],
                           'avg_length_1'      : [avg_length_1[i]],
                           'avg_coverage_0'    : [avg_coverage_0[i]],
                           'avg_coverage_1'    : [avg_coverage_1[i]],
                           'pred_error'          : [pred_error[i]],
                           'p_val'           : [p_val],
                           'full_name'       : [full_name]
                           })

        if os.path.isfile(out_name):
            df2 = pd.read_csv(out_name)
            df = pd.concat([df2, df], ignore_index=True)

        df.to_csv(out_name, index=False)

        print(full_name)
        print("Num experiments %02d | Avg. Pred Err = %.4f | Avg Length 0 = %.4f | Avg Length 1 = %.4f | Avg Coverage 0 = %.4f | Avg Coverage 1 = %.4f | Avg p_val = %.4f | min p_val = %.4f" %
              (i+1, np.mean(pred_error[:i+1]), np.mean(avg_length_0[:i+1]), np.mean(avg_length_1[:i+1]),
                    np.mean(avg_coverage_0[:i+1]), np.mean(avg_coverage_1[:i+1]),
                    np.mean(avg_p_val[:i+1]), np.min(avg_p_val[:i+1])))
        print("======== Done =========")
        sys.stdout.flush()
