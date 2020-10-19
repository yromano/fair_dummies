
# Code to reproduce the regression experiments in the paper
import numpy as np
from fair_dummies.cf_regression import run_experiment

# parameters are tuned on training set
# choose test=True to evaluate training performance

# test seed
random_state_train_test = 123
# repeat the experiments for num_experiments times
num_experiments = 20




test_methods    = []
dataset_names   = []
batch_size      = []
lr              = []
steps           = []
mu_val          = []
second_scale    = []
epochs          = []
model_type      = []
reg_type        = []


################################################################################
## Fairness-unaware baseline methods
################################################################################


test_methods +=  ['FairDummies']
dataset_names += ['meps']
batch_size +=    [10000]
lr +=            [0.01]
steps +=         [50]
mu_val +=        list(np.linspace(0,0.99,100))
second_scale +=  [20]
epochs +=        [20]
model_type +=    ["linear_model"]
reg_type +=      ["mreg"]

for exp_id in range(1):
    for mu_val_id in range(100):
        cur_test_method = test_methods[exp_id]
        cur_dataset_name = dataset_names[exp_id]
        cur_batch_size = batch_size[exp_id]
        cur_lr_loss = lr[exp_id]
        cur_lr_dis = lr[exp_id]
        cur_loss_steps = steps[exp_id]
        cur_dis_steps = steps[exp_id]
        cur_mu_val = mu_val[mu_val_id]
        cur_epochs = epochs[exp_id]
        cur_random_state = random_state_train_test
        cur_model_type = model_type[exp_id]
        cur_regression_type = reg_type[exp_id]
        cur_second_scale = second_scale[exp_id]
    
        # run an experiment and save average results to CSV file
        run_experiment(cur_test_method,
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
                       random_state_train_test,
                       cur_second_scale,
                       num_experiments)
