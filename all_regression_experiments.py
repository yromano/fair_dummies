
# Code to reproduce the regression experiments in the paper

from fair_dummies.cf_regression import run_experiment

# parameters are tuned on training set
# choose test=True to evaluate training performance

test = True

if test:
    # test seed
    random_state_train_test = 123456789
    # repeat the experiments for num_experiments times
    num_experiments = 20
else:
    # train seed
    random_state_train_test = 0
    # repeat the experiments for num_experiments times
    num_experiments = 10



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

# MEPS

test_methods +=  ['Baseline']
dataset_names += ['meps']
batch_size +=    [10000]
lr +=            [0.01]
steps +=         [1]
mu_val +=        [0.0]
second_scale +=  [0.0]
epochs +=        [100]
model_type +=    ["linear_model"]
reg_type +=      ["mreg"]

test_methods +=  ['Baseline']
dataset_names += ['meps']
batch_size +=    [10000]
lr +=            [0.01]
steps +=         [1]
mu_val +=        [0.0]
second_scale +=  [0.0]
epochs +=        [200]
model_type +=    ["deep_model"]
reg_type +=      ["mreg"]

# CRIMES

test_methods +=  ['Baseline']
dataset_names += ['crimes']
batch_size +=    [10000]
lr +=            [0.01]
steps +=         [1]
mu_val +=        [0.0]
second_scale +=  [0.0]
epochs +=        [600]
model_type +=    ["linear_model"]
reg_type +=      ["mreg"]

test_methods +=  ['Baseline']
dataset_names += ['crimes']
batch_size +=    [10000]
lr +=            [0.01]
steps +=         [1]
mu_val +=        [0.0]
second_scale +=  [0.0]
epochs +=        [4000]
model_type +=    ["deep_model"]
reg_type +=      ["mreg"]

###############################################################################
# Adversarial Debiasing EO Penalty
# paper: Mitigating Unwanted Biases with Adversarial Learning,
# Zhang, B.H., Lemoine, B. and Mitchell, M., AAAI/ACM Conference on AI, Ethics,
# and Society, 2018
###############################################################################

# MEPS


test_methods +=  ['AdversarialDebiasing']
dataset_names += ['meps']
batch_size +=    [128]
lr +=            [0.01]
steps +=         [20]
mu_val +=        [0.95]
second_scale +=  [0.0]
epochs +=        [300]
model_type +=    ["linear_model"]
reg_type +=      ["mreg"]

test_methods +=  ['AdversarialDebiasing']
dataset_names += ['meps']
batch_size +=    [64]
lr +=            [0.001]
steps +=         [2]
mu_val +=        [0.95]
second_scale +=  [0.0]
epochs +=        [400]
model_type +=    ["deep_model"]
reg_type +=      ["mreg"]

# CRIMES

test_methods +=  ['AdversarialDebiasing']
dataset_names += ['crimes']
batch_size +=    [64]
lr +=            [0.001]
steps +=         [40]
mu_val +=        [0.3]
second_scale +=  [0.0]
epochs +=        [100]
model_type +=    ["linear_model"]
reg_type +=      ["mreg"]

test_methods +=  ['AdversarialDebiasing']
dataset_names += ['crimes']
batch_size +=    [64]
lr +=            [0.001]
steps +=         [2]
mu_val +=        [0.3]
second_scale +=  [0.0]
epochs +=        [100]
model_type +=    ["deep_model"]
reg_type +=      ["mreg"]

################################################################################
# HGR EO Penalty
# paper: Fairness-Aware Learning for Continuous Attributes and Treatments,
# J. Mary, C. Calauzènes, N. El Karoui, ICML 2019
################################################################################

# MEPS

test_methods +=  ['HGR']
dataset_names += ['meps']
batch_size +=    [256]
lr +=            [0.001]
steps +=         [1]
mu_val +=        [0.8]
second_scale +=  [0.0]
epochs +=        [15]
model_type +=    ["linear_model"]
reg_type +=      ["mreg"]

test_methods +=  ['HGR']
dataset_names += ['meps']
batch_size +=    [256]
lr +=            [0.001]
steps +=         [1]
mu_val +=        [0.9]
second_scale +=  [0.0]
epochs +=        [50]
model_type +=    ["deep_model"]
reg_type +=      ["mreg"]


# CRIMES

test_methods +=  ['HGR']
dataset_names += ['crimes']
batch_size +=    [256]
lr +=            [0.001]
steps +=         [1]
mu_val +=        [0.7]
second_scale +=  [0.0]
epochs +=        [50]
model_type +=    ["linear_model"]
reg_type +=      ["mreg"]

test_methods +=  ['HGR']
dataset_names += ['crimes']
batch_size +=    [256]
lr +=            [0.001]
steps +=         [1]
mu_val +=        [0.9]
second_scale +=  [0.0]
epochs +=        [50]
model_type +=    ["deep_model"]
reg_type +=      ["mreg"]


################################################################################
# Fair Dummies EO Penalty (proposed method)
# Paper: Achieving Equalized Odds by Resampling Sensitive Attributes,
# Y. Romano, S. Bates, and E. J. Candès, 2020
################################################################################

# MEPS


test_methods +=  ['FairDummies']
dataset_names += ['meps']
batch_size +=    [10000]
lr +=            [0.01]
steps +=         [50]
mu_val +=        [0.6]
second_scale +=  [20]
epochs +=        [20]
model_type +=    ["linear_model"]
reg_type +=      ["mreg"]

test_methods +=  ['FairDummies']
dataset_names += ['meps']
batch_size +=    [10000]
lr +=            [0.01]
steps +=         [80]
mu_val +=        [0.5]
second_scale +=  [10.0]
epochs +=        [20]
model_type +=    ["deep_model"]
reg_type +=      ["mreg"]

# CRIMES

test_methods +=  ['FairDummies']
dataset_names += ['crimes']
batch_size +=    [10000]
lr +=            [0.01]
steps +=         [80]
mu_val +=        [0.7]
second_scale +=  [1.0]
epochs +=        [20]
model_type +=    ["linear_model"]
reg_type +=      ["mreg"]

test_methods +=  ['FairDummies']
dataset_names += ['crimes']
batch_size +=    [10000]
lr +=            [0.01]
steps +=         [40]
mu_val +=        [0.8]
second_scale +=  [1.0]
epochs +=        [100]
model_type +=    ["deep_model"]
reg_type +=      ["mreg"]


for exp_id in range(16):
    cur_test_method = test_methods[exp_id]
    cur_dataset_name = dataset_names[exp_id]
    cur_batch_size = batch_size[exp_id]
    cur_lr_loss = lr[exp_id]
    cur_lr_dis = lr[exp_id]
    cur_loss_steps = steps[exp_id]
    cur_dis_steps = steps[exp_id]
    cur_mu_val = mu_val[exp_id]
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
