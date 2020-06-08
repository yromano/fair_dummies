# Code to reproduce the classification experiments in the paper

from fair_dummies.cf_classification import run_experiment

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



test_methods =   []
dataset_names =  []
batch_size =     []
lr =             []
steps =          []
mu_val =         []
second_scale =   []
epochs =         []
model_type =     []

################################################################################
## Fairness-unaware baseline methods
################################################################################

test_methods +=  ['Baseline']
dataset_names += ['nursery']
batch_size +=    [32]
lr +=            [0.1]
steps +=         [1]
mu_val +=        [0.0]
second_scale +=  [0.0]
epochs +=        [20]
model_type +=    ["linear_model"]

test_methods +=  ['Baseline']
dataset_names += ['nursery']
batch_size +=    [32]
lr +=            [0.001]
steps +=         [1]
mu_val +=        [0.0]
second_scale +=  [0.0]
epochs +=        [100]
model_type +=    ["deep_model"]


################################################################################
# HGR EO Penalty
# paper: Fairness-Aware Learning for Continuous Attributes and Treatments,
# J. Mary, C. Calauzènes, N. El Karoui, ICML 2019
################################################################################


test_methods +=  ['HGR']
dataset_names += ['nursery']
batch_size +=    [128]
lr +=            [0.001]
steps +=         [1]
mu_val +=        [0.98]
second_scale +=  [0.0]
epochs +=        [50]
model_type +=    ["linear_model"]


test_methods +=  ['HGR']
dataset_names += ['nursery']
batch_size +=    [128]
lr +=            [0.001]
steps +=         [1]
mu_val +=        [0.98]
second_scale +=  [0.0]
epochs +=        [50]
model_type +=    ["deep_model"]


###############################################################################
# Adversarial Debiasing EO Penalty
# paper: Mitigating Unwanted Biases with Adversarial Learning,
# Zhang, B.H., Lemoine, B. and Mitchell, M., AAAI/ACM Conference on AI, Ethics,
# and Society, 2018
###############################################################################


test_methods +=  ['AdversarialDebiasing']
dataset_names += ['nursery']
batch_size +=    [32]
lr +=            [0.5]
steps +=         [2]
mu_val +=        [0.999999]
second_scale +=  [0.0]
epochs +=        [40]
model_type +=    ["deep_model"]


test_methods +=  ['AdversarialDebiasing']
dataset_names += ['nursery']
batch_size +=    [32]
lr +=            [0.5]
steps +=         [1]
mu_val +=        [0.999999]
second_scale +=  [0.0]
epochs +=        [200]
model_type +=    ["linear_model"]

################################################################################
# Fair Dummies EO Penalty (proposed method)
# Paper: Achieving Equalized Odds by Resampling Sensitive Attributes,
# Y. Romano, S. Bates, and E. J. Candès, 2020
################################################################################


test_methods +=  ['FairDummies']
dataset_names += ['nursery']
batch_size +=    [32]
lr +=            [0.5]
steps +=         [1] #60
mu_val +=        [0.99999]
second_scale +=  [0.01]
epochs +=        [50]
model_type +=    ["linear_model"]


test_methods +=  ['FairDummies']
dataset_names += ['nursery']
batch_size +=    [32]
lr +=            [0.5]
steps +=         [2]
mu_val +=        [0.9]
second_scale +=  [0.00001]
epochs +=        [50]
model_type +=    ["deep_model"]


for exp_id in range(8):
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
    cur_second_scale = second_scale[exp_id]

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
                   random_state_train_test,
                   cur_second_scale,
                   num_experiments)
