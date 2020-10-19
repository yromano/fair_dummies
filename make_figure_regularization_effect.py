import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# display results

SMALL_SIZE = 24
MEDIUM_SIZE = SMALL_SIZE
BIGGER_SIZE = SMALL_SIZE

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('text', usetex=True)

mu_array = list(np.linspace(0,0.99,100)[::5])
# Linear
filename = "/Users/romano/CJ_get_tmp/6f9771d7f10b5f9f67e815d4a0ae469533c8751a/results/results.csv"

# Deep
filename = "/Users/romano/CJ_get_tmp/4e0c952eeda7c4cc861592c0b43e7848cc9cb3d2/results/results.csv"

results = pd.read_csv(filename)
results = results.loc[results['mu_val'].isin(mu_array)]
results['mu_val'] = results['mu_val'].round(decimals=2)
results['rmse'] = results['mse'].pow(1./2)
results.groupby(['mu_val', 'p_val']).mean()

plt.figure(figsize=(10,5))
ax = sns.boxplot(y="p_val", x="mu_val", data=results,color='white')
ax.set(xlabel='$\lambda$', ylabel='Fairness p-value')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center")
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')
plt.savefig("p_value_lambda.png", bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(10,5))
ax = sns.boxplot(y="rmse", x="mu_val", data=results,color='white')
ax.set(xlabel='$\lambda$', ylabel='RMSE')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center")
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')
plt.savefig("rmse_lambda.png", bbox_inches='tight', dpi=300)
plt.show()