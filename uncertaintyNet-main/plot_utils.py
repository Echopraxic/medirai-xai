#%%
import os 
import glob
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

#%%

def compute_aupr(df, uncert_metric='Test Entropy', metric='Error'):
	"""Compute AUPR given labels and uncertainty scores."""
	y_true = df[metric].astype(int).values.ravel()
	uncertainty_scores = df[uncert_metric].values.ravel()
	precision, recall, _ = precision_recall_curve(y_true, uncertainty_scores)

	return auc(recall, precision)

#%%

metric = 'Test Entropy'
wd = "/home/zeinab/scripts/SymanticSparseNetwork-medir/output/cifar10/"

##CIFAR RUNS
runs = {
	"Deterministic-WRN": "run_20260120-131930"
}

# runs = {
# 	"Deterministic (pretrain)": "with_smoothing/run_20251118-074605",
# 	"Deterministic-Rand": "with_smoothing/run_20251123-173546",
# 	"Deterministic-Rand2": "with_smoothing/run_20251125-173822",
# 	"V-LL-Frozen (pretrain)": "with_smoothing/run_20251118-074812",
# 	"V-LL-Tune": "with_smoothing/run_20251124-215340",
# 	"V-LL-Tune-backward": "with_smoothing/backward/run_20251125-195723",
# 	"VVL-frozen (pretrain)": "with_smoothing/run_20251118-164747",
# 	"VVL-Tune (pretrain)": "with_smoothing/run_20251123-154947",
# 	"VVL-Rand (pretrain)": "with_smoothing/run_20251123-225808",
# 	"VVL-Tune": "with_smoothing/run_20251124-215252",
# 	"Deterministic-Wide": "wide_resnet50/run_20251209-215031",
# 	"Deterministic-Rand3": "backward/run_20251209-225320",
# 	"FirstVL-Tune-Wide": "wide_resnet50/run_20251210-090613",
# }

# runs = {
	# "Deterministic (pretrain)": "run_20251209-203508",
	# "Deterministic-Rand": "run_20251123-173546",
	# "Deterministic-Rand2": "run_20251125-173822",
	# "V-LL-Frozen (pretrain)": "run_20251118-074812",
	# "V-LL-Tune": "run_20251124-215340",
	# "V-LL-Tune-backward": "backward/run_20251125-195723",
	# "VVL-frozen (pretrain)": "run_20251118-164747",
	# "VVL-Tune (pretrain)": "run_20251123-154947",
	# "VVL-Rand (pretrain)": "run_20251123-225808",
	# "VVL-Tune": "run_20251124-215252"
# }

metric = "Test Entropy"
all_data = []
plt.figure()
bar_width = 1/(len(runs.keys())+1)
x=np.array([0, 1])
colors = sns.color_palette("colorblind", len(runs.keys()))
for i, (run_name, run) in enumerate(runs.items()):
	assert os.path.exists(f"{wd}/{run}/test.csv"), f"Path {wd}/{run}/test.csv does not exist."
	data = pd.read_csv(f"{wd}/{run}/test.csv")
	data['Error'] = 1-data["Test Accuracy"]

	aupr = compute_aupr(data, uncert_metric=metric, metric='Error')
	print(run_name, aupr)

	results = pd.DataFrame(data.groupby('Test Accuracy')[metric].mean()).reset_index()
	errors = pd.DataFrame(data.groupby('Test Accuracy')[metric].std()).reset_index()

	correct_data = data[data["Test Accuracy"] == 1]
	incorrect_data = data[data["Test Accuracy"] == 0]

	sns.kdeplot(
		correct_data[metric], 
		label=f"{run_name} (Correct)",
		linestyle='-',
		linewidth=2,
		fill=True,
		color='green',
		alpha=0.3,
	)

	# 3. Plot Incorrect (Dashed Line for comparison/error distribution)
	sns.kdeplot(
		incorrect_data[metric], 
		label=f"{run_name} (Incorrect)",
		linestyle='--',
		linewidth=2,
		fill=True,
		color='red',
		alpha=0.3,
	)

	data['Model'] = run_name
	data['Outcome'] = data['Test Accuracy'].map({0: 'Incorrect', 1: 'Correct'})
	all_data.append(data)

	
	# plt.bar(x-bar_width*i, results[metric], yerr=errors[metric], width=bar_width, label=run_name)
# combined_df = pd.concat(all_data)
# plt.figure(figsize=(10, 6))
# sns.violinplot(data=combined_df, x='Outcome', y=metric, hue='Model', cut=0)
# sns.boxplot(data=combined_df, x='Outcome', y=metric, hue='Model', showfliers=False)
# plt.xticks(x-bar_width/2, labels=['Incorrect', 'Correct'])
# # plt.ylim(0,0.8)
plt.legend()
plt.ylabel(metric)

# %%
metric = "Test Entropy"
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 5), sharex=True)
axes = axes.flatten()
if len(runs) == 1:
	axes = [axes]

for i, (run_name, run) in enumerate(runs.items()):
	# Load data
	data = pd.read_csv(f"{wd}/{run}/test.csv")
	if metric not in data.columns:
		continue 
	# Select the specific axis for this model
	ax = axes[i]

	# Plot Correct (Green, Filled)
	sns.kdeplot(
		data=data[data["Test Accuracy"] == 1][metric], 
		label='Correct',
		fill=True,
		color='green',
		alpha=0.3,
		ax=ax
	)

	# Plot Incorrect (Red, Filled)
	sns.kdeplot(
		data=data[data["Test Accuracy"] == 0][metric], 
		label='Incorrect',
		fill=True,
		color='red',
		alpha=0.3,
		ax=ax
	)

	accuracy = round(data["Test Accuracy"].mean(), 3)
	ax.set_title(f"{run_name} \n acc={accuracy}", fontsize=8)
	ax.set_ylabel("Density")
	ax.legend(loc="upper right")

# General layout adjustments
plt.xlabel(metric)
plt.tight_layout()
plt.show()