#%%
# 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('all_logits.csv')
accuracies = {}
precisions = {}
specificities = {}
models = []
for col in data.columns: 
    if '_1' in col:
        models.append(col.split('_1')[0])

#%%
def softmax(x_0, x_1): 
    x = np.vstack([x_0, x_1]).T
    e_x = np.exp(x-x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

metric = 'entropy'

for model in models: 
    probs = softmax(data[f"{model}_0"], data[f"{model}_1"])
    preds = np.argmax(probs, axis=1)
    targets = data["truth"].values
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    
    accuracy = (preds == targets).mean()
    
    precision = (preds[targets==1] == 1).mean()
    precisions[model] = precision

    specificity = (preds[targets==0] == 0).mean()
    specificities[model] = specificity

    accuracies[model] = accuracy
    print(f"Model: {model}, Accuracy: {accuracy:.4f}")

    data[f'{model}_accuracy'] = (preds == targets).astype(int)
    data[f'{model}_entropy'] = entropy
    data[f'{model}_maxSoftmax'] = probs.max(axis=1)

    if accuracy < 0.6:
        continue
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=True)

    ax = axes
    sns.kdeplot(
        data=data[data[f'{model}_accuracy'] == 1][f'{model}_{metric}'], 
        label='Correct',
        fill=True,
        color='green',
        alpha=0.3,
        ax=ax
    )

    sns.kdeplot(
        data=data[data[f'{model}_accuracy'] == 0][f'{model}_{metric}'], 
        label='Incorrect',
        fill=True,
        color='red',
        alpha=0.3,
        ax=ax
    )

    plt.title(f"Model: {model}, Accuracy: {accuracy:.4f}", fontsize=16)
    plt.xlabel(metric, fontsize=14)
    plt.legend(fontsize=14)


#%%

######################
######## EVALUATE BASED ON MODEL AGREEMENT/DISAGREEMENT
######################
models_dnn = ['densenet', 'efficientnet', 'inception', 'resnet']#, 'dnn_fusion']
models_dnn_acc = [f'{model}_accuracy' for model in models_dnn]
accuracies = data[['truth']+models_dnn_acc]

#check agreement between models
accuracies['agreement']  = (accuracies[models_dnn_acc].nunique(axis=1) == 1)

#check correctness of prediction when all models agree
accuracies['prediction'] = accuracies['densenet_accuracy'] & accuracies[ 'efficientnet_accuracy'] & accuracies['inception_accuracy'] & accuracies['resnet_accuracy'] #& accuracies['dnn_fusion_accuracy']

#check accuracy when all models agree
accuracy = accuracies[accuracies['agreement']==True]['prediction'].mean()
print("Accuracy when all models agree: ", accuracy)

#deferral rate
num_deferred_samples = accuracies[accuracies['agreement']==False].shape[0]
total_samples = accuracies.shape[0]
deferral_rate = num_deferred_samples / total_samples
print("Deferral rate when models disagree: ", deferral_rate)

#accuracy based on mean accuracy of individual models
accuracies['mean_individual_accuracy'] = accuracies[models_dnn_acc].mean(axis=1)
accuracy2 = (accuracies['truth'] == (accuracies['mean_individual_accuracy']>0.5).astype(int)).mean()
print("Accuracy based on mean individual model accuracy: ", accuracy2)

#%%

######################
######## EVALUATE BASED ON MODEL CONFIDENCE - BAYESIAN CALIBRATION

def bayesian_calibration(probs, model_precision, class_freq_prior=0.00028): 

    #prevalence of melanoma in canada 0.00028: 10,800 cases per year / 38 million people
    
    p_x_given_y = model_precision
    p_y = probs

    p_x_given_neg_y = 1 - model_precision
    p_neg_y = 1 - probs

    p_y_given_x = (p_x_given_y * p_y) / (p_x_given_y * p_y + p_x_given_neg_y * p_neg_y)

    return p_y_given_x




