import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
result_resne18_5k = pd.read_csv('correct_predictions_imagenet_vgg19_50000.csv')
fgsm_resnet = pd.read_csv('results_test/Adversarials_fgsm_imagenet_vgg19_50000.csv')
bim_resnet = pd.read_csv('results_test/Adversarials_bim_imagenet_vgg19_50000.csv')
deepfool_resnet = pd.read_csv('results_test/Adversarials_deepfool_imagenet_vgg19_50000.csv')
result_resne18_5k['fgsm_label'] = fgsm_resnet.iloc[:,2]
result_resne18_5k['BIM_label'] = bim_resnet.iloc[:,2]
d = {'model': ['Resnet18'], 'nb_imgs': [len(result_resne18_5k)], 'percentage_missclass_fgsm': [len(result_resne18_5k[result_resne18_5k['real_label'] != result_resne18_5k['fgsm_label']])/len(result_resne18_5k)], 'percentage_missclass_new_strategy': [len(result_resne18_5k[result_resne18_5k['real_label'] != result_resne18_5k['new_strategy_label']])/len(result_resne18_5k)], 'alpha': [0.001], 'neighbors':[70], 'dec':[0.006], 'percentage_missclass_deepfool': [len(result_resne18_5k[result_resne18_5k['real_label'] != result_resne18_5k['deep_fool_label']])/len(result_resne18_5k)], 'percentage_missclass_BIM': [len(result_resne18_5k[result_resne18_5k['real_label'] != result_resne18_5k['BIM_label']])/len(result_resne18_5k)]}
df =pd.DataFrame(data = d)
df.to_csv("save.csv")

fig, ax = plt.subplots(figsize=(6, 5))
p = sns.histplot(data= bim_resnet[result_resne18_5k['real_label'] != result_resne18_5k['deep_fool_label']], x=bim_resnet[result_resne18_5k['real_label'] != result_resne18_5k['deep_fool_label']].iloc[:,1], stat='probability', ax=ax, binwidth=0.05, color='lightblue')
plt.xlabel("Confidence of adversarial")
plt.title("CIFAR10 - deepfool")
plt.savefig("confidence.png")
