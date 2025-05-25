import pandas as pd
x = pd.read_csv('transferability_from_cifar10_secondCNN_nb_strat4.csv')
transfer_strat = list(x.columns[8:])
print(transfer_strat)
dict_transfers = {}
for i in transfer_strat:
    dict_transfers[i] = len(x[x[i] != x['real_label']])/len(x)

print(dict_transfers)
