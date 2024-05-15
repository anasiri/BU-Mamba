import os
import json
import numpy as np
from natsort import os_sorted


def get_max_last_index(result):
    result = np.array(result)
    index = result.shape[0]-np.argmax(result[::-1])-1
    return index

results_dir = 'checkpoints/results/'
experiments = [i for i in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir,i))]
for experiment_name in experiments[:]:
    folds = os_sorted(os.listdir(os.path.join(results_dir, experiment_name)))
    experiment_results = []
    for fold in folds:
        path = os.path.join(results_dir, experiment_name, fold, 'log.txt')
        if not os.path.isfile(path):
            continue
        with open(path, 'r') as f:
            logs = f.readlines()
        logs = [json.loads(log[:-1]) for log in logs]

        fold_results = {}

        for key in logs[0].keys():
            fold_results[key] = []
            for log in logs:
                fold_results[key].append(log[key])
        experiment_results.append(fold_results)
    
    if len(experiment_results) == 0:
        continue

    test_accs = [fold_result['test_acc1'][get_max_last_index(fold_result['val_acc1'])] for fold_result in experiment_results]
    test_aucs = [fold_result['test_auc'][get_max_last_index(fold_result['val_acc1'])] for fold_result in experiment_results]
    print(f"Experiment Name: {experiment_name}")
    print("Accs: ", [round(i,2) for i in test_accs])
    print("AUCs: ", [round(i,2) for i in test_aucs], "\n")
    #     print(f"Average Test Accuracy over {len(experiment_results)} Folds: {round(np.mean(test_accs), 2)} +- {np.std(test_accs)}")
    print(f"Average Test Acc over {len(experiment_results)} Folds:"\
          f" {round(np.mean(test_accs), 2)} +- {round(np.std(test_accs), 4)}"\
          f"\t Max Fold Acc: {round(np.max(test_accs), 2)}, Min Fold Acc: {round(np.min(test_accs), 2)}")
    print(f"Average Test AUC over {len(experiment_results)} Folds:"\
          f" {round(np.mean(test_aucs), 2)} +- {round(np.std(test_aucs), 4)}"\
          f"\t Max Fold AUC: {round(np.max(test_aucs), 2)}, Min Fold AUC: {round(np.min(test_aucs), 2)}")
    print("\n\n")