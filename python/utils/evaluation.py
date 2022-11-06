from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
import datetime
from pathlib import Path
import numpy as np


def evaluate_option_price(predicted, actual, expected_payoff_list, experiment_details):

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)

    print(f'MSE:{mse}')
    print(f'MAE: {mae}')

    experiment_details['MSE'] = mse
    experiment_details['MAE'] = mae
    experiment_details['Expected Payoff Variance'] = np.var(expected_payoff_list)
    variance_reduction = experiment_details['variance_reduction']
    Nsim = experiment_details['Nsim']

    folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path(f'../results/{folder}').mkdir(parents=True, exist_ok=True)
    with open(f'../results/{folder}/details.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_details, f, ensure_ascii=False, indent=4)

    plt.figure(figsize=(30, 10))
    plt.plot(predicted, label='Predicted', color='red')
    plt.plot(actual, label='Actual', color='blue')
    plt.legend(loc='upper left')
    plt.savefig(f'../results/{folder}/results_{variance_reduction}_{Nsim}.png')
