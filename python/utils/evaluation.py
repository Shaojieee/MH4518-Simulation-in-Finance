from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
import datetime
from pathlib import Path


def evaluate_option_price(predicted, actual, experiment_details):

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)

    print(f'MSE:{mse}')
    print(f'MAE: {mae}')

    experiment_details['MSE'] = mse
    experiment_details['MAE'] = mae

    folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path(f'../results/{folder}').mkdir(parents=True, exist_ok=True)
    with open(f'../results/{folder}/details.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_details, f, ensure_ascii=False, indent=4)

    plt.figure(figsize=(30, 10))
    plt.plot(predicted, label='Predicted', color='red')
    plt.plot(actual, label='Actual', color='blue')
    plt.legend(loc='upper left')
    plt.savefig(f'../results/{folder}/results.png')