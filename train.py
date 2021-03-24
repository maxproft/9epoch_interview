from prepare_data import get_test_train
from model import XGBootstrapModel
from article import Article, DailyArticles
import matplotlib.pyplot as plt
import numpy as np
from random import choice
from sklearn.metrics import mean_squared_error


def train():
    test_num_bootstrap = 100
    train_bootstrap = 100

    print('Loading Data')
    test_raw, train_raw = get_test_train()
    print(f'#test: {len(test_raw)}, #train: {len(train_raw)}')
    print('Changing data format')

    test_art_features = [[Article.from_pandas(row).to_article_features()
                          for idx, row in day.iterrows()] for day in test_raw]

    test_true = np.array([day['tomorrow_sentiment'].iloc[-1] for day in test_raw])
    train_art_features = [[Article.from_pandas(row).to_article_features()
                           for idx, row in day.iterrows()] for day in train_raw]
    train_true = np.array([day['tomorrow_sentiment'].iloc[-1] for day in train_raw])

    print('Training Model')

    DailyArticles.num_to_concat = 5
    model = XGBootstrapModel(max_depth=3, n_estimators=60, dropout=True, skip_drop=0.5, rate_drop=0.1,
                             objective='reg:squarederror', learning_rate=0.1)

    model.fit(train_art_features, train_true, train_bootstrap)
    model.save()

    print('Predicting')

    test_pred = model.predict_many_days(test_art_features, test_num_bootstrap)
    train_pred = model.predict_many_days(train_art_features, test_num_bootstrap)

    print('Statistics')

    statistics(test_true, test_pred, 'Test')
    statistics(train_true, train_pred, 'Train')

    print("Consistancy")
    std_vals = []
    for _ in range(100):
        random_test = choice(test_art_features)
        consistancy_prediction_values = [model.predict_one_day(random_test, test_num_bootstrap) for _ in range(100)]
        std = np.std(consistancy_prediction_values)
        std_vals.append(std)

    print(f'random day prediction standard deviation (average): {np.mean(std_vals)}')
    print(f'random day prediction standard deviation (std of std): {np.std(std_vals)}')

    print('Finished')


def statistics(y_true, y_pred, name):
    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    plt.plot([-1, 1], [-1, 1])
    plt.scatter(y_true, y_pred)
    ax.set_title(name)
    ax.set_xlabel('True')
    ax.set_xlabel('Prediction')
    fig.savefig(f'{name}.png')

    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    plt.hist(y_true - y_pred)
    ax.set_title(f'{name} histogram of True-Prediction')
    fig.savefig(f'{name}_hist.png')

    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    plt.hist(y_true, bins=40)
    ax.set_title(f'{name} histogram of True values')
    fig.savefig(f'{name}_hist_true.png')


    abs_diff = sorted(np.abs(y_true - y_pred))
    proportion = 0.9
    cutoff = abs_diff[int(proportion * len(abs_diff))]
    print(f'{name} - The average sentiment will be within {round(cutoff, ndigits=3)} '
          f'for {round(proportion * 100)}% of all days')

    print(f'{name} mean squared error = {mean_squared_error(y_true, y_pred)}')




if __name__ == '__main__':
    for _ in range(1):
        train()
