from prepare_data import get_test_train
from model import XGBootstrapModel
from article import Article, DailyArticles
import numpy as np
from sklearn.metrics import mean_squared_error

from random import shuffle

# Training 909/1728


def generate_options():
    all_options = []
    for lr in [0.05, 0.1, 0.3]:
        for num_to_concatenate in [3, 5]:
            for max_depth in [1, 3, 5]:
                for objective in ['reg:squarederror', 'reg:pseudohubererror']:
                    for train_bootstrap in [50, 100, 300, 1000]:
                        for n_estimator in [30, 60, 100, 200]:
                            for drop in [{'dropout': False},
                                         {'dropout': True,
                                          'skip_drop': 0.5,
                                          'rate_drop': 0.1},
                                         {'dropout': True,
                                          'skip_drop': 0.5,
                                          'rate_drop': 0.1}]:
                                d = {'learning_rate': lr,
                                     'num_to_concatenate': num_to_concatenate,
                                     'max_depth': max_depth,
                                     'objective': objective,
                                     'train_bootstrap': train_bootstrap,
                                     'n_estimator': n_estimator,
                                     }
                                d.update(drop)
                                all_options.append(d)
    shuffle(all_options)
    return all_options


class Data:
    def __init__(self, test_art_features, test_true, train_art_features, train_true):
        self.test_true = test_true
        self.test_art_features = test_art_features
        self.train_art_features = train_art_features
        self.train_true = train_true


def train_many():
    file = open("results.txt", "a")

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

    d = Data(test_art_features, test_true, train_art_features, train_true)

    print('starting_training')
    options = generate_options()
    num_options = len(options)
    for i, parameters in enumerate(options):
        print(f'Training {i}/{num_options}')
        train_error, test_error = train_one(d, parameters)
        file.write(f'\n{test_error}, {train_error}, {parameters}')
    file.close()


def train_one(d: Data, parameters: dict):
    DailyArticles.num_to_concat = parameters['num_to_concatenate']
    test_num_bootstrap = 100

    model = XGBootstrapModel(max_depth=parameters['max_depth'], n_estimators=parameters['n_estimator'],
                             learning_rate=parameters['learning_rate'], dropout=parameters['dropout'],
                             objective=parameters['objective'], skip_drop=parameters.get('skip_drop'),
                             rate_drop=parameters.get('rate_drop'),
                             )

    model.fit(d.train_art_features, d.train_true, parameters['train_bootstrap'])

    test_pred = model.predict_many_days(d.test_art_features, test_num_bootstrap)
    train_pred = model.predict_many_days(d.train_art_features, test_num_bootstrap)

    train_error = mean_squared_error(d.train_true, train_pred)
    test_error = mean_squared_error(d.test_true, test_pred)

    return train_error, test_error


if __name__ == '__main__':
    train_many()
