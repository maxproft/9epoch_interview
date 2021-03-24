import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

data_dir = os.path.abspath('prelim_images')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def get_data():
    columns = ('publish_datetime',
               'author',
               'article_title',
               'article_summary',
               'article_content',
               'sentiment_score',
               'sector',
               'category',
               'sub_category',
               'tags',
               'date_time_utc',
               'date_time_aest')
    data = pd.read_csv('afr_articles.csv', )
    data.columns = columns
    print(f'number of records: {len(data)}')
    return data


def publish_datetime(data: pd.Series):
    data = [pd.Timestamp(d) for d in data]
    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.hist(data, bins=20)
    ax.set_title('publish_datetime histogram')
    fig.savefig(os.path.join(data_dir, 'publish_datetime.png'))

    print(f'Earliest publish_datetime = {min(data)}, latest = {max(data)}, difference = {max(data) - min(data)}')


def author(data: pd.Series):
    print(f'Number Authors (kinda): {len(set(data))}')

    top_frequency(data, 'author')


def article_title(data: pd.Series):
    print(f"Duplicate article_title: {len(data) - len(set(data))}")

    num_words = [len(i.split()) for i in data if type(i) != float]
    num_words_not_zero = [n for n in num_words if n > 1]
    print(f'title num words mean, std: {np.mean(num_words_not_zero)}, {np.std(num_words_not_zero)}')


def article_summary(data: pd.Series):
    print(f"Article summaries which are not blank: {len([1 for d in data if type(d) != float])} / {len(data)}")
    num_words = [len(i.split()) for i in data if type(i) != float]
    num_words_not_zero = [n for n in num_words if n > 1]
    print(f'summary num words mean, std: {np.mean(num_words_not_zero)}, {np.std(num_words_not_zero)}')

    


def article_content(data: pd.Series):
    lengths = [len(i) for i in data if type(i) != float]

    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.hist(lengths, bins=40)
    ax.set_title('article_content histogram')
    fig.savefig(os.path.join(data_dir, 'article_content.png'))

    # Zoomed plot
    subset = [i for i in lengths if i < 12500]
    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.hist(subset, bins=40)
    ax.set_title('article_content zoomed histogram')
    fig.savefig(os.path.join(data_dir, 'article_content_zoomed.png'))

    print(f'article_content num chars mean: {np.mean(lengths)}, stddev {np.std(lengths)}')

    num_words = [len(i.split()) for i in data if type(i) != float]
    num_words_not_zero = [n for n in num_words if n > 1]
    print(f'article_content num words mean, std: {np.mean(num_words_not_zero)}, {np.std(num_words_not_zero)}')


def sentiment_score(data: pd.Series):
    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.hist(data, bins=40)
    ax.set_title('sentiment histogram')
    fig.savefig(os.path.join(data_dir, 'sentiment_score.png'))

    print(f'sentiment_score length mean: {np.mean(data)}, stddev {np.std(data)}')


def sector(data: pd.Series):
    print(
        f'sectors: {set(data)}. Companies: {len([True for i in data if i == "Companies"])}, '
        f'Markets {len([True for i in data if i == "Markets"])}')


def category(data: pd.Series):
    print(f'There are {len(set(data))} categories')

    top_frequency(data, 'category')


def sub_category(data: pd.Series):
    print(f'There are {len(set(data))} sub_categories')
    top_frequency(data, 'sub_category')


def tags(data: pd.Series):
    all_tags = set([d.strip() for t in data if type(t) == str for d in t.split(',')])
    print(f'There are {len(all_tags)} unique tags')

    top_frequency(data, 'tags')


def date_time_utc(data: pd.Series):
    data = [pd.Timestamp(d) for d in data]
    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.hist(data, bins=20)
    ax.set_title('date_time_utc histogram')
    fig.savefig(os.path.join(data_dir, 'date_time_utc.png'))

    print(f'Earliest date_time_utc = {min(data)}, latest = {max(data)}, difference = {max(data) - min(data)}')


def date_time_aest(data: pd.Series):
    data = [pd.Timestamp(d) for d in data]
    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.hist(data, bins=20)
    ax.set_title('date_time_aest histogram')

    fig.savefig(os.path.join(data_dir, 'date_time_aest.png'))

    print(f'Earliest date_time_aest = {min(data)}, latest = {max(data)}, difference = {max(data) - min(data)}')


def main():
    data = get_data()
    publish_datetime(data['publish_datetime'])
    author(data['author'])
    article_title(data['article_title'])
    article_summary(data['article_summary'])
    article_content(data['article_content'])
    sentiment_score(data['sentiment_score'])
    sector(data['sector'])
    category(data['category'])
    sub_category(data['sub_category'])
    tags(data['tags'])
    date_time_utc(data['date_time_utc'])
    date_time_aest(data['date_time_aest'])


def top_frequency(data: pd.Series, name: str, threshold=500, delimiter=','):
    split_data = [s.strip() for d in data if type(d) != float for s in d.split(delimiter)]
    options = {s: 0 for s in set(split_data)}
    for d in split_data:
        options[d] += 1
    item_count = [[k, v] for k, v in options.items()]
    names = [i[0] for i in item_count]
    values = [i[1] for i in item_count]
    sorted_values, sorted_names = zip(*sorted(zip(values, names), reverse=True))
    top = [n for v, n in zip(sorted_values, sorted_names) if v > threshold]

    print(f'top {name} with more than {threshold} articles: {top}')


if __name__ == '__main__':
    main()
