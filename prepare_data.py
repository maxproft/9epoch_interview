from prelim import get_data
import pandas as pd
from random import shuffle


def get_today_date(s: str):
    ts = pd.Timestamp(s)
    return f'{ts.year}-{ts.month}-{ts.day}'


no_tomorrow = set()


def get_avg_sentiment(df: pd.DataFrame):
    df_subset = df[['today', 'sentiment_score']]
    grouped_avg = df_subset.groupby('today').mean()
    avg_sentiment = {}

    for time, [score] in grouped_avg.iterrows():
        avg_sentiment[time] = score

    def get_next_day(today: str):
        ts = pd.Timestamp(today)
        tomorrow = f'{ts.year}-{ts.month}-{ts.day + 1}'

        try:
            return avg_sentiment[tomorrow]
        except KeyError:
            no_tomorrow.add(today)

    return get_next_day


def split_test_train(df: pd.DataFrame, proportion: float):
    all_days = [new_df for day, new_df in df.groupby('today')]
    shuffle(all_days)
    test_cutoff = int(proportion * len(all_days))
    return all_days[:test_cutoff], all_days[test_cutoff:]


def get_test_train(test_proportion=0.2):
    raw_data = get_data()
    raw_data['today'] = raw_data['publish_datetime'].map(get_today_date)
    avg_sent_funct = get_avg_sentiment(raw_data)
    raw_data['tomorrow_sentiment'] = raw_data['today'].map(avg_sent_funct)
    print(f'There are {len(no_tomorrow)} days without a score for the next day')

    filtered_data = raw_data.dropna(subset=['tomorrow_sentiment'])
    test, train = split_test_train(filtered_data, test_proportion)
    return test, train


if __name__ == '__main__':
    get_test_train()
