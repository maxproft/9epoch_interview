import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
import json
from math import isnan
from random import choice, shuffle


class _CommonValues:
    authors = ('Angela Macdonald-Smith', 'Simon Evans', 'Sue Mitchell', 'Max Mason', 'James Frost', 'James Eyers',
               'Brad Thompson', 'Timothy Moore', 'Peter Ker', 'Jonathan Shapiro', 'James Fernyhough', 'William McInnes',
               'Jenny Wiggins')
    categories = ('Financial Services', 'Equity Markets', 'Energy', 'Retail', 'Mining', 'Media And Marketing',
                  'Professional Services', 'Transport', 'Healthcare And Fitness', 'Infrastructure')
    subcategories = ('Coronavirus pandemic', 'Sharemarket', 'Before the Bell')
    sectors = ('Markets', 'Companies')
    tags = ('Coronavirus pandemic', 'Shares', 'Opinion', 'Investing', 'Sharemarket', 'World markets', 'Mining', 'Bonds',
            'Retail', 'Commodities', 'Big four', 'Currencies', 'Gas', 'Westpac Banking Corporation', 'Earnings season',
            'Mergers & acquisitions', 'National Australia Bank', 'China', 'Commonwealth Bank', 'BHP Billiton',
            'Analysis', 'Before the Bell', 'Legal industry', 'USA', 'Oil', 'Hayne fallout', 'Rio Tinto', 'Electricity')

    title_mean = 8.2
    title_std = 1.9
    article_mean = 701.2
    article_std = 354.5
    summary_mean = 24.0
    summary_std = 6.3

    @staticmethod
    def col_name(names: Tuple[str, ...], prefix):
        return [prefix + n for n in names] + [prefix + 'None']


class Article:
    def __init__(self,
                 publish_datetime: str,
                 author: str,
                 article_title: str,
                 article_summary: str,
                 article_content: str,
                 sentiment_score: float,
                 sector: str,
                 category: str,
                 sub_category: str,
                 tags: str,
                 ):
        self.publish_datetime = publish_datetime
        self.author = author
        self.article_title = article_title
        self.article_summary = article_summary
        self.article_content = article_content
        self.sentiment_score = sentiment_score
        self.sector = sector
        self.category = category
        self.sub_category = sub_category
        self.tags = tags

    def to_article_features(self):
        return ArticleFeatures.from_article(self)

    @staticmethod
    def from_pandas(row: pd.DataFrame):
        return Article(publish_datetime=row['publish_datetime'],
                       author=row['author'],
                       article_title=row['article_title'],
                       article_summary=row['article_summary'],
                       article_content=row['article_content'],
                       sentiment_score=row['sentiment_score'],
                       sector=row['sector'],
                       category=row['category'],
                       sub_category=row['sub_category'],
                       tags=row['tags'],
                       )


class ArticleFeatures:
    def __init__(self,
                 publish_vec,  # datetime to a more useful feature
                 author_vec, category_vec, sub_vec, tag_vec, sector_vec,  # one hot vec of common values
                 title_count_normalized, article_count_normalized, sentiment,  # single value
                 summary_vec,  # either [1, mean_word_count] OR [0, article_word_count]

                 ):
        self.publish_vec = publish_vec

        self.author_vec = author_vec
        self.category_vec = category_vec
        self.sub_vec = sub_vec
        self.tag_vec = tag_vec
        self.sector_vec = sector_vec

        self.title_count_normalized = title_count_normalized
        self.article_count_normalized = article_count_normalized
        self.summary_vec = summary_vec

        self.sentiment: float = sentiment

    @classmethod
    def from_json(cls, json_str: str):
        """
        Json Structure:
        {
        "publish_datetime": "2020-01-05 12:32:15",
        
        "author": "john smith, another name, yo momma",
        "category": "Category",
        "sub_category": "sub category, second subcategory",
        "sector": "Market",
        "tags": "blah,blah blah,blah blah blah"
        
        "title_word_count": 10,
        "article_word_count": 10000,
        "summary_word_count": 500,
        
        "sentiment_score": 0.2,
        }
        """

        data = json.loads(json_str)
        return ArticleFeatures(
            publish_vec=cls.time2vec(data['publish_datetime']),

            author_vec=cls.str2one_hot(data['author'], _CommonValues.authors),
            category_vec=cls.str2one_hot(data['category'], _CommonValues.categories),
            sub_vec=cls.str2one_hot(data['sub_category'], _CommonValues.subcategories),
            sector_vec=cls.str2one_hot(data['sector'], _CommonValues.sectors),
            tag_vec=cls.str2one_hot(data['tags'], _CommonValues.tags),

            title_count_normalized=cls.normalize(None, _CommonValues.title_mean,
                                                 _CommonValues.title_std, False, word_count=data['title_word_count']),
            article_count_normalized=cls.normalize(None, _CommonValues.article_mean,
                                                   _CommonValues.article_std, False,
                                                   word_count=data['article_word_count']),
            summary_vec=cls.normalize(None, _CommonValues.summary_mean,
                                      _CommonValues.summary_std, True, word_count=data['summary_word_count']),
            sentiment=data['sentiment_score'],
        )

        pass

    @classmethod
    def from_article(cls, article: Article):
        return ArticleFeatures(
            publish_vec=cls.time2vec(article.publish_datetime),

            author_vec=cls.str2one_hot(article.author, _CommonValues.authors),
            category_vec=cls.str2one_hot(article.category, _CommonValues.categories),
            sub_vec=cls.str2one_hot(article.sub_category, _CommonValues.subcategories),
            sector_vec=cls.str2one_hot(article.sector, _CommonValues.sectors),
            tag_vec=cls.str2one_hot(article.tags, _CommonValues.tags),
            title_count_normalized=cls.normalize(article.article_title, _CommonValues.title_mean,
                                                 _CommonValues.title_std, False),
            article_count_normalized=cls.normalize(article.article_title, _CommonValues.article_mean,
                                                   _CommonValues.article_std, False),
            summary_vec=cls.normalize(article.article_title, _CommonValues.summary_mean,
                                      _CommonValues.summary_std, True),
            sentiment=article.sentiment_score,
        )

    def input_vec(self):
        concat_list = self.publish_vec + \
                      self.author_vec + self.category_vec + self.sub_vec + self.tag_vec + self.sector_vec + \
                      [self.title_count_normalized, self.article_count_normalized, self.sentiment] + self.summary_vec
        return np.array(concat_list, dtype=np.float32)

    @staticmethod
    def column_names():
        return ['publish_time_of_day', 'publish_day_of_week', 'publish_time_of_year'] + \
               _CommonValues.col_name(_CommonValues.authors, 'Author: ') + \
               _CommonValues.col_name(_CommonValues.categories, 'Category: ') + \
               _CommonValues.col_name(_CommonValues.subcategories, 'Sub_Cat: ') + \
               _CommonValues.col_name(_CommonValues.tags, 'Tag: ') + \
               _CommonValues.col_name(_CommonValues.sectors, 'Sector: ') + \
               ['Title Word Count', 'Article_word_count', 'sentiment_score'] + \
               ['summary_is_empty', 'summary_count_or_avg']

    def input_pd_arr(self):
        return pd.DataFrame(data=self.input_vec(), columns=self.column_names())

    def prediction_target_vec(self):
        return np.array([self.sentiment], dtype=np.float32)

    @staticmethod
    def time2vec(time_str: str):
        ts = pd.Timestamp(time_str)
        time_of_day = ts.hour / 24 + ts.minute / 1440
        day_of_week = ts.day_of_week / 7
        time_of_year = ts.day_of_year / 365

        return [time_of_day, day_of_week, time_of_year]

    @staticmethod
    def str2one_hot(tags: Union[str, float], options: Tuple[str, ...]):
        if type(tags) == float and isnan(tags):
            tags = ''
        indexes = set()

        for d in tags.split(','):
            tag = d.strip()
            try:
                indexes.add(options.index(tag))
            except ValueError:
                pass

        one_hot = [0] * (len(options) + 1)
        for idx in indexes:
            one_hot[idx] = 1
        if indexes == set():
            one_hot[-1] = 1
        return one_hot

    @staticmethod
    def normalize(text: Optional[str], mean: float, std: float, return_vec: bool, word_count=None):
        if word_count is None:
            num_words = len(text.split())
        else:
            num_words = word_count
        is_empty = num_words <= 1

        if return_vec:
            if is_empty:
                return [1, 0]
            else:
                return [0, (num_words - mean) / std]

        else:
            return (num_words - mean) / std


class DailyArticles:
    _columns = tuple(ArticleFeatures.column_names())
    num_to_concat = 5

    def __init__(self, article_vectors: List[np.ndarray], next_day_avg_sentiment: Optional[float]):
        self.article_vectors = article_vectors
        self.next_day_avg_sentiment = next_day_avg_sentiment

    def bootstrap(self, number_times_to_bootstrap: int):
        return np.array([np.concatenate([choice(self.article_vectors) for _ in range(self.num_to_concat)]) for _ in
                         range(number_times_to_bootstrap)], dtype=np.float32)

    def bootstrap_to_pandas(self, number_times_to_bootstrap: int):
        arr = self.bootstrap(number_times_to_bootstrap)
        return pd.DataFrame(data=arr, columns=self.cols())

    def true_vec(self, number_times_to_bootstrap: int):
        return np.array([self.next_day_avg_sentiment] * number_times_to_bootstrap, dtype=np.float32)

    @classmethod
    def cols(cls):
        return [f'{c} #{i}' for i in range(cls.num_to_concat) for c in cls._columns]

    @classmethod
    def bootstrap_train_pd(cls, data: List[List[ArticleFeatures]], true_output: np.ndarray, num_bootstrap):
        daily_articles = [DailyArticles([dd.input_vec() for dd in d], y) for d, y in zip(data, true_output)]

        input_x = [vec for da in daily_articles for vec in da.bootstrap(num_bootstrap)]
        y_true = [vec for da in daily_articles for vec in da.true_vec(num_bootstrap)]
        z = list(zip(input_x, y_true))
        shuffle(z)
        input_x, y_true = zip(*z)

        array = np.array(input_x, dtype=np.float32)

        df = pd.DataFrame(array, columns=cls.cols())

        return df, np.array(y_true, dtype=np.float32)
