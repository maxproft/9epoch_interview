from flask import Flask, request, Response
from article import ArticleFeatures
from json import dumps
from model import XGBootstrapModel


app = Flask(__name__)

model = XGBootstrapModel.load()


@app.route('/get_prediction', methods=['post'])
def get_prediction():
    articles_json = request.get_json()
    articles_features = [ArticleFeatures.from_json(dumps(a)) for a in articles_json]
    result = model.predict_series(articles_features, 100)
    return Response(f'{{"success":true, "prediction":{result}}}', status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(port=5000)

"""
Example which works:

curl -H "content-type: application/json" -X POST -d '[{"publish_datetime": "2018-12-01 05:19:00+00", "author": "Zandi Shabalala", "sentiment_score": -0.512, "sector": "Companies", "category": "Mining", "sub_category": "Copper", "tags": "Copper", "title_word_count": 13, "summary_word_count": 0, "article_word_count": 392}, {"publish_datetime": "2018-12-01 05:28:00+00", "author": "Jessica Resnick-Ault, Scott DiSavino", "sentiment_score": 0.49, "sector": "Companies", "category": "Energy", "sub_category": "Oil", "tags": "Oil", "title_word_count": 14, "summary_word_count": 0, "article_word_count": 275}, {"publish_datetime": "2018-12-01 08:36:00+00", "author": "Vildana Hajric, Brendan Walsh", "sentiment_score": 0.171, "sector": "Markets", "category": "Equity Markets", "sub_category": "", "tags": "", "title_word_count": 8, "summary_word_count": 0, "article_word_count": 374}, {"publish_datetime": "2018-12-02 13:18:00+00", "author": "Timothy Moore", "sentiment_score": -0.234, "sector": "Companies", "category": "Mining", "sub_category": "Iron ore", "tags": "Iron ore, BHP Billiton, Fortescue Metals Group, Rio Tinto", "title_word_count": 11, "summary_word_count": 0, "article_word_count": 932}, {"publish_datetime": "2018-12-02 14:42:00+00", "author": "William McInnes", "sentiment_score": 0.274, "sector": "Markets", "category": "-", "sub_category": "", "tags": "", "title_word_count": 9, "summary_word_count": 0, "article_word_count": 591}, {"publish_datetime": "2018-12-02 16:19:00+00", "author": "Neil Hume, Michael Pooler", "sentiment_score": -0.105, "sector": "Companies", "category": "Financial Services", "sub_category": "", "tags": "", "title_word_count": 8, "summary_word_count": 0, "article_word_count": 640}, {"publish_datetime": "2018-12-02 16:41:00+00", "author": "James Hall", "sentiment_score": 0.117, "sector": "Companies", "category": "Retail", "sub_category": "", "tags": "", "title_word_count": 7, "summary_word_count": 0, "article_word_count": 381}, {"publish_datetime": "2018-12-02 17:40:00+00", "author": "Brian K. Sullivan, Naureen S. Malik", "sentiment_score": -0.122, "sector": "Companies", "category": "Energy", "sub_category": "", "tags": "", "title_word_count": 10, "summary_word_count": 0, "article_word_count": 568}, {"publish_datetime": "2018-12-02 23:00:00+00", "author": "Matthew Stevens", "sentiment_score": -0.039, "sector": "Companies", "category": "Electricity", "sub_category": "", "tags": "Electricity, Gas, Opinion", "title_word_count": 11, "summary_word_count": 0, "article_word_count": 1324}, {"publish_datetime": "2018-12-02 23:00:00+00", "author": "Katrina King", "sentiment_score": 0.168, "sector": "Markets", "category": "-", "sub_category": "", "tags": "Opinion, Portfolio management", "title_word_count": 9, "summary_word_count": 0, "article_word_count": 606}, {"publish_datetime": "2018-12-02 23:00:00+00", "author": "Vesna Poljak", "sentiment_score": 0.052, "sector": "Markets", "category": "-", "sub_category": "", "tags": "Sydney Airport, Transurban", "title_word_count": 10, "summary_word_count": 0, "article_word_count": 1380}]' http://localhost:5000/get_prediction

Result
{"success":true, "prediction":[0.018596187, 0.002564243, 0.011357293, 0.022220597, 0.034481943, 0.03061306, 0.033345744, 0.036189508, 0.037143186, 0.02775693, 0.01494329]}
"""
