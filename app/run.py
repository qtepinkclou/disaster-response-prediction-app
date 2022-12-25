import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request
import joblib

import sys
import os
this_dir = os.path.dirname(__file__) # Path to loader.py
sys.path.append(os.path.join(this_dir, 'app/models/pipeline_pipeline.py'))


import re

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain

from train_classifier_just_so_joblib_works import ModelGenerator


URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
PUNC_PAT = re.compile(r'[^a-zA-Z0-9]')


options = [
    'BinaryRelevance-MultinomialNB-TFIDF',
    'ClassifierChain-MultinomialNB-TFIDF',
    'LabelPowerset-MultinomialNB-TFIDF',
    'LabelPowerset-RandomForest-TFIDF',
    'MultiOutput-RandomForest-TFIDF',
    'MultiOutput-RandomForest-Doc2Vec'
]


app = Flask(__name__)
app.debug = True

# load data
db_connect = 'sqlite:///data/DisasterResponse.db'
df = pd.read_sql('supervised_data', con=db_connect)

# load model
model_br_mult = joblib.load('models/BinaryRelevance-MultinomialNB-TFIDF.joblib')
#model_cc_mult =
#model_lp_mult =
#model_lp_rand =
#model_mo_rand_d2v =
#model_mo_rand_tfidf =

genre_columns = df.drop(['message', 'id', 'original', 'genre'], axis=1)
genre_counts = [df[col].sum() for col in genre_columns.columns]
genre_names = [col for col in genre_columns.columns]
genre_var_count = [i for i in range(25)]
genre_var_count_spec_totals = genre_columns.sum(axis=1).value_counts()

br_mult_pd = pd.read_excel('models/BinaryRelevance-MultinomialNB-TFIDF.xlsx').drop('Unnamed: 0', axis=1)
#cc_mult_pd =
#lp_mult_pd =
#lp_rand_pd =
#mo_rand_d2v_pd =
#mo_rand_tfidf_pd =

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    graphs = [
        {
            'data': [
                {
                    'x': genre_names,
                    'y':genre_counts,
                    'type':'bar'
        }],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                {
                    'x': genre_var_count,
                    'y':genre_var_count_spec_totals,
                    'type':'bar'
        }],

            'layout': {
                'title': 'How many labels each message has',
                'yaxis': {
                    'title': "Total message"
                },
                'xaxis': {
                    'title': "Total label in single message"
                }
            }
        },
        {
            'data': [
                {
                    'x': [col for col in br_mult_pd.columns],
                    'y': [br_mult_pd[col][0] for col in br_mult_pd.columns],
                    'type':'line',
                    'name': options[0]
                }
#                {
#                    'x': [col for col in cc_mult_pd.columns],
#                    'y': [cc_mult_pd[col][0] for col in cc_mult_pd.columns],
#                    'type':'line',
#                    'name': options[1]
#                }
            ],

            'layout': {
                'title': 'Model Performance Precision',
                'yaxis': {
                    'title': "Value in %"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                {
                    'x': [col for col in br_mult_pd.columns],
                    'y': [br_mult_pd[col][1] for col in br_mult_pd.columns],
                    'type':'line',
                    'name': options[0]
                }
#                {
#                    'x': [col for col in cc_mult_pd.columns],
#                    'y': [cc_mult_pd[col][1] for col in cc_mult_pd.columns],
#                    'type':'line',
#                    'name': options[1]
#                },
            ],

            'layout': {
                'title': 'Model Performance Recall',
                'yaxis': {
                    'title': "Value in %"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                {
                    'x': [col for col in br_mult_pd.columns],
                    'y': [br_mult_pd[col][2] for col in br_mult_pd.columns],
                    'type':'line',
                    'name': options[0]
                }
#                {
#                    'x': [col for col in cc_mult_pd.columns],
#                    'y': [cc_mult_pd[col][2] for col in cc_mult_pd.columns],
#                    'type':'line',
#                    'name': options[1]
#                }
            ],

            'layout': {
                'title': 'Model Performance F1 Score',
                'yaxis': {
                    'title': "Value in %"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)

@app.route('/go', methods=['GET'])
def classify_message():
    process = request.args.get('process')
    query = request.args.get('query', '')
    if process == options[0]:
        clas_labels = model_br_mult.predict([query])[0]
        clas_labels = clas_labels.toarray()[0]
        clas_res = dict(zip(df.columns[4:], clas_labels))
    else:
        clas_labels = ['NOT IMPLEMENTED']
        clas_res = dict(zip(['NOT IMPLEMENTED'], clas_labels))


    return render_template('go.html', query=query, classification_result=clas_res, process=process)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
