'''module to generate/fit/predict/evaluate various models applicable to the project.'''
import sys
import re
import multiprocessing
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

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

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.base import BaseEstimator, TransformerMixin

cores = multiprocessing.cpu_count()

URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
PUNC_PAT = re.compile(r'[^a-zA-Z0-9]')


class ModelGenerator:
    '''generate/fit/predict/evaluate various models applicable to the project.'''

    def __init__(self, database_filepath, model_type='BinaryRelevance-MultinomialNB-TFIDF'):
        '''init'''
        self.database_filepath = database_filepath[0]
        self.model_type = model_type
        self.features_data = None
        self.labels_data = None
        self.label_names = None
        self.features_train = None
        self.features_test = None
        self.labels_train = None
        self.labels_test = None
        self.model = None
        self.model_type = model_type
        self.results = None

    def tokenize(self, text: str):
        '''Tokenize given text as follows.'''
        # Remove url
        detected_urls = URL_PATTERN.findall(text)
        for url in detected_urls:
            text = text.replace(url, 'http')

        # Lowercase all
        text = text.lower()

        # Remove punc
        text = PUNC_PAT.sub(' ', text)

        # Tokenize words
        words = [w for w in word_tokenize(text)]

        # Get rid of stopwords
        words = [w for w in words if w not in set(stopwords.words('english')) ]

        # Lemmatize
        words = [WordNetLemmatizer().lemmatize(w) for w in words]

        # Stemming
        words = [PorterStemmer().stem(w) for w in words]

        return words

    def relegate_pipeline(self, mode: str):
        '''Given 'mode' argument, return respective pipeline or grid search object.
           You can add additional options here to produce different models.'''
        if mode == 'BinaryRelevance-MultinomialNB-TFIDF':
            return Pipeline([
                ('vect', CountVectorizer(tokenizer=self.tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', BinaryRelevance(MultinomialNB()))
                ])

        elif mode == 'ClassifierChain-MultinomialNB-TFIDF':
            return Pipeline([
                ('vect', CountVectorizer(tokenizer=self.tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', ClassifierChain(MultinomialNB()))
            ])

        elif mode == 'LabelPowerset-MultinomialNB-TFIDF':
            return Pipeline([
                ('vect', CountVectorizer(tokenizer=self.tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', LabelPowerset(MultinomialNB()))
            ])

        elif mode == 'LabelPowerset-RandomForest-TFIDF':
            return Pipeline([
                ('vect', CountVectorizer(tokenizer=self.tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', LabelPowerset(RandomForestClassifier()))
            ])

        elif mode == 'MultiOutput-RandomForest-TFIDF':
            pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=self.tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])
            parameters = {
                'vect__ngram_range': ((1, 1), (1, 2)),
                'clf__estimator__n_estimators': [50, 100],
                'clf__estimator__min_samples_split': [2, 4],
                'clf__estimator__verbose': [1],
            }
            # create grid search object
            return GridSearchCV(pipeline, param_grid=parameters, cv=3)

        elif mode == 'MultiOutput-RandomForest-Doc2Vec':
            pipeline = Pipeline([
                ('d2v', D2VTransformer(tokenizer=self.tokenize)),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])
            parameters = {
                'd2v__negative': [0, 5],
                'd2v__epochs': [20, 30],
                'clf__estimator__min_samples_split': [2, 3, 4],
                'clf__estimator__verbose': [1],
            }
            # create grid search object
            return GridSearchCV(pipeline, param_grid=parameters, cv=3)

        else:
            raise NotImplementedError

    def build_model(self):
        '''create model object as well as train test splits.'''
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(
            self.features_data,
            self.labels_data,
            test_size=0.3,
            random_state=42
        )
        model = self.relegate_pipeline(mode=self.model_type)
        self.model = model
        return model

    def fit(self):
        '''fit the model'''
        self.model.fit(self.features_train, self.labels_train)

    def load_data(self):
        '''load data from database to import feature and label information.'''
        db_connect = f'sqlite:///{self.database_filepath}'
        d_f = pd.read_sql('supervised_data', con=db_connect)
        self.features_data = d_f.message.values

        y_df = d_f.drop(['id', 'message','original', 'genre'], axis=1)
        self.label_names = y_df.columns
        self.labels_data = y_df.to_numpy()

    def evaluate_model(self):
        '''evaluate trained model by its success in terms of both on-per label basis
        as well as micro-averaged values of precision, recall and f1 score'''
        label_pred = self.model.predict(self.features_test)
        if self.model_type.startswith(('BinaryRelevance', 'ClassifierChain', 'LabelPowerset')):
            label_pred = label_pred.toarray()
        label_pred_df = pd.DataFrame(data=label_pred, columns=self.label_names)
        label_test_df = pd.DataFrame(data=self.labels_test, columns=self.label_names)
        self.results = pd.DataFrame(columns=self.label_names)

        for column in label_pred_df.columns:
            precision, recall, f_score, _ = precision_recall_fscore_support(
                label_test_df[column],
                label_pred_df[column] ,
                average='weighted'
            )
            self.results[column] = [precision, recall, f_score]

        tot_precision = 0
        tot_recall = 0
        tot_fscore = 0
        for column in label_pred_df.columns:
            tot_precision += self.results[column][0]
            tot_recall += self.results[column][1]
            tot_fscore += self.results[column][2]

        self.results['micro_avg'] = [tot_precision/len(label_pred.T), tot_recall/len(label_pred.T), tot_fscore/len(label_pred.T)]
        for column in self.results.columns:
            print('\n Category Name: ', column, '\n')
            print('Precision: ', self.results[column][0])
            print('Recall: ', self.results[column][1])
            print('F1 Score: ', self.results[column][2])
        self.results.to_excel(f'models/{self.model_type}.xlsx')


class D2VTransformer(BaseEstimator, TransformerMixin):
    '''Doc2Vec plugin for pipeline'''

    def __init__(
        self,
        tokenizer,
        vector_size=150,
        window=5,
        min_count=2,
        negative=5,
        workers=cores,
        epochs=40
        ):

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.workers = workers
        self.epochs = epochs
        self.model_ = None
        self.tokenizer = tokenizer

    def fit(self, docs, labels=None):
        '''fit'''
        labeled_documents = [
            TaggedDocument(words=self.tokenizer(text), tags=[i]) for i, text in enumerate(docs)
        ]

        model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            negative=self.negative,
            workers=self.workers
        )

        model.build_vocab(labeled_documents)
        model.train(labeled_documents, total_examples=model.corpus_count, epochs=self.epochs)
        self.model_ = model
        return self

    def transform(self, docs):
        '''transform'''
        return [self.model_.infer_vector(self.tokenizer(text)) for text in docs]


def main():
    '''main'''
    if len(sys.argv) >= 2:

        if len(sys.argv) == 2:
            database_filepath = sys.argv[1:]
            classifier = ModelGenerator(
                database_filepath=database_filepath
            )

        elif len(sys.argv) == 3:
            database_filepath, model_type = sys.argv[1:]
            classifier = ModelGenerator(
                database_filepath=database_filepath,
                model_type=model_type
            )

        print('Loading data...\n    DATABASE: {}'.format(classifier.database_filepath))
        classifier.load_data()

        print('Building model...')
        model = classifier.build_model()

        print('Training model...')
        classifier.fit()

        print('Evaluating model...')
        classifier.evaluate_model()

        print('Saving model...\n    MODEL: {}'.format(classifier.model_type))
        joblib.dump(model, f'models/{classifier.model_type}.joblib')

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
