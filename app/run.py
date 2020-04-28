import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
import nltk

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("models/classifier.pkl")

def freq_top_n_words_with_stopwords(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def freq_top_n_words_no_stopwords(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def freq_top_n_trigram_no_stopwords(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre',as_index =False).count()
    category_values = df.iloc[:,4:].astype(bool).sum(axis=0).sort_values(ascending=False)
    df['message_len'] = df['message'].astype(str).apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
    
    common_words_with_stopwords = freq_top_n_words_with_stopwords(df['message'], 20)
    df_common_words_with_stopwords = pd.DataFrame(common_words_with_stopwords, columns = ['MsgText' , 'count'])
    df_common_words_with_stopwords = df_common_words_with_stopwords.groupby('MsgText').sum()['count'].sort_values(ascending=False)
    
    common_words_no_stopwords = freq_top_n_words_no_stopwords(df['message'], 20)
    df_common_words_no_stopwords = pd.DataFrame(common_words_no_stopwords, columns = ['MsgText' , 'count'])
    df_common_words_no_stopwords = df_common_words_no_stopwords.groupby('MsgText').sum()['count'].sort_values(ascending=False)
    
    common_trigram_no_stopwords = freq_top_n_trigram_no_stopwords(df['message'], 20)
    df_top_n_trigram_no_stopwords = pd.DataFrame(common_trigram_no_stopwords, columns = ['MsgText' , 'count'])
    df_top_n_trigram_no_stopwords = df_top_n_trigram_no_stopwords.groupby('MsgText').sum()['count'].sort_values(ascending=False)
    
    # create visuals
    graphs = [
            # GRAPH 1 - genre graph
        {'data' : [{'type' : 'pie',
                  'name' : "Messages per genres",
                 'labels' : genre_counts['genre'],
                 'values' : genre_counts['id'],
                 'direction' : 'clockwise',
                 'marker' : {'colors' : ["rgb(183,101,184)", "rgb(236,77,216)", "rgb(176,164,216)"]}}],
      'layout' : {'title' : 'Messages per genres - imbalance'}},
            # GRAPH 2 - category graph    
        {'data' : [{'type' : 'bar',
                  'name' : "categories",
                 'x' : category_values.index,
                 'y' : category_values.values,
                 'marker' : {'color': "rgb(255,168,255)"}}],
      'layout' : {'title' : 'Distibution of classes','yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }}},
        # GRAPH 3 - category graph    
        {'data' : [{'type' : 'bar',
                  'name' : "Top 20 without stopwords",
                 'x' : df_common_words_with_stopwords.index,
                 'y' : df_common_words_with_stopwords.values,
                 'marker' : {'color': "rgb(183,101,184)"}}],
      'layout' : {'title' : 'Top 20 words in review with stop words','yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }}},
        # GRAPH 4 - category graph    
        {'data' : [{'type' : 'bar',
                  'name' : "Top 20 without stopwords",
                 'x' :  df_common_words_no_stopwords.index,
                 'y' :  df_common_words_no_stopwords.values,
                 'marker' : {'color': "rgb(236,77,216)"}}],
      'layout' : {'title' : 'Top 20 words in message after removing stop words','yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }}},
        # GRAPH 5 - category graph    
        {'data' : [{'type' : 'bar',
                  'name' : "Top 20 without stopwords",
                 'x' :  df_top_n_trigram_no_stopwords .index,
                 'y' :  df_top_n_trigram_no_stopwords .values,
                 'marker' : {'color': "rgb(176,164,216)"}}],
      'layout' : {'title' : 'Top 20 trigrams in message after removing stop words','yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }}}
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)
   

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()