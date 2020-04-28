import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier



def load_data(database_filepath):
    """
    This function load sql db
    
    Arguments:
        database_filepath = path to SQLite db
    Output:
        X = feature DataFrame
        Y = targets DataFrame
        category_names 
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text = list of text messages 
    Output:
        clean_words = tokenized text, clean from 
                      stop words and puntuaction
    """
    stop_words = stopwords.words("english")
    symbol_to_drop = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_symbol = re.findall(symbol_to_drop, text)
    
    for s in detected_symbol:
        text = text.replace(s, "placeholder")
        
    
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.lower())
    stemmed_words = [PorterStemmer().stem(w) for w in words]
    clean_words = [WordNetLemmatizer().lemmatize(w) for w in stemmed_words if w not in stop_words]
    
    return clean_words



def build_model():
    
    pipeline = Pipeline([('countvec', CountVectorizer(tokenizer=tokenize)),
                                 ('tfidf', TfidfTransformer()),
                                 ('lsa', TruncatedSVD(random_state=42,
                                                      n_components=30)),
                                 ('clf',
                                  MultiOutputClassifier(RandomForestClassifier()))
                               ])

    parameters = {
                    'clf__estimator__n_estimators': [10, 20],
                    'clf__estimator__min_samples_split': [2, 5]
                  }

    cv = GridSearchCV(pipeline, param_grid= parameters, verbose=7)
        
    
    return cv

def classification_evaluation(y_true, y_pred):
    
    y_pred = pd.DataFrame(y_pred, columns = y_true.columns)
    report = pd.DataFrame ()
    
    for col in y_true.columns:
        
        class_dict = classification_report (output_dict = True, y_true = y_true.loc [:,col], y_pred = y_pred.loc [:,col])
    
       
        eval_df = pd.DataFrame (pd.DataFrame.from_dict (class_dict))
        
        eval_df.drop(index = 'support', inplace = True)
        
        
        av_eval_df = pd.DataFrame (eval_df.transpose ().mean ())
        
       
        av_eval_df = av_eval_df.transpose ()
    
        
        report = report.append (av_eval_df, ignore_index = True)    
        
    return report

def evaluate_model(model, X_test, Y_test, category_names):
    """
    
    This function applies pipeline to a test set and prints out
    model metrics
    
    Arguments:
        model = Scikit ML Pipeline
        X_test = test features
        Y_test = test labels
        category_names = label names 
    """
    Y_pred = model.predict(X_test)
    
    print(classification_evaluation(Y_test, Y_pred))
    pass


def save_model(model, model_filepath):
    """
    
    This function saves trained model as Pickle file,
    
    Arguments:
        model = GridSearchCV  object
        model_filepath =  path to save pkl file
    
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()