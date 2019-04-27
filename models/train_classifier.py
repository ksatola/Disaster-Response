import sys

import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    '''
    Leads messages table from an sqlite database file
    Args:
        database_filepath (string): filepath to sqlite database file
    Returns:
        X (pandas dataframe): messages strings
        Y (pandas dataframe): multiclassification labels
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    
    # split into independent and dependent variables
    X = df.message
    Y = df.loc[:, 'related':'direct_report']
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    '''
    Converts text to lowercase, removes punctuation characters, 
    splits text into words, removes stop words and lemmatize them
    Args:
        text (string): text to be tokenized
    Returns:
        tokens (list): tokens from the text
    '''
    
    # Convert to lowercase
    text = text.lower() 
    
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Split text into words using NLTK
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed


def build_model():
    '''
    Returns Grid Search model
    Args:
        None
    Returns:
        GridSearchCV (sklearn): multi-output classifier model based on Random Forest
    '''
    
    # Define pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    
    # The parameters are narrowed to bare minimum to shorten the execution time
    parameters = {
        #'tfidf__max_df': (0.8, 1.0),
        #'clf__estimator__max_features': ('auto', 'sqrt', 'log2'),
        #'clf__estimator__min_samples_split': (2, 10, 20, 50, 100),
        'clf__estimator__n_estimators': [10, 20]
    }

    # Return grid seasrch model
    return GridSearchCV(pipeline, parameters, cv=3, n_jobs=1)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints precision, recall and f1 scores for multi-output classification
    Args:
        model (sklearn): multi-output classifier model
        X_test (pandas dataframe): test dataset
        Y_test (pandas dataframe): classificators
        category_names (list): classificators names
    Returns:
        None
    '''
    
    # Predict
    Y_pred = model.predict(X_test)
    
    # Summarize the results of the grid search
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))
    

def save_model(model, model_filepath):
    '''
    Save the model to disk
    Args:
        model (sklearn model): trained model
        model_filepath (string): filepath to pickle file
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        #print(model)
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