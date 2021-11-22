# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    """
       Function:
       load data from database
       Args:
       database_filepath: the path of the database required
       Return:
       X (DataFrame) : features dataframe
       Y (DataFrame) : target dataframe
    """
    # create engine to the database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    # read in the sql database
    df = pd.read_sql_table(con = engine, table_name='Disaster_Table')
    
    # split the data into features and target variables
    X = df['message']
    Y = df[df.columns[3:]]
    
    return X, Y


def tokenize(text):
    """
       Function:
       Tokenize the given text
       Args:
       text: text input to be tokenized
       Return:
       lemm: Tokenized, normalized, lemmatized version of the text
    """
    # delete any character that is not alphabet and number
    # normalize the text to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize the text into different words
    words = word_tokenize(text)
    
    # remove any stopwords in the text
    stop = stopwords.words("english")
    words = [w for w in words if w not in stop]
    
    # Lemmatize the text
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemm


def build_model():
    """
       Function:
       Build a ML pipeline and GridSearch on some of the pipeline parameters
       Args:
       None
       Return:
       cv: GridSearch object built on ML pipeline
    """
    # create ML pipline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    # Obtain paramaters of the pipeline
    parameters = pipeline.get_params()
    param = {'tfidf__use_idf':(True, False),
        'moc__estimator__n_estimators':[8,12]
        }
    
    # Use GridSearch to find the optimal results
    cv = GridSearchCV(pipeline, param_grid=param)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
       Function:
       Use model to predict on X_test and evaluete its performance
       Args:
       model: The model to be tested
       X_test: Test set input values
       Y_test: Test set y values. Truth values corresponding to X_test
       Return:
       None
    """
    # Use model to predict on test set
    y_pred = model.predict(X_test)
    
    # Report the f1 score, precision and recall for each output category of the dataset
    i = 0
    for col in Y_test:
        print(classification_report(Y_test[col], y_pred[:,i]))
        i += 1
        
    # Print the accuracy of the model
    acc = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.2f}'.format(acc))
    


def save_model(model, model_filepath):
    """
       Function:
       Save the model as a file
       Args:
       model: model to be saved
       model_filepath: the path where model will be saved
       Return:
       None
    """
    file = model_filepath
    with open (file, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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