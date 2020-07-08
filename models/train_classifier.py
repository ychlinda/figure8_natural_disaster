import sys
import re
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import nltk
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load a database table and return values, labels and category category names
    Arg:
    filepath of the database file
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_categories', con=engine)

    X = df['message'].values
    category_names = df.columns[4:]
    Y = df[category_names].values
    return X, Y, category_names


def tokenize(text):
    """
    Normaliz, tokenize, lematize the text

    Arg: text
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    stop_words = stopwords.words("english")

    #tokenize
    words = word_tokenize(text)

    words_lemmed = [WordNetLemmatizer().lemmatize(w).strip() for w in words if w not in stop_words]

    return words_lemmed


def build_model():
    """
    Build machine learning pipeline

    """
    pipeline = Pipeline([
                     ('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, class_weight='balanced')))
                    ])
    parameters =  {
        'vect__ngram_range': ((1, 1), (1, 2))
    }

    model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, verbose=3)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performances by f1 score, precision and recall

    Args:
    model: the model that build_model function returns
    X_test: test set of disaster messages
    y_test: test set of disaster categories
    category_names: disaster category names
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("-" *20, "\n")
        print(category_names[i], "\n")
        print(" Precision: {}\t\t Recall: {}\t\t F1_score: {}".format(
            precision_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            recall_score(Y_test[:, i], Y_pred[:, i],average='weighted'),
            f1_score(Y_test[:, i], Y_pred[:, i], average='weighted')))


def save_model(model, model_filepath):
    """
    save the model as a pickle file

    Args:
    model: the optimized classifier
    model_filepath: string, file location
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
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
