#import libraries
import sys
import re
import pickle 
import numpy as np
import pandas as pdimport nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

# Import datasets for NLTK
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Load data from SQLite database. 
    Function returns: 
    1. X - the independent variables
    2. Y - the target labels (This is a multiclass machine learning exercise
    3. features- the list of features in the target
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(con=engine, table_name=database_filepath)
    features = ['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']
    X = df['message'].values
    y = df[features].values

    return X, y, features


def tokenize(text):
    '''
    Functions cleans and tokenizes a body of text: 
        1. urls are replaced with urlplaceholder
        2. the text is then tokenized using NLTK word_tokenizer
        3. Engloish stopwords are removed
        4. Lemmatization is performed using WordNetLemmatizer
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # remove urls
    
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    ''' 
    This is the model building phase of te machine learning process
    function builds a machine learning model pipeline with:
    countvectoriser, tf-idf transformer and randomforestclassifier
    GridSearchCV is used for finding the optimal parameters for the model
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__min_samples_split': [10, 2],
        'clf__estimator__n_estimators': [5, 10, 100]
    }

    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv 



def evaluate_model(model, X_test, Y_test, features):
    '''
    This is the evalaution phase of the Machine Learning Process
    Evaluate the model performance looking at f1, accuracy and recall score. 
    Print classification report. 
    '''
    
    y_pred = model.predict(X_test) # Run moel on test features
    
    # Get performance for each feature in the target
    for i in range(len(features)):
        labels = np.unique(y_pred[i])
        confusion_mat = confusion_matrix(Y_test[i], y_pred[i], labels=labels)
        accuracy = (y_pred[i] == Y_test[i]).mean()
        f1 = f1_score(Y_test[i], Y_pred[i], labels=labels)
        recall = recall_score(Y_test[i], Y_pred[i], labels=labels)
        
        print("Feature:", features[i])
        print("Labels:", labels)
        print("Confusion Matrix:\n", confusion_mat)
        print("Accuracy:", accuracy, "F1 score:", f1, "Recall:", recall)
    
    print("\nBest Parameters:", model.best_params_) 


def save_model(model, model_filepath):
    '''
    save model weights 
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