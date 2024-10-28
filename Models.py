from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_classifier(training_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=42)
    
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words='english'), SVC())
    param_grid = {
        'svc__C': [0.01, 0.1, 1, 10, 100, 1000],
        'svc__kernel': ['linear', 'rbf', 'poly'],
        'svc__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
        'svc__degree': [2, 3, 4],
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'tfidfvectorizer__analyzer': ['word', 'char']
    }

    grid_search = GridSearchCV(
        pipeline, 
        param_grid,  
        cv=10,  
        scoring='accuracy',  
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate accuracy on training and test sets
    train_accuracy = accuracy_score(y_train, grid_search.predict(X_train))
    test_accuracy = accuracy_score(y_test, grid_search.predict(X_test))
    
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    
    return grid_search.best_estimator_

def train_random_forest_classifier(training_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=42)
    
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words='english'), RandomForestClassifier())
    param_grid = {
        'randomforestclassifier__n_estimators': [10, 50, 100, 200],
        'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
        'randomforestclassifier__max_depth': [None, 10, 20, 30, 40, 50],
        'randomforestclassifier__min_samples_split': [2, 5, 10],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'tfidfvectorizer__analyzer': ['word', 'char']
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
        
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    
    train_accuracy = accuracy_score(y_train, grid_search.predict(X_train))
    test_accuracy = accuracy_score(y_test, grid_search.predict(X_test))
    
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    
    return grid_search.best_estimator_

def classify_description(classifier, description):
    return classifier.predict([description])[0]

def train_neural_network_classifier(training_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=42)
    
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words='english'), MLPClassifier(max_iter=1000))
    param_grid = {
        'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'mlpclassifier__activation': ['tanh', 'relu'],
        'mlpclassifier__solver': ['sgd', 'adam'],
        'mlpclassifier__alpha': [0.0001, 0.001, 0.01],
        'mlpclassifier__learning_rate': ['constant', 'adaptive'],
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'tfidfvectorizer__analyzer': ['word', 'char']
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
        
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    
    train_accuracy = accuracy_score(y_train, grid_search.predict(X_train))
    test_accuracy = accuracy_score(y_test, grid_search.predict(X_test))
    
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    
    return grid_search.best_estimator_
