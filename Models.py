from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#import matplotlib.pyplot as plt


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


##def train_xgboost_classifier(training_data, labels):
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded)

    # Define the pipeline
    pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ])

    # Define the parameter grid
    param_dist = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__max_features': [None, 5000, 10000],
        'tfidf__analyzer': ['word', 'char', 'char_wb'],
        'xgb__n_estimators': [100, 200, 300, 400, 500],
        'xgb__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'xgb__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'xgb__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    # Use RandomizedSearchCV for initial broad search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    # Fit the random search
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Print the results
    print("Best parameters:", random_search.best_params_)
    print("Training accuracy:", random_search.best_score_)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return best_model

    
    
    
    
def train_random_forest_classifier2(training_data, labels):
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded)

    # Define the pipeline
    pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Define the parameter grid
    param_dist = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__max_features': [None, 5000, 10000],
        'tfidf__analyzer': ['word', 'char', 'char_wb'],
        'rf__n_estimators': [100, 200, 300, 400, 500],
        'rf__max_features': ['auto', 'sqrt', 'log2'],
        'rf__max_depth': [None, 10, 20, 30, 40, 50],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__bootstrap': [True, False]
    }

    # Use RandomizedSearchCV for initial broad search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    # Fit the random search
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Print the results
    print("Best parameters:", random_search.best_params_)
    print("Training accuracy:", random_search.best_score_)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return best_model



def train_random_forest_classifier(training_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=42)
    
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words='english'), RandomForestClassifier())
    param_grid = {
        'randomforestclassifier__n_estimators': [ 100, 200],
        'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
        'randomforestclassifier__max_depth': [None, 40, 50],
        'randomforestclassifier__min_samples_split': [2, 5, 10],
        'randomforestclassifier__min_samples_leaf': [1, 10],
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'tfidfvectorizer__analyzer': ['word', 'char']
    }

    grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=20,
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
