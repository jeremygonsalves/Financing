from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



def train_classifier(training_data, labels):
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words='english'),SVC())
    param_grid = {
        'svc__C': [0.01, 0.1, 1, 10, 100, 1000],  # Regularization parameter
        'svc__kernel': ['linear', 'rbf', 'poly'],  # Kernels: linear, RBF, and polynomial
        'svc__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],  # Kernel coefficient
        'svc__degree': [2, 3, 4],  # Degree for polynomial kernel
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
        'tfidfvectorizer__analyzer': ['word', 'char']  # Use both word and char n-grams
    }

    #grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', verbose=2)
    grid_search = GridSearchCV( #param_grid is typically used with a grid search method, which is why we use GridSearchCV
        pipeline, #pipeline is the model we are using
        param_grid,  #param_grid is the dictionary of hyperparameters we want to search over
        cv=10,  # 100-fold cross-validation
        scoring='f1_weighted',  # Use F1 weighted as the scoring metric for imbalanced classes
        verbose=2,
        n_jobs=-1  # Utilize all cores for faster computation
    )
    
    grid_search.fit(training_data, labels)

    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_



def train_random_forest_classifier(training_data, labels):

    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words='english'), RandomForestClassifier())
    param_grid = {
        'randomforestclassifier__n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
        'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at every split
        'randomforestclassifier__max_depth': [None, 10, 20, 30, 40, 50],  # Maximum number of levels in tree
        'randomforestclassifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
        'tfidfvectorizer__analyzer': ['word', 'char']  # Use both word and char n-grams
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,  # 10-fold cross-validation
        scoring='f1_weighted',  # Use F1 weighted as the scoring metric for imbalanced classes
        verbose=2,
        n_jobs=-1  # Utilize all cores for faster computation
    )
        
    grid_search.fit(training_data, labels)

    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def classify_description(classifier, description):
    return classifier.predict([description])[0]  # Return the predicted category


def train_neural_network_classifier(training_data, labels):
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words='english'), MLPClassifier(max_iter=1000))
    param_grid = {
        'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Different layer configurations
        'mlpclassifier__activation': ['tanh', 'relu'],  # Activation functions
        'mlpclassifier__solver': ['sgd', 'adam'],  # Solvers for weight optimization
        'mlpclassifier__alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
        'mlpclassifier__learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
        'tfidfvectorizer__analyzer': ['word', 'char']  # Use both word and char n-grams
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,  # 10-fold cross-validation
        scoring='f1_weighted',  # Use F1 weighted as the scoring metric for imbalanced classes
        verbose=2,
        n_jobs=-1  # Utilize all cores for faster computation
    )
        
    grid_search.fit(training_data, labels)

    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_