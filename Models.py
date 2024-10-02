from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def train_classifier(training_data, labels):
    # Create a pipeline with TfidfVectorizer and SVM
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words='english'),
                             SVC())

    # Define the parameter grid for SVM
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],  # Regularization parameter
        'svc__kernel': ['linear', 'rbf'],  # Kernel types: linear and RBF
        'svc__gamma': ['scale', 'auto'],  # Kernel coefficient for RBF
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
    }

    # Use GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', verbose=1)
    grid_search.fit(training_data, labels)

    # Print the best parameters and return the best model
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_



def classify_description(classifier, description):
    # Use the trained classifier to predict the category for a single description
    return classifier.predict([description])[0]  # Return the predicted category
