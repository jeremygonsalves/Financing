import pandas as pd
import sys
import time
from Models import train_classifier, classify_description, train_random_forest_classifier, train_neural_network_classifier

def load_transactions(file_path):
    transactions = pd.read_csv(file_path, header=None)
    if len(transactions.columns) == 5:
        transactions.columns = ['Date', 'Description', 'Charge', 'Debit', 'Ignore']
        transactions = transactions.drop(columns=['Ignore'])
        ##if the length of the columns is greater than 4, then the last column is dropped and only the first 4 are used
    else:
        transactions.columns = ['Date', 'Description', 'Charge', 'Debit']
        #otherwise it assumes we can simply take the first 4 columns
    return transactions

def load_training_data(file_path):
    training_data = pd.read_excel(file_path, header=None, skiprows=1)
    training_data = training_data.iloc[:, [1, 4]]
    training_data.columns = ['Description', 'Category']
    training_data['Description'].fillna("", inplace=True)
    training_data.dropna(subset=['Category'], inplace=True)
    descriptions = training_data['Description']
    labels = training_data['Category']
    return descriptions, labels
#this function loads the training data from an excel file and returns the descriptions and labels

def categorize_transactions(transactions, classifier):
    transactions['Category'] = transactions['Description'].apply(lambda desc: classify_description(classifier, desc))
    return transactions

if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) != 3:
        print("Usage: python categorize_transactions.py <training_file_path> <file_path>")
        sys.exit(1)
    
    training_file_path = sys.argv[1]
    file_path = sys.argv[2]
    
    print(f"Loading training data from: {training_file_path}")
    training_data, labels = load_training_data(training_file_path)
    
    print("Training classifier...")
    classifier = train_random_forest_classifier(training_data, labels)  # Train a classifier using your data
    
    print(f"Processing file: {file_path}")
    transactions = load_transactions(file_path)
    categorized_transactions = categorize_transactions(transactions, classifier)
    
    output_file_path = file_path.replace(".csv", "_categorized.csv")
    categorized_transactions.to_csv(output_file_path, index=False)

    end_time = time.time()
    print(f"Categorized transactions saved to: {output_file_path}")
    print(f"Time taken: {end_time - start_time} seconds")
    print("All done!")
    
    