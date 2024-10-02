import pandas as pd
import sys
import time
from Models import train_classifier, classify_description

def load_transactions(file_path):
    transactions = pd.read_csv(file_path, header=None)
    transactions.columns = ['Date', 'Description', 'Charge', 'Debit', 'Ignore']
    transactions = transactions.drop(columns=['Ignore'])
    return transactions

# Modify this function to handle missing values and load training data
def load_training_data(file_path):
    # Load the Excel file without headers
    training_data = pd.read_excel(file_path, header=None)
    
    # Select only the relevant columns (assuming first is 'Description' and second is 'Category')
    training_data = training_data.iloc[:, [1, 4]]
    
    # Manually assign column names based on the expected format
    training_data.columns = ['Description', 'Category']
    
    # Fill missing values in 'Description' with an empty string
    training_data['Description'].fillna("", inplace=True)
    
    # Drop rows where 'Category' is missing, as it is necessary for training
    training_data.dropna(subset=['Category'], inplace=True)
    
    descriptions = training_data['Description']
    labels = training_data['Category']
    return descriptions, labels

def categorize_transactions(transactions, classifier):
    # Use the trained classifier to predict the category for each transaction
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
    classifier = train_classifier(training_data, labels)  # Train a classifier using your data
    
    print(f"Processing file: {file_path}")
    transactions = load_transactions(file_path)
    categorized_transactions = categorize_transactions(transactions, classifier)
    
    output_file_path = file_path.replace(".csv", "_categorized.csv")
    categorized_transactions.to_csv(output_file_path, index=False)

    end_time = time.time()
    print(f"Categorized transactions saved to: {output_file_path}")
    print(f"Time taken: {end_time - start_time} seconds")
