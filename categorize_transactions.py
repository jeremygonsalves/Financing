import pandas as pd
import sys
import time
from Models import classify_description, train_random_forest_classifier
from openpyxl import load_workbook
from datetime import datetime

def load_transactions(file_path):
    transactions = pd.read_csv(file_path, header=None)
    
    if 'wealthsimple' in file_path.lower():
        if len(transactions.columns) > 1:
            transactions = transactions.drop(columns=[1])
        
        transactions.insert(len(transactions.columns) - 1, 'Debit', 0)
        transactions.columns = ['Date', 'Description', 'Charge', 'Debit', 'amount']
        transactions = transactions.drop(columns=['amount'])
        transactions['Charge'] = transactions['Charge'].str.replace('$', '', regex=False)
        transactions['Charge'] = pd.to_numeric(transactions['Charge'], errors='coerce')   
        transactions['Debit'] = transactions['Charge'].apply(lambda x: x if x > 0 else 0)  # Positive values to 'Debit'
        transactions['Charge'] = transactions['Charge'].apply(lambda x: abs(x) if x < 0 else 0)  # Negative values to 'Charge'
    
    elif len(transactions.columns) == 5:
        transactions.columns = ['Date', 'Description', 'Charge', 'Debit', 'Ignore']
        transactions = transactions.drop(columns=['Ignore'], errors='ignore')
    else:
        transactions.columns = ['Date', 'Description', 'Charge', 'Debit']
    
    if 'Date' in transactions.columns:
        transactions['Date'] = pd.to_datetime(transactions['Date'], errors='coerce')
        transactions['Date'] = transactions['Date'].dt.strftime("%Y-%m-%d")
    
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
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("All done!")
    
    output_excel_path = file_path.replace('.csv', '_categorized.xlsx')

    print(f"Saving categorized transactions to: {output_excel_path}")

    # Save the categorized transactions to a new Excel file
    categorized_transactions.to_excel(output_excel_path, index=False, sheet_name='Categorized Transactions')

    print(f"Categorized transactions saved to: {output_excel_path}")
