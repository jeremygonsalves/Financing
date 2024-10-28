import pandas as pd
import sys
import time
from Models import train_classifier, classify_description, train_random_forest_classifier, train_neural_network_classifier
from openpyxl import load_workbook

def load_transactions(file_path):
    transactions = pd.read_csv(file_path, header=None)
    if len(transactions.columns) == 5:
        transactions.columns = ['Date', 'Description', 'Charge', 'Debit', 'Ignore']
        transactions = transactions.drop(columns=['Ignore'])
    else:
        transactions.columns = ['Date', 'Description', 'Charge', 'Debit']
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
    
    # Training and test accuracy will already be printed from `train_random_forest_classifier`.
    
    print(f"Processing file: {file_path}")
    transactions = load_transactions(file_path)
    categorized_transactions = categorize_transactions(transactions, classifier)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("All done!")
    
    output_excel_path = 'Budget Planner.xlsx'
    sheet_name = 'Transaction Data 2024'

    print(f"Appending categorized transactions to: {output_excel_path}, Sheet: {sheet_name}")

    # Load the existing workbook
    workbook = load_workbook(output_excel_path)
    if sheet_name not in workbook.sheetnames:
        print(f"Sheet {sheet_name} does not exist in {output_excel_path}.")
        sys.exit(1)

    # Load the existing sheet
    sheet = workbook['Transaction Data 2024']

    # Append the categorized transactions to the sheet
    for row in categorized_transactions.itertuples(index=False, name=None):
        sheet.append(row)

    # Save the workbook
    workbook.save(output_excel_path)

    print(f"Categorized transactions appended to: {output_excel_path}, Sheet: {sheet_name}")
