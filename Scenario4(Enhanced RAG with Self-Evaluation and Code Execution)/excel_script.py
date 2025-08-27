import pandas as pd

# Read Excel file, column A
def read_excel_column(file_path):
    try:
        df = pd.read_excel(file_path, usecols='A')
        return df.iloc[:, 0].tolist()  # Convert column to list
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []
    
def read_excel(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        print("Columns in the DataFrame:")
        print(df.columns)  # Print column names
        return df # Convert column to list
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

# Find missing prompts
def find_missing_prompts(excel_prompts, second_excel_prompts):
    missing_prompts = [prompt for prompt in excel_prompts if prompt not in second_excel_prompts]
    return missing_prompts

def find_duplicates(file_path, sheet_name='Sheet1'):
    """
    Function to check if column A in an Excel spreadsheet has duplicates and return the duplicated values.
    
    Parameters:
    - file_path (str): Path to the Excel file.
    - sheet_name (str): Name of the sheet to read (default is 'Sheet1').
    
    Returns:
    - tuple: A boolean indicating if duplicates exist, and a list of duplicated values (if any).
    """
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Check if column A exists
    if 'query' in df.columns:
        # Find duplicated values
        duplicates = df['query'][df['query'].duplicated(keep=False)].unique()
        return len(duplicates) > 0, list(duplicates)
    else:
        raise ValueError("Column 'query' does not exist in the spreadsheet.")

def find_phrase_excel(df, phrase = "Agent stopped due to iteration limit or time limit"):
    # Find the indices (cell numbers) where the phrase exists
    matching_indices = df[df['output'].str.contains(phrase, na=False)].index.tolist()

    # Save the results in a list
    results_list = matching_indices

    print("Matching cell numbers:", results_list)
    return results_list

# Main function
def main():
    excel_file_path ='qwen2.5-coder32b_mbpp_plus_results_scenario4_.xlsx'
    second_excel_file_path =  'qwen2.5-coder32b_mbpp_plus_results_scenario5_test.xlsx'  # Changed to .xlsx

    excel_prompts = read_excel_column(excel_file_path)
    second_excel_prompts = read_excel_column(second_excel_file_path)

    missing_prompts = find_missing_prompts(excel_prompts, second_excel_prompts)

    if missing_prompts:
        print("Prompts not found in the second Excel file:")
        for prompt in missing_prompts:
            print(prompt)
    else:
        print("All prompts are present in the second Excel file.")
    return missing_prompts
    # try:
    #     # has_dupes, duplicate_values = find_duplicates(excel_file_path)
    #     df =  read_excel(excel_file_path)
    #     phrase_list = find_phrase_excel(df)
    #     if phrase_list:
    #         print("The phrase exists in the file. The values in the file:", phrase_list)
    #     else:
    #         print("The phrase doesn't exists in the file.")
    # except ValueError as e:
    #     print(e)

if __name__ == "__main__":
    main()
