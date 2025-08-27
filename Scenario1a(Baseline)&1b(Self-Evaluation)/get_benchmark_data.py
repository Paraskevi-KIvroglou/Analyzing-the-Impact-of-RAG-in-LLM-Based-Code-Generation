# pylint: disable=missing-module-docstring
import re
import polars as pl


def load_data():
    """ Load data from Hugging Face. """
    #Retrieve dataset MBBP+ from Hugging Face: https://huggingface.co/datasets/evalplus/mbppplus?row=2
    df = pl.read_parquet('hf://datasets/evalplus/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet')

    return df 

def run_preliminary_analysis(df):
    """ Run preliminary analysis. """
    # Get the shape of the DataFrame
    num_rows, num_columns = df.height, df.width

    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_columns}")

    data_info = df.glimpse()
    print(f"Data types and memory usage: {data_info}")

def get_prompts(df):
    """ Get the prompts from the dataset. """
    prompts = df.select("prompt")

    code_column = df.select("code")
    # Modify the list comprehension to work with DataFrame iteration
    function_names = [
        extract_function_name_from_code(row[0]) 
        for row in code_column.iter_rows()
    ]

    # Convert prompts to list
    prompt_list = [row[0] for row in prompts.iter_rows()]

    updated_prompts = []
    for i in range(len(prompts)):
        function_name, prompt = function_names[i], prompt_list[i]
        updated_prompt = f"Function Name: {function_name}\nPrompt: {prompt}"
        updated_prompts.append(updated_prompt)
    return updated_prompts

def get_expected_responses(df):
    """ Get expected responses from the dataset. Reference code."""
    responses = df.select("code")
    return responses.iter_rows()

def get_rows(df):
    """Get rows from the dataset. """
    return df.shape[0]

def extract_response(cell_value):
    """ Extract the respons of the list, cell value."""
    if isinstance(cell_value, list) and len(cell_value) > 0 and isinstance(cell_value[0], list):
        return cell_value[0][0]
    return cell_value

def extract_function_name_from_code(code_text):
    """ Extract the name of the function from the code (Reference value)"""
    # Use regex to find the function name
    pattern = r'def\s+(\w+)\s*\('
    match = re.search(pattern, code_text)
    
    if match:
        return match.group(1)  # Will return 'similar_elements'
    return None

def __main__():
    """ Load data and run preliminary analysis. """
    df = load_data()
    run_preliminary_analysis(df)