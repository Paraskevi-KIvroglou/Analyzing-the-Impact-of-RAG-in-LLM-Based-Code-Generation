# pylint: disable=function-redefined
from typing import Dict, List, Any

def write_results_txt(filename, dict_scores:dict):
    """ 
    Write score results in a txt file 
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for key, value in dict_scores.items():

                file.write(f"{key} Score: {value}\n")
    except PermissionError:
        print(f"Error: No write permissions for '{filename}'.")
    except OSError as e:  # Catch other OS-related errors
        print(f"OS error: {e}")

def write_results_to_txt(filename, res_list:List):
    """ 
    Write runtime results in a txt file 
    """
    with open(filename, 'a') as file:
        for i in range(len(res_list)):
            value = res_list[i]
            file.write(f"R {i}: \n {value}\n")

def write_results_to_txt(filename, res):
    """ 
    Write runtime results of type Any in a txt file 
    """
    try:
        with open(filename, 'a', encoding='utf-8', errors='replace') as file:
            file.write(f"R: \n { res }\n")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Creating it...")
        with open(filename, 'w', encoding='utf-8') as file:  # Create file if missing
            file.write(f"R: \n{res}\n")
    except PermissionError:
        print(f"Error: No write permissions for '{filename}'.")
    except OSError as e:  # Catch other OS-related errors
        print(f"OS error: {e}")
    print(f"Data appended to file: {filename}")