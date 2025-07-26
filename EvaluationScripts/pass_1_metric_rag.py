# pylint: disable=trailing-whitespace
import ast
import logging
from typing import List, Union
import itertools
import numpy as np
import re
import textwrap
import os
import sys
import subprocess
sys.path.append(r'C:\Thesis-Project\WriteResults')
import write_results_txt as write
import textwrap

filename = "test_log_rag.txt"
namespace = {"__builtins__": __builtins__}

def calculate_pass_at_1_rag(model_responses, test_cases, check_function_name_tests = False, remove_whitespace=False):
    """ Calculate pass@1 for k=1 responses from the model """
    correct_count = 0
    total_count = len(model_responses)

    # Here we use enumerate to get the i index of the model response because 
    # it is the aligned test case to get the test_case[i] to match the problem.
    
    for i, response in list(enumerate(model_responses)):

        # Extract only the code block from the response
        write.write_results_to_txt(filename, response)
        code_block = extract_code(response, remove_whitespace)
        write.write_results_to_txt("extracted_code_responses.txt", code_block)
        # Execute test cases on the extracted code
        try:
            write.write_results_to_txt(filename, code_block)
            write.write_results_to_txt(filename, test_cases[i])
            # Run all test cases for this problem
            test_results = run_test_cases(code_block, test_cases[i], check_function_name_tests)
            # Count as correct only if all test cases pass
            
            # The all() function is a built-in Python function that returns True if all elements in an iterable are true. If any element is false, it returns False.
            
            write.write_results_to_txt(filename, test_results)
            if all(test_results):
                correct_count += 1
                
        except Exception as e:
            # If code execution fails, count as incorrect
            continue
    
    # Calculate pass@1 score
    pass_at_1 = correct_count / total_count if total_count > 0 else 0
    print(f"Correct count: {correct_count}")
    print(f"Total count: {total_count}")

    #Calculate pass@1 score for each answer 
    pass_k1_research_estimator = estimate_pass_at_k([total_count], [correct_count], 1)
    write.write_results_to_txt(filename, test_results)
    return pass_at_1, correct_count, total_count, pass_k1_research_estimator

def check_six_backticks(input_string):
    """
    Check if the string contains ``````

    (Returns) : bool
    """
    return '``````' in input_string


def check_triple_backticks(input_string):
    """
    Check if the string contains ```

    (Returns) : bool
    """
    return '```' in input_string

def check_triple_quote_ticks(input_string):
    """
    Check if the string contains  " " " 

    (Returns) : bool
    """
    return '"""' in input_string

def check_collon(input_string):
    """
    Check if the string contains  :

    (Returns) : bool
    """
    return ':' in input_string

def clean_invalid_chars_until_def(text):
    """
    Clean invalid characters before def.
    Clean the string from model's output.

    (Returns) : string
    """
    remove_trailing_whitespace(text)
    parts = text.split('def', 1)
    if len(parts) > 1:
        if check_triple_quote_ticks(parts[0]):
            return parts[0].replace('"""', '') + 'def' + parts[1]
        if check_collon(parts[0]):
            return parts[0].replace(':', '') + 'def' + parts[1]
        # if ' ' in parts[0]:
        #     return parts[0].replace(' ', '') + 'def' + parts[1]
    return text

def remove_trailing_whitespace(input_string):
    """
     1. Remove leading/trailing whitespace from each line
     2. Join with consistent newlines

     (Returns) : string
    """
     # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in input_string.splitlines()]
    
    # Join with consistent newlines
    return '\n'.join(lines)

def evaluate_code_syntax(code):
    """
    This function does the following:
    
    1. Cleans code formatted strings outputed from LLMs. 

    2. Write the raw code in a temp.py file 

    3. Use autopep8 to format the code

    4. Read the formatted code and return it. 
    """
    #Clean from intent python\n, python and replace might \\n, ''', and """ 
    # from models output parsing. 
    code_block = code.replace('python\n', '') 
    code_block = code_block.replace('python', '')
    code_block = code_block.replace(' python\n', '')
    code_block = code_block.replace('\\n', '\n')
    if check_six_backticks(code_block):
        code_block = code_block.replace("``````", "")
    if check_triple_backticks(code_block):
        code_block = code_block.replace("```", "")
    if "```n" in code_block:
        code_block = code_block.replace("```n", "")
    # code_block = clean_invalid_chars_until_def(code_block)
    # code_block = remove_trailing_whitespace(code_block)
    write.write_results_to_txt("cleaned_code_log.txt", f"Code Block Before Cleaning:\n{code_block}\n")
    
    # Before writing code to file:
    code_block = textwrap.dedent(code_block)
    # Write the raw code to a temporary file
    with open("temp.py", "w", encoding="utf-8") as file:
        file.write(code_block)

    # Use autopep8 to format the code
    subprocess.run(["autopep8", "--in-place","--aggressive","--aggressive", "--aggressive", "temp.py"])

    # Read the formatted code
    with open("temp.py", "r", encoding="utf-8") as file:
        formatted_code = file.read()

    # Log the cleaned code block
    write.write_results_to_txt("cleaned_code_log.txt", f"Code Block :\n{formatted_code}\n")
    os.remove("temp.py")
    # write.write_results_to_txt("wrong_syntax_log.txt", code_block)
    return f"\n{formatted_code}\n"

def extract_code(response, remove_whitespace=False):
    """
    This function does the following:
        It uses a regex pattern r"``````" to match code blocks.
        text
        (.*?) captures everything until the closing backticks (non-greedy).
        text
        re.findall() is used to find all matches in the text.
        The re.DOTALL flag allows the dot (.) to match newline characters.
        The function returns a list of all matched code blocks.
    """
    # 1. Ensure valid input type (from search results [1][3])
    if not isinstance(response, (str, bytes)):
        response = str(response)  # Convert non-string responses
    
    # 2. Handle byte strings (common in API responses)
    if isinstance(response, bytes):
        response = response.decode('utf-8', errors='ignore')
    
    # 3. Clean non-ASCII characters (from earlier solution)
    response = re.sub(r'[^\x00-\x7F]+', '', response)
    
    # 4. Find code blocks with error handling
    code_pattern = r"```(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)

    if matches:
        for i, code in enumerate(matches, 1):
            return evaluate_code_syntax(code)
    else:
        formatted_code = evaluate_code_syntax(response)
        if remove_whitespace:
            formatted_code =  textwrap.dedent(formatted_code)
        #write.write_results_to_txt("wrong_syntax_log.txt", f"Code Block not corrected: {formatted_code}")  
        return formatted_code

def execute_code(code_block):
    """ Execute the code block and handle exception. """
    try:
        # Parse the code to check for syntax errors
        ast.parse(code_block)
        
        # Create a new namespace for execution
        namespace = {}
        
        # Execute the code
        exec(code_block, namespace)
        
        return "Code executed successfully."
    except SyntaxError as se:
        return f"Syntax Error: {se}"
    except Exception as e:
        return f"Error during execution: {e}"
    
def fix_backslashes(code_str):
    """Remove trailing backslashes not part of line continuation"""
    lines = []
    for line in code_str.split('\n'):
        stripped = line.rstrip()
        if stripped.endswith('\\') and not line.strip().endswith('\\'):
            lines.append(stripped[:-1])
        else:
            lines.append(line)
    return '\n'.join(lines)

def extract_function_name(code_block):
    """
    Extract the function name. 

    (Returns) : str 
    """
    match = re.search(r'def\s+(\w+)\s*\(', code_block)
    function_name = match.group(1) if match else None
    print(function_name)
    return function_name

def extract_isclose_variable(s):
    """
    Extracts the first variable name from an 'assert math.isclose(...)' statement.
    
    Args:
        s (str): The input string containing the assert statement.
    
    Returns:
        str: The extracted variable name if a match is found, None otherwise.
    """
    pattern = r'assert\s+math\.isclose\(\s*(\w+)'
    match = re.search(pattern, s)
    return match.group(1) if match else None

def extract_assert_variable(s):
    """
    Extracts the variable name from an 'assert' statement that matches either:
    - 'assert set(variable_name)'
    - 'assert variable_name('
    
    Args:
        s (str): The input string containing the assert statement.
    
    Returns:
        str: The extracted variable name if a match is found, None otherwise.
    """
    pattern = r'assert\s+(?:set\()?\s*(\w+)\s*\('
    match = re.search(pattern, s)
    return match.group(1) if match else None

def extract_assert_not_function(s):
    """
    Extracts the function name from an 'assert not' statement, such as:
    'assert not count_divisors(100)'.
    
    Args:
        s (str): The input string containing the assert statement.
    
    Returns:
        str: The extracted function name if a match is found, None otherwise.
    """
    pattern = r'assert\s+not\s+(\w+)\('
    match = re.search(pattern, s)
    return match.group(1) if match else None

def extract_assert_parethenses(s):
    """
    Extracts the function name from an 'assert (' statement, such as:
    'assert (count_divisors(100))'.
    
    Args:
        s (str): The input string containing the assert statement.
    
    Returns:
        str: The extracted function name if a match is found, None otherwise.
    """
    pattern = r'assert\s+\((\w+)\(([^)]+)\)\)'
    match = re.search(pattern, s)
    return match.group(1) if match else None


def replace_function_name(assert_statement, correct_function_name):
    """
        Replace the assert statement with the function name that the model 
        generated. These few test cases that the function name is not matching with
        the test case we want to count them and execute them. 

        (Returns) : str 
    """
    match = ""
    if "math.isclose" in assert_statement:
        match = extract_isclose_variable(assert_statement)
    elif "assert not" in assert_statement:
        match = extract_assert_not_function(assert_statement)
    elif "assert (" in assert_statement:
        match = extract_assert_parethenses(assert_statement)
    else:
        match = extract_assert_variable(assert_statement)
    
    if match:
        # old_function_name = match.group(1)
        new_assert_statement = assert_statement.replace(match, correct_function_name)
        return new_assert_statement
    else:
        return assert_statement  # Return original if no function found
    
def execute_code_safely(code_block):
    """ Execute the code safely and try to catch syntax errors. """
    # Validate syntax first
    # This approach resolves the unterminated string literal error while maintaining your desired quote style conversions.
    code_block = rf'''
    {code_block}'''
    #.replace("'''", '"""') 
    #code_block = code_block[:-1]
    # code_block = code_block.strip("\n")
    #code_block = fix_backslashes(code_block) #Fix error ending line \. 
    #Note this causing a problem during execution. Result: Decided to clean the \. code_block[:-1]
    # print(code_block)
    try:
        ast.parse(code_block)
        print("✅ Code is syntactically valid")
    except SyntaxError as e:
        print(f"❌ Syntax Error: {e}")
        error_info = {
            'message': e.msg,
            'line_number': e.lineno,
            'column': e.offset,
            'error_line': e.text.strip(),
            'error_type': 'SyntaxError'
        }
        print(error_info)
        write.write_results_to_txt("wrong_syntax_log.txt", error_info)
        write.write_results_to_txt("wrong_syntax_log.txt", code_block)
        return False, error_info
    
    # Prepare execution environment
    # namespace = {"__builtins__": __builtins__}
    
    # Execute with runtime error handling
    try:
        exec(code_block, namespace)
        return True
    except Exception as e:
        print(f"⚠️ Runtime Error: {type(e).__name__}: {e}")
        return False

def run_test_cases(code_block: str, test_cases: List[str], check_test_function_name = False) -> List[bool]:
    """
    Execute the code and run test cases.
    Return list of boolean results for each test case.
    """
    results = []

    try:
        exec_safe_code = execute_code_safely(code_block)
        if exec_safe_code:
            for test in test_cases:
                function_name = extract_function_name(code_block)
                if check_test_function_name and isinstance(function_name, str):
                    write.write_results_to_txt(filename, function_name)
                    test = replace_function_name(test, function_name)
                    write.write_results_to_txt(filename, test)
                try:
                    # Execute the test case in the same namespace
                    if "assert" in test:
                        test = test.replace("assert", "")
                        exec(f"result = {test}", namespace)
                        test_result = namespace['result']
                        write.write_results_to_txt(filename, test_result)
                        if bool(test_result) is True:
                            results.append(bool(test_result))
                        if bool(test_result) is False:
                            # results.append(bool(test_result))
                            write.write_results_to_txt(filename, test)
                            write.write_results_to_txt(filename, test_result)
                    
                except (ValueError, SyntaxError) as e:
                    logging.warning(f"Invalid test case: {test}. Error: {e}")
                    results.append(False)
                    write.write_results_to_txt(filename, [test, e])
                except Exception as e:
                    logging.error(f"Error executing test case: {test}. Error: {e}")
                    results.append(False)
                    write.write_results_to_txt(filename, [test, e])
                
    except Exception as e:
        logging.error(f"Error executing code: {e}")
        write.write_results_to_txt(filename, f"Code Block : {e}")
        return [False] * len(test_cases)

    return results

# unbiased estimator from https://github.com/openai/human-eval
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.mean(np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    ))
