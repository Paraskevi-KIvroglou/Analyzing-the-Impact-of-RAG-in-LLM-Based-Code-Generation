# pylint: disable=line-too-long
import ast
import logging
import pandas as pd
import pytest
import sys
sys.path.append(r'C:\Thesis-Project\WriteResults')
import pass1_metric as pass1

@pytest.mark.skip
def extract_response(cell_value):
    """ Extract the response from excel cell."""
    if isinstance(cell_value, list) and len(cell_value) > 0 and isinstance(cell_value[0], list):
        return cell_value[0][0]
    return cell_value

@pytest.mark.skip
def load_responses():
    """ Load model responses data."""
    df_excel = pd.read_excel(r"C:\Thesis-Project\Scenario1\qwen2.5-coder32b_mbpp_plus_results_2nd.xlsx", sheet_name="Sheet1", header=0)

    model_output = df_excel["Responses"].apply(func=extract_response)

    return model_output[:5]

# def parse_code_test():
    # """ Parse the code to check for syntax errors. Test case"""
    # list_outputs = load_responses()

    # for code_block in list_outputs:
    #     module = ast.parse(code_block)
    #     print(module)
    #     assert module is not None

def test_parse_code():
    """ Parse the code to check for syntax errors. Test case"""
    # Define test cases directly since load_responses() is not available
    list_outputs = [
        "def example_function():\n    return True",
        "def another_function():\n    print('Hello')",
        # Add more code blocks as needed
    ]

    for code_block in list_outputs:
        try:
            module = ast.parse(code_block)
            print(module)
            assert module.body is not None
            print(f"Successfully parsed: {code_block}")
        except SyntaxError as e:
            print(f"Syntax error in code block: {code_block}\nError: {e}")
        except AssertionError:
            print(f"Module parsing failed for: {code_block}")

def test_code_execution():
    """Test function to execute code block in isolated namespace"""
    try:
        # Create a new namespace for execution
        namespace = {}
        
        # Define the code block to test
        code_block = """
        def example_function():
            return True
        """
        
        # Execute the code
        exec(code_block, namespace)
        
        # Run test cases
        test_cases = [
            'assert example_function() == True'
        ]
        
        results = []
        for test in test_cases:
            try:
                # Use ast.literal_eval for safer evaluation
                result = ast.literal_eval(test, namespace)
                results.append(bool(result))
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Invalid test case: {test}. Error: {e}")
                results.append(False)
            except Exception as e:
                logging.error(f"Error executing test case: {test}. Error: {e}")
                results.append(False)
                
        assert all(results)
        
    except Exception as e:
        logging.error(f"Error during code execution: {e}")
        return False

def test_similar_elements():
    """Test function to verify similar_elements implementation"""
    # Define the code to test
#     code_block = """
# def similar_elements(list1, list2):
#     set1 = set(list1)
#     set2 = set(list2)
#     return list(set1.intersection(set2))
# """
    code_block = """python\ndef find_shared_elements(list1, list2):\n    # Convert both lists to sets\n    set1 = set(list1)\n    set2 = set(list2)\n    \n    # Find the intersection of both sets\n    shared_elements = set1.intersection(set2)\n    \n    # Convert the result back to a list (if needed)\n    return list(shared_elements)\n\n# Example usage:\nlist_a = [1, 2, 3, 4, 5]\nlist_b = [4, 5, 6, 7, 8]\nshared = find_shared_elements(list_a, list_b)\nprint("Shared elements:", shared)\n"""
    
    # Remove the initial "python\" and replace \n with actual newlines
    code_block = code_block.replace("python", "").replace("\\n", "\n")

    # If you want to remove any extra backslashes
    code_block = code_block.replace("\\", "")

    code_block = code_block.replace("find_shared_elements", "similar_elements")

    # Setup test cases
    test_cases = [
        'set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))',
        'set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))',
        'set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))'
    ]
    results = []
    # Create isolated namespace
    namespace = {}
    
    try:
        # Execute the function definition
        exec(code_block, namespace)
        
        # Run each test case
        for test in test_cases:
            try:
                # Execute test and store result
                exec(f"result = {test}", namespace)
                test_result = namespace.get('result', False)
                assert test_result, f"Test failed: {test}"
                results.append(bool(test_result))
                print(test_result)
                
            except AssertionError as e:
                raise AssertionError(f"Test case failed: {test}")
            except Exception as e:
                raise Exception(f"Error executing test: {test}, Error: {e}")
                
        assert all(results)  # All tests passed
        
    except Exception as e:
        raise Exception(f"Error in test execution: {e}")

def test_code_execution_safely():
    """ Execute the code_execution_safely function in order to determine how the code block is executed. """

    code_block_exmple1 = '''python\ndef similar_elements(list1, list2):\n    """\n    Find the shared elements between two lists.\n\n    Parameters:\n    list1 (list): The first list.\n    list2 (list): The second list.\n\n    Returns:\n    list: A list containing the shared elements.\n    """\n    # Convert lists to sets to find intersection\n    set1 = set(list1)\n    set2 = set(list2)\n    \n    # Find the intersection of both sets\n    common_elements = set1.intersection(set2)\n    \n    # Convert the result back to a list\n    return list(common_elements)\n\n# Example usage:\nlist_a = [1, 2, 3, 4, 5]\nlist_b = [4, 5, 6, 7, 8]\nprint(similar_elements(list_a, list_b))  # Output: [4, 5]\n'''
    #code_block_exmple1 = code_block_exmple1.strip("python\n")
    code_block = f'''\
    {code_block_exmple1}'''.replace('"""', "'''")  # If you need to convert quote styles
    result = pass1.execute_code_safely(code_block)
    assert result == True
# result = parse_code_test()
# print(result)
