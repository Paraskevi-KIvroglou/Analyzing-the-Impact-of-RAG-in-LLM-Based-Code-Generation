import os
import sys
import pandas as pd
from tqdm import tqdm

# Scenario 1 add on files in system path
sys.path.append(r"C:\Thesis-Project\Scenario1")
sys.path.append(r'C:\Thesis-Project\WriteResults')
sys.path.append(r"C:\Thesis-Project\Evaluation_Scripts")
sys.path.append(r"C:\Thesis-Project\Scenario2\Smaller_Scale_RAG")
sys.path.append(r"C:\Thesis-Project\Scenario3")

from get_benchmark_data import load_data, get_prompts, get_rows, extract_response
from write_results_in_excel import sanitize_filename, write_data_excel
from evaluate_rag import save_evaluation_results
from rag_hello_world import set_key
from agent_rag import RAG_Agent
from excel_script import find_phrase_excel, read_excel

def __main__():
    """
    Load benchmark data.
    Run the rag agent on top of the benchmark. 
    """
    df = load_data() #Benchmark data from MBPP + 
    model_name = "qwen2.5-coder:32b"
    prompts = get_prompts(df)
    number_prompts = get_rows(df)
    sanitized_name = sanitize_filename(model_name)
    #Write results in txt file.
    filename = f"{sanitized_name}_mbpp_plus_results_scenario4_3000.xlsx"
    #Write results in txt file.
    filename_txt = f"{sanitized_name}_mbpp_plus_results_scenario4_3000.txt"

    empty= [18, 253]

    rag_agent = RAG_Agent()
    queries = []

    for i, prompt in enumerate(tqdm(prompts, desc="Running queries")):
        pre_query = prompt[0] if isinstance(prompt, tuple) else prompt
        if i in empty:
            queries.append(pre_query)
    
    # Adjust the prompt of the agent. 
    agent_prompt_self_evaluator = """Extract, evaluate, and refine the coding snippet from the following problem. 
                                    Use the function name to generate a function that solves the problem. 
                                    You have access to the following tools:
                                    {tools}

                                    Use this format:
                                    Question: the input question you must answer
                                    Thought: you should always think about what to do
                                    Action: the action to take, should be one of [{tool_names}]
                                    Action Input: the input to the action
                                    Observation: the result of the action
                                    ... (this Thought/Action/Action Input/Observation can repeat as needed)
                                    Thought: If the tools do not provide a useful result, generate a code snippet using the language model.
                                    Action: If the tools fail, use the LLM to generate a code snippet.
                                    Action Input: The original input question.
                                    Observation: The generated code snippet from the LLM.
                                    
                                    Thought: I need to verify if this code is correct. Let me analyze it for:
                                    1. Logical errors
                                    2. Edge cases
                                    3. Syntax errors
                                    4. Performance issues
                                    
                                    Action: Self-evaluate the code
                                    Action Input: The generated code snippet
                                    Observation: The self-evaluation results
                                    
                                    Thought: Based on my evaluation, I need to determine if the code needs correction.
                                    Action: If errors are found, correct the code
                                    Action Input: The original code and identified issues
                                    Observation: The corrected code snippet
                                    
                                    Thought: I now have a verified and corrected code snippet
                                    Final Answer: 
                                    - **Code Snippet**: The final, corrected code snippet.
                                    - **Evaluation**: The evaluation process, including:
                                    - Issues identified
                                    - Corrections made
                                    - Verification of correctness
                                    Begin!
                                    Question: {input}
                                    Thought:{agent_scratchpad}"""
    rag_agent.adjust_prompting_template(agent_prompt_self_evaluator)

    # Run the benchmark
    # data = run_benchmark_in_batches(queries, rag_agent, filename=filename)

    # Evaluate data 
    expected_responses = df["code"].to_list()
    test_cases = df["test_list"].to_list()

    #Read Excel and get responses from outputs
    df_excel = pd.read_excel(filename, sheet_name="Sheet1", header=0)

    # Step 2: Filter rows that contain the specific string
    try:
        # has_dupes, duplicate_values = find_duplicates(excel_file_path)
        df =  read_excel(file_path=filename)
        phrase_list = find_phrase_excel(df, "Agent stopped due to the iteration limit or the time limit.")
        if phrase_list:
            print("The phrase exists in the file. The values in the file:", phrase_list)
        else:
            print("The phrase doesn't exists in the file.")
    except ValueError as e:
        print(e)

    model_output = df_excel["output"].apply(func=extract_response)

    # for i, prompt in enumerate(tqdm(prompts, desc="Running queries")):
    #     pre_query = prompt[0] if isinstance(prompt, tuple) else prompt
    #     # if i in rows_to_check:
    #     queries.append(pre_query)

    # data = run_benchmark_in_batches(queries, rag_agent, filename=filename)

    save_evaluation_results(model_responses=model_output,
                            expected_benchmark=expected_responses, 
                            test_cases = test_cases, 
                            filename=filename_txt,
                            check_test_function_name=True)

    print("Finished running the benchmark using agent with RAG (self-evaluating).")

def get_agent_stopped_rows(column_header="row_number"):
    """ 
        Read the Excel file, based on the column header 
        and return the list of the responses.
    """
    #Read Excel and get responses from outputs
    df_excel = pd.read_excel("agent_runtime_output.xlsx", sheet_name="Sheet1", header=0)

    rows = df_excel[column_header].apply(func=extract_response)

    return rows

def run_benchmark_in_batches(queries, rag_agent : RAG_Agent, batch_size=10, model_name = "qwen2.5-coder:32b", filename="qwen2.5-coder32b_mbpp_plus_results.xlsx"):
    # Split data into batches
    batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]

    # Process each batch with progress tracking
    for batch in tqdm(batches, desc = "Batches: "):

        for prompt in batch:
            input_variables = {
                "input": f"{prompt}",
                "tools": "RAG_Search",  # Placeholder text
                "tool_names": "RAG_Search",  # Placeholder text
                "agent_scratchpad": ""  # Optional
            }
            data = rag_agent.agent_executor.invoke(input_variables)
            write_data_excel(data, filename)
    return data

__main__()