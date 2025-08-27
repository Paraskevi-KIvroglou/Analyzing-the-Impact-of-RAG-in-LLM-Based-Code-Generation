# pylint: disable=trailing-whitespace
# pylint: skip-file
# The path to the directory containing your write_results_in_excel.py file is added to the Python path.
import asyncio
import sys
import os
import time 
import ollama 
from langchain_together import Together
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import requests
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import pandas as pd

sys.path.append(r'C:\Thesis-Project\WriteResults')
sys.path.append(os.path.abspath('C:\Thesis-Project\Evaluation_Scripts'))

#Load environment variables
load_dotenv("..\Thesis-Project\ollama_config.env")
dotenv_path = os.path.abspath("C:\Thesis-Project\config.env")
load_dotenv(dotenv_path)

#Custom modules
import get_benchmark_data as benchData
import ollama_helper as ollamaHelper
import write_results_in_excel as write_results_in_excel
import evaluate 

def set_key():
    together_api_key = os.getenv('TOGETHER_API_KEY')
    print(together_api_key)
    os.environ['TOGETHER_API_KEY'] = together_api_key

set_key()

def get_ollama_client():
    """ Get the ollama client url. """
    client = ollama.Client(host=ollamaHelper.get_ollama_host())
    return client

def run_benchmark(model_name, prompt, iterations = 1):
    """ Run the benchmark with the model from Ollama clients. Save the information of the avg time and response."""
    model = get_ollama_client()
    total_time = 0
    total_tokens = 0
    total_responses = []

    for _ in range(iterations):
        start_time = time.time()
        response = model.generate(model=model_name, prompt=prompt)
        end_time = time.time()

        response_msg = response['response']
        total_responses.append(response_msg)

        total_time += end_time - start_time
        #eval_count: Tokens used for output, prompt_eval_count: tokens of the input prompt
        total_tokens += response['eval_count'] + response['prompt_eval_count'] #Ollama client response version: 0.5.4

    avg_time = total_time / iterations
    avg_tokens = total_tokens / iterations
    tokens_per_second = avg_tokens / avg_time
    
    print(f"Model: {model_name}")
    print(f"Average time: {avg_time:.2f} seconds")
    print(f"Average tokens: {avg_tokens:.2f}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Total responses: {total_responses}")

    data = {
        "Model Name" : [model_name], 
        "Average Time (sec)" : [f"{avg_time:.2f}"],
        "Responses" : [total_responses],
        "Average tokens" : [f"{avg_tokens:.2f}"],
        "Tokens per second" : [f"{tokens_per_second:.2f}"]
    }
    return data

@retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout))
    )
def answer_coding_query(llm, query):
    # Prompt template 
    prompt_template = PromptTemplate.from_template("{query}")
    #Format prompt
    formatted_prompt = prompt_template.format(query=query)
    # Get the answer
    result = llm.invoke(formatted_prompt)
    print(result)
    return result

def run_benchmark_API(prompt, model_name="Qwen/Qwen2.5-Coder-32B-Instruct", iterations=1):
    """Run benchmark and collect performance metrics using the Together API."""
    llm = Together(
            model=model_name,
            max_tokens=2048, 
            temperature = 0.3, 
    )

    results = []
    sources = []
    total_time = 0
    total_tokens = 0
    total_responses = 0
    
    print(f"Processing query: {prompt}")
    start_time = time.time()
    answer = answer_coding_query(query=prompt, llm=llm)
    end_time = time.time()
        
    # Calculate metrics for this query
    query_time = end_time - start_time
    # You'll need to implement token counting based on your model
    tokens = len(answer.split())  # Simple approximation
        
    results.append(answer)
    total_time += query_time
    total_tokens += tokens
    total_responses += 1

    q_data = {
        "time": [query_time],
        "tokens": [tokens],
        "result" : [answer], 
    }
    # Note: This was commented out for the other scenarios. 
    # self.save_metrics(q_data)
    
    # Calculate averages
    avg_time = total_time / total_responses if total_responses > 0 else 0
    avg_tokens = total_tokens / total_responses if total_responses > 0 else 0
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    # Print summary
    print(f"Model: {model_name}")
    print(f"Average time: {avg_time:.2f} seconds")
    print(f"Average tokens: {avg_tokens:.2f}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Total responses: {total_responses}")
    
    # Create data dictionary for saving
    data = {
        "Model Name": [model_name],
        "Average Time (sec)": [f"{avg_time:.2f}"],
        "Result" : answer, 
        "Responses": [total_responses],
        "Average tokens": [f"{avg_tokens:.2f}"],
        "Tokens per second": [f"{tokens_per_second:.2f}"],
        "prompt" : f"{prompt}",
        # "Responses": results,
        # "Sources": sources
    }
    
    return data


def run_model_over_benchmark(model_name, prompts, number_prompts, system_prompt="You are a helpful coding assistant. Please generate code in Python for the following request "):
    """ Run model over benchmark prompts. """
    #Loop over the prompts and write the benchmark 
    sanitized_name = write_results_in_excel.sanitize_filename(model_name)
    filename = f"{sanitized_name}_mbpp_plus_results_a2_evaluation_prompt.xlsx"

    prompts_to_check = ["Function Name: count"]

    for i, row in enumerate(prompts):

        print(f"Running prompt {i} out of {number_prompts}")
        prompt = row[0] if isinstance(row, tuple) else row
        # for function in prompts_to_check:
        #     if function in prompt:
        prompt = f"{system_prompt} : {prompt}"
        data = run_benchmark_API(prompt=prompt)
        print(f"Finsihed running prompt {i} out of {number_prompts}")

        write_results_in_excel.write_data_excel(data, filename)
    
    print("Running benchmark finished.")

def evaluate_results(df, model_name):
    """ Evaluate the results from the model and save them in an excel file. """
    expected_responses = df["code"].to_list()
    test_cases = df["test_list"].to_list()

    #Read Excel and get responses from outputs
    df_excel = pd.read_excel("qwen2.5-coder32b_mbpp_plus_results_a2_evaluation_prompt.xlsx", sheet_name="Sheet1", header=0)

    model_output = df_excel["Result"].apply(func=benchData.extract_response)

    sanitized_name = write_results_in_excel.sanitize_filename(model_name)
    #Write results in txt file.
    filename = f"{sanitized_name}_mbpp_plus_results_a2_evaluation_prompt.txt"
    evaluate.save_evaluation_results(model_responses=model_output,
                                     expected_benchmark=expected_responses, 
                                     test_cases = test_cases,
                                     filename=filename,
                                     check_test_function_name = True)

def __main__():
    """ Run the get rows, evaluate results."""
    #Load data 
    df = benchData.load_data()
    prompts = benchData.get_prompts(df=df)
    number_prompts =  benchData.get_rows(df)

    #Load - Pull Model from Ollama TODO: Async is not running properly to download the model. Not fast enough connection to ngrok fails.
    #asyncio.run(asyncCall.pull_model_ollama())

    #Call ollama list from ollama client and print the downloaded models in the current run.
    #ollamaHelper.get_ollama_list()

    #Make sure filename is appropriate for operating systems, not contain invalid characters
    model_name = "qwen2.5-coder:32b"

    system_evaluation_prompt = """
                            You are a helpful coding assistant. Please generate code in Python for the following request:

                            {query}

                            Please follow this structured process:

                            STEP 1: Write an initial Python solution for the request above.

                            STEP 2: Evaluate your solution using the following criteria:
                            1. Logical errors
                            2. Edge cases
                            3. Syntax errors
                            4. Performance issues

                            STEP 3: Based on your evaluation, either:
                            a) Confirm that your initial solution is optimal, or
                            b) Provide an improved version with explanations of what you changed and why

                            STEP 4: Summarize the strengths and limitations of your final solution.

                            STEP 5: Based on the previous steps, please provide the best coding solution for the 
                            user's request. 
                            """

    #Run the model on top of benchmark
    # run_model_over_benchmark(model_name= model_name, prompts=prompts, number_prompts= number_prompts, system_prompt=system_evaluation_prompt)

    #Read results and write pass@1 and CodeBleu in txt file. 
    evaluate_results(df, model_name)

__main__()
