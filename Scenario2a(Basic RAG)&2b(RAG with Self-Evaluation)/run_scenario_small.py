import os
import sys
import pandas as pd
from tqdm import tqdm
from rag_hello_world import set_key, RAG_Pipeline

# Scenario 1 add on files in system path
sys.path.append(r"C:\Thesis-Project\Scenario1")
sys.path.append(r'C:\Thesis-Project\WriteResults')
sys.path.append(r"C:\Thesis-Project\Evaluation_Scripts")

from get_benchmark_data import load_data, get_prompts, get_rows, extract_response
from write_results_in_excel import sanitize_filename
from evaluate_rag import save_evaluation_results

def __main__():
    """
    Load benchmark data.
    Run the rag pipeline on top of the benchmark. 
    """

    filename = "qwen2.5-coder32b_mbpp_plus_results_rag_3000"
    df = load_data() #Benchmark data from MBPP + 

    prompt_template = """
                        You are a coding assistant specialized in generating code solutions in Python.
                        Use the following pieces of context to answer the coding question at the end.
                        First, generate the Python code that best solves the question.
                        After retrieving relevant context and generating your Python code solution, 
                        perform a self-evaluation considering correctness, efficiency, readability, 
                        and adherence to best practices.
                        If your code fully meets these criteria, return only the final, correct Python code 
                        snippet-no explanations or extra text.
                        Do not include explanations before the code.
                        If the code could be improved, revise it accordingly and return only the improved Python code snippet.
                        Do not include any explanations, comments, or self-evaluation in your output-only the final code.
                        Output format:

                        Python code only

                        {context}

                        Question: {question}
                        Answer:
                      """

    rag_pipeline = RAG_Pipeline() 
    prepreccessed_queries = []

    model_name = "qwen2.5-coder:32b"
    prompts = get_prompts(df)
    number_prompts = get_rows(df)

    # for i, prompt in enumerate(tqdm(prompts, desc="Running queries")):
    #     pre_query = prompt[0] if isinstance(prompt, tuple) else prompt
    #     preprocessed_response = rag_pipeline.preprocess_query(pre_query)
    #     print(preprocessed_response)
    #     prepreccessed_queries.append(preprocessed_response)
    #     print(f"Finsihed running prompt {i} out of {number_prompts}")
    # # Run the benchmark
    # # results, sources, data = rag_pipeline.run_benchmark_with_metrics(prepreccessed_queries, model_name)
    # data = run_benchmark_in_batches(prepreccessed_queries, rag_pipeline, filename)
    # rag_pipeline.save_metrics(data, filename)

    # Evaluate data 
    expected_responses = df["code"].to_list()
    test_cases = df["test_list"].to_list()

    #Read Excel and get responses from outputs -Add to name for later: self_evaluation
    df_excel = pd.read_excel(f"{filename}.xlsx", sheet_name="Sheet1", header=0)

    model_output = df_excel["result"].apply(func=extract_response)

    #Write results in txt file. self-evaluation
    filename = f"{filename}.txt"
    save_evaluation_results(model_responses=model_output,expected_benchmark=expected_responses, test_cases = test_cases, filename=filename, check_test_function_name=True)

    print("Finished running the benchmark using RAG.")


def run_benchmark_in_batches(queries, rag_pipeline : RAG_Pipeline, filename, batch_size=10, model_name = "qwen2.5-coder:32b"):
    # Split data into batches
    batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]

    # Process each batch with progress tracking
    for batch in tqdm(batches, desc = "Batches: "):
        batch_result, sources, data = rag_pipeline.run_benchmark_with_metrics(batch, model_name)
        rag_pipeline.save_metrics(data, filename) 
    return data

__main__()