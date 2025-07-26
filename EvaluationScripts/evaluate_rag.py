from pass_1_metric_rag import calculate_pass_at_1_rag
from code_bleu_metric import code_bleu


def evaluate(model_responses, expected_benchmark, test_cases, check_test_function_name=False, remove_whitespace=False):
    """
    Evaluate the model responses against the test cases.
    """
    pass1_metric, correct_count, total_count, pass_k1_research_estimator = calculate_pass_at_1_rag(model_responses=model_responses, 
                                                                                                   test_cases=test_cases, 
                                                                                                   check_function_name_tests=check_test_function_name,
                                                                                                   remove_whitespace=remove_whitespace)

    dict_scores = {
        "Pass@1" : f"{pass1_metric:.3f}",
        "Pass@1 (%)" : f"{pass1_metric*100:.3f}%",
        "Correct count" : f"{correct_count}",
        "Total count" : f"{total_count}",
        "Pass@1 Research Estimator" : f"{pass_k1_research_estimator:.3f}",
        "Pass@1 Research Estimator (%)" : f"{pass_k1_research_estimator*100:.3f}%",
    }
    
    return dict_scores

def write_results_to_txt(filename, dict_scores):
    """
    Write the results to a txt file from dictionary scores. 
    """
    with open(filename, 'w') as file:
        for key, value in dict_scores.items():

            file.write(f"{key} Score: {value}\n")

def save_evaluation_results(model_responses, expected_benchmark, test_cases, filename, check_test_function_name=False, remove_whitespace=False):
    """
    Save the evaluation results in a txt file. 
    """
    results = evaluate(model_responses=model_responses, expected_benchmark=expected_benchmark, 
                       test_cases = test_cases, check_test_function_name=check_test_function_name,
                       remove_whitespace=remove_whitespace)
    write_results_to_txt(filename, results)