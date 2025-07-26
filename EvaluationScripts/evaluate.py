from pass1_metric import calculate_pass_at_1
from code_bleu_metric import code_bleu
# from codebleu import calc_codebleu
# from syntax_match import calc_syntax_match

# def get_codebleu_score(references, candidates):
#     weights = (0.25, 0.25, 0.25, 0.25)  # Default weights
#     score = calc_codebleu(references, candidates, lang="python", weights=weights)
#     return score

def evaluate(model_responses, expected_benchmark, test_cases, check_test_function_name):

    pass1_metric, correct_count, total_count, pass_k1_research_estimator = calculate_pass_at_1(model_responses=model_responses, 
                                                                                               test_cases=test_cases,
                                                                                               check_test_function_name=check_test_function_name)

    # Calculate CodeBLEU
    #codebleu_score = calculate_codebleu(expected_benchmark, model_responses)
    #print(f"CodeBLEU score: {codebleu_score}")

    # Calculate CodeBLEU
    # codebleu_score = get_codebleu_score(expected_benchmark, model_responses)

    # syntax_match_score = calc_syntax_match(references=expected_benchmark, candidate=model_responses)

    dict_scores = {
        "Pass@1" : f"{pass1_metric:.3f}",
        "Pass@1 (%)" : f"{pass1_metric*100:.3f}%",
        "Correct count" : f"{correct_count}",
        "Total count" : f"{total_count}",
        "Pass@1 Research Estimator" : f"{pass_k1_research_estimator:.3f}",
        "Pass@1 Research Estimator (%)" : f"{pass_k1_research_estimator*100:.3f}%",
        # "Overall CodeBLEU" : f"{codebleu_score['codebleu']*100:.2f}",
        # "BLEU" : f"{codebleu_score['bleu']*100:.2f}",
        # "Weighted BLEU" : f"{codebleu_score['weighted_ngram_match']*100:.2f}",
        #"AST Match" : f"{syntax_match_score *100:.2f}",
        # "Data Flow Match" : f"{codebleu_score['dataflow_match']*100:.2f}"
    }

    # Print all components of the CodeBLEU score
    # print("CodeBLEU Score Components:")
    # for key, value in codebleu_score.items():
    #     print(f"{key}: {value*100:.2f}%")
    #     dict_scores[key] = f"{value*100:.2f}%"
    
    return dict_scores

def write_results_to_txt(filename, dict_scores):
    with open(filename, 'w') as file:
        for key, value in dict_scores.items():

            file.write(f"{key} Score: {value}\n")

def save_evaluation_results(model_responses, expected_benchmark, test_cases, filename, check_test_function_name=False):
    results = evaluate(model_responses=model_responses, expected_benchmark=expected_benchmark, test_cases = test_cases, check_test_function_name=check_test_function_name)
    write_results_to_txt(filename, results)