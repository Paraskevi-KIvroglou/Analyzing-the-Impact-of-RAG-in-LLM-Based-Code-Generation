# pylint: disable=import-error
from get_benchmark_data import extract_function_name_from_code, load_data, get_prompts

def test_extract_function_name_from_code():
    """ This test is supposed to check the functionallity of extracting the name of the function from the code column of the dataset."""

    example1 = "def similar_elements(test_tup1, test_tup2): return tuple(set(test_tup1) & set(test_tup2))"

    function_name = extract_function_name_from_code(example1)

    assert function_name == "similar_elements"

def test_extract_function_name_from_code_2():
    """ This test is supposed to check the functionallity of extracting the name of the function from the code column of the dataset."""

    example1 = "import heapq as hq def heap_queue_largest(nums: list,n: int) -> list: largest_nums = hq.nlargest(n, nums) return largest_nums"

    function_name = extract_function_name_from_code(example1)

    assert function_name == "heap_queue_largest"

def test_combined_prompts():
    """ Test if the combined list contains both function names and prompts."""

    df = load_data()
    combined_promts = get_prompts(df)
    print(combined_promts[0])

    assert len(combined_promts) > 0
    