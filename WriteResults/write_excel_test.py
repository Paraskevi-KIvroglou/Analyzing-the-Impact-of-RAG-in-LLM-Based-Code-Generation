import os
import pandas as pd
import write_results_in_excel

def test_write_data_excel():
    test_filename = "test_results.xlsx"

    # Clean up any existing test file
    if os.path.exists(test_filename):
        os.remove(test_filename)

    # First write
    write_results_in_excel.write_data_excel({"col1": 1, "col2": 2}, test_filename)

    # Second write
    write_results_in_excel.write_data_excel({"col1": 3, "col2": 4}, test_filename)

    # Read back the file to verify
    df = pd.read_excel(test_filename, engine="openpyxl")

    # Clean up test file
    os.remove(test_filename)

    # Check if data was appended correctly
    expected_data = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
    assert df.equals(expected_data), f"Test failed! Data in file: {df}"
    print("Test passed! Data appended correctly.")

# Run the test
test_write_data_excel()
