import os
import sys
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

dotenv_path = os.path.abspath("C:\Thesis-Project\config.env")
load_dotenv(dotenv_path)
sys.path.append(r"C:\Thesis-Project\Scenario2\Smaller_Scale_RAG")
sys.path.append(r"C:\Thesis-Project\Scenario2\Smaller_Scale_RAG\faiss_index_1000")
sys.path.append(r"C:\Thesis-Project\WriteResults")
from rag_hello_world import set_key, RAG_Pipeline
import tenacity

def evaluate_code(code: str, test_cases: list) -> str:
    """
    Evaluates Python code for correctness, efficiency, and best practices.

    This is a sophisticated function that evaluates code. Afterwards
    provides feedback regarding the coding output and useful 
    implementation improvements. 
    
    Args:
        code (str): The Python code to evaluate
        
    Returns:
        str: Evaluation results with suggestions for improvement
    """
    report = {
            "passed_tests": 0,
            "failed_tests": [],
            "security_violations": [],
            "quality_issues": []
        }

    try:
        # Check for syntax errors
        compile(code, '<string>', 'exec')
        
        # Analyze code structure and patterns
        evaluation_results = []
        
        # Check for basic code quality issues
        if "import *" in code:
            evaluation_results.append("- Avoid using 'import *' as it can lead to namespace pollution")
            
        # Check for proper error handling
        if "try:" in code and "except:" in code and "except Exception:" not in code:
            evaluation_results.append("- Consider catching specific exceptions rather than using bare except")
            
        # Check for comments and docstrings
        if not any(line.strip().startswith('#') for line in code.split('\n')) and '"""' not in code:
            evaluation_results.append("- Add comments or docstrings to improve code readability")
        
        # Add more sophisticated checks as needed
        if evaluation_results:
            report["quality_issues"].append("Code evaluation completed with the following suggestions:\n" + "\n".join(evaluation_results))
        else:
            report["quality_issues"].append("Code looks good! No immediate issues found.")
        
        sandbox = {"__builtins__": {}}
    
        try:
            # Execute code in sandbox
            exec(code, sandbox)
            
            # Validate test cases
            for idx, test in enumerate(test_cases, 1):
                try:
                    exec(test, {"__builtins__": {}, **sandbox})
                    report["passed_tests"] += 1
                except AssertionError:
                    report["failed_tests"].append(f"Test {idx}: Assertion failed")
                except Exception as e:
                    report["failed_tests"].append(f"Test {idx}: {str(e)}")
        except SyntaxError as e:
            report["execution_error"] = f"Syntax error in the code: {str(e)}"
        except Exception as e:
            report["execution_error"] = f"Error evaluating code: {str(e)}"
        finally:
            # Add quality checks
            if "import *" in code:
                report["quality_issues"].append("Wildcard import detected")

            return report
    except Exception as e:
        return f"Error evaluating code: {str(e)}"

class RAG_Evaluator_Agent:
    def __init__(self):
        self.rag_pipeline = RAG_Pipeline()

        self.rag_tool = Tool(
                        name="RAG_Search",
                        func=self.rag_search,
                        description="Retrieves relevant code snippets from knowledge base"
                    )

        self.code_evaluation_tool = Tool(
            name="Code_Evaluator",
            func=evaluate_code,
            description="Validates code for correctness, style, and security"
        )

        # Updated prompt template with proper variable placement
        self.react_template = """You are an expert Python developer. Use these tools:
        {tools}

        **Process**
        1. Retrieve code with RAG_Search
        2. Generate new code if needed
        3. Evaluate with Code_Evaluator
        4. Improve iteratively (max 3 cycles)

        **Format**
        Question: {input}
        Thought: [Analysis]
        Action: {tool_names}
        Action Input: [Parameters]
        Observation: [Result]
        ... (Repeat)
        Final Answer: ``````

        Begin!
        Question: {input}
        Thought:{agent_scratchpad}"""

        # Create prompt template with correct variables
        prompt_template = PromptTemplate(
            template=self.react_template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )

        # Create agent with proper tool configuration
        agent = create_react_agent(
            llm=self.rag_pipeline.llm,
            tools=[self.rag_tool, self.code_evaluation_tool],
            prompt=prompt_template
        )

        # Create an agent executor
        self.agent_executor = AgentExecutor(agent=agent, 
                                tools=[self.rag_tool],
                                handle_parsing_errors=True,
                                max_execution_time=500, # Stop after x ammount of seconds
                                verbose=True) 

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=25),
        reraise=True
    )
    def rag_search_with_retry(self, query):
        """
        This function is searching and returning an answer from the RAG Pipeline class
        with a retry mechanism using Tenacity.
        """
        preprocessed_query = self.rag_pipeline.preprocess_query(query)
        result, answer, sources_per_query = self.rag_pipeline.answer_coding_query(preprocessed_query)
        return answer

    def rag_search(self, query):
        """
            Retry Rag Search function used for the main function of the tool

            (Returns): str -> answer
        """
        return self.rag_search_with_retry(query)
    
    def adjust_prompting_template(self, new_prompt):
        self.react_template = new_prompt
    
    # Define the RAG tool
    # def rag_search(self, query):
    #     """
    #     This function is searching and returning an answer from the RAG Pipeline class
    #     """
    #     preprocessed_query = self.rag_pipeline.preprocess_query(query)
    #     result, answer, sources_per_query = self.rag_pipeline.answer_coding_query(preprocessed_query)
    #     return answer

# Define variables for the prompt
input_variables = {
    "input": "Write a function to find the shared elements from the given two lists",
    "tools": "RAG Search",  # Placeholder text
    "tool_names": "RAG Search",  # Placeholder text
    "agent_scratchpad": ""  # Optional
}
# Invoke the agent with formatted prompt
# agent = RAG_Agent()
# result = agent.agent_executor.invoke(input_variables)
# print(result)
# # Use the Agent
# response = agent_executor.invoke({"input": "Write a function to find the shared elements from the given two lists."})
# print(response["output"])