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

class RAG_Agent:
    def __init__(self):
        self.rag_pipeline = RAG_Pipeline()

        self.rag_tool = Tool(
            name="RAG_Search",
            func=self.rag_search,
            description="Useful for answering questions about specific documents or knowledge bases."
            "Input should be a question or a coding query."
        )

        # Required template structure for React agents
        # Define the prompt template
        self.react_template = """Find or implement the coding snippet in Python only from the following problem. You have access to the 
                            following tools:
                            {tools}

                            Use this format:
                            Question: the input question you must answer
                            Thought: you should always think about what to do
                            Action: the action to take, should be one of [{tool_names}]
                            Action Input: the input to the action
                            Observation: the result of the action
                            ... (this Thought/Action/Action Input/Observation can repeat 1 time)
                            Thought: If the tools do not provide a useful result, generate a code snippet in Python using the language model.
                            Action: If the tools fail, use the LLM to generate a code snippet in Python.
                            Action Input: The original input question.
                            Observation: The generated code snippet from the LLM  in Python.
                            Thought: I now know the final code snippet
                            Final Answer: the final code snippet in Python
                            Begin!
                            Question: {input}
                            Thought:{agent_scratchpad}"""
        
        # Create the prompt template
        prompt_template = PromptTemplate.from_template(self.react_template)

        # Create the agent
        agent = create_react_agent(
            llm=self.rag_pipeline.llm,
            tools=[self.rag_tool],
            prompt=prompt_template  # Pass PromptTemplate object
        )

        # Create an agent executor
        self.agent_executor = AgentExecutor(agent=agent, 
                                tools=[self.rag_tool],
                                handle_parsing_errors=True,
                                max_execution_time=500, # Stop after x ammount of seconds
                                verbose=True) 

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=15),
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