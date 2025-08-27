from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings  import HuggingFaceEmbeddings
from langchain_together import Together
from langchain_core.runnables.retry import RunnableRetry
from transformers import AutoModel
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from tqdm import tqdm
import sys
import os
from dotenv import load_dotenv
from datasets import load_from_disk,load_dataset
import torch
import time 
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import requests

dotenv_path = os.path.abspath("C:\Thesis-Project\config.env")
load_dotenv(dotenv_path)

sys.path.append(r"C:\Thesis-Project\WriteResults")
import write_results_txt as txt
import write_results_in_excel as excel_wr

def set_key():
    together_api_key = os.getenv('TOGETHER_API_KEY')
    print(together_api_key)
    os.environ['TOGETHER_API_KEY'] = together_api_key

set_key()

class RAG_Pipeline:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-32B-Instruct", 
                        embedding_model="jinaai/jina-embeddings-v2-base-en", 
                        prompt_template=None):
        """Initialize the RAG benchmark system."""
        
        self.llm = Together(
            model=model_name,
            max_tokens=2048, 
            temperature = 0.3
        )

        # Create a text splitter optimized for code
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=1000,
            chunk_overlap=200
        )

        model_name = embedding_model
        model_kwargs = {'device': 'cpu'} 
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # vector_store = FAISS.load_local(r"C:\Thesis-Project\Scenario2\Smaller_Scale_RAG\faiss_index_1000", 
        #                                 self.embeddings, 
        #                                 allow_dangerous_deserialization=True)
        vector_store1 = FAISS.load_local(
            r"C:\Thesis-Project\Scenario2\Smaller_Scale_RAG\faiss_index_1000", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # Load the second vector store
        vector_store2 = FAISS.load_local(
            r"C:\Thesis-Project\Scenario2\Smaller_Scale_RAG\faiss_index_1001_2000", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # Load third vector store
        vector_store3 = FAISS.load_local(
            r"C:\Thesis-Project\Scenario2\Smaller_Scale_RAG\faiss_index_2001_3000",  # Replace with actual path
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Merge the second vector store into the first
        vector_store1.merge_from(vector_store2)
        vector_store1.merge_from(vector_store3)  # Merge third into first

        # Create a custom prompt template
        base_prompt_rag = """
        You are a coding assistant specialized in generating code solutions in Python. 
        Use the following pieces of context to answer the coding question at the end.
        Do not include explanations before or after the code. Just output the code in Python directly.

        {context}

        Question: {question}
        Answer:
        """
        if prompt_template is None:
            prompt_template = base_prompt_rag
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Create the chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store1.as_retriever(
                search_type = "mmr", 
                search_kwargs={"k": 8, "fetch_k": 10, "lambda_mult": 0.85}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def preprocess_query(self, query):
        prompt = f"""
        You are an expert in reformulating coding queries for retrieval systems.
        Your task is to enhance technical documentation searchability by:
        1. Explicitly specifying that the programming language should be Python, 
        required libraries, and technical constraints
        2. Incorporating API-specific terms
        3. Focusing strictly on implementation details (avoid conceptual questions)

        Original query: {query}

        Rewritten query:
        """
        
        response = self.llm.invoke(prompt)
        return response.strip()
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout))
    )
    def answer_coding_query(self, query):
        # Preprocess the query
        processed_query = self.preprocess_query(query)
        
        # Get the answer
        result = self.chain.invoke({"query": processed_query})
        print(result)
        return result, result["result"], result["source_documents"]
        
        # #retriever = vector_store.as_retriever(search_type = "mmr", search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.85})

    def run_benchmark_with_metrics(self, queries, model_name, filename="qwen2.5-coder32b_mbpp_plus_results_rag_2000_self_evaluation"):
        """Run benchmark and collect performance metrics."""
        results = []
        sources = []
        total_time = 0
        total_tokens = 0
        total_responses = 0
        
        for i, query in enumerate(tqdm(queries, desc="Processing queries")):
            start_time = time.time()
            result, answer, sources_per_query = self.answer_coding_query(query)
            end_time = time.time()
            
            # Calculate metrics for this query
            query_time = end_time - start_time
            # You'll need to implement token counting based on your model
            tokens = len(answer.split())  # Simple approximation
            
            # Add metrics to the result
            result["metrics"] = {
                "time": query_time,
                "tokens": tokens
            }
            
            results.append(answer)
            sources.append([x for x in sources_per_query])
            total_time += query_time
            total_tokens += tokens
            total_responses += 1

            q_data = {
                "time": [query_time],
                "tokens": [tokens],
                "result" : answer if isinstance(answer, str) else list(answer), 
                "sources" : ["; ".join(str(s) for s in sources_per_query)],
                "prompt" : query, 
            }
            # Note: This was commented out for the other scenarios. 
            self.save_metrics(q_data, filename=filename)
        
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
            "Responses": [total_responses],
            "Average tokens": [f"{avg_tokens:.2f}"],
            "Tokens per second": [f"{tokens_per_second:.2f}"],
            # "Responses": results,
            # "Sources": sources
        }
        
        return results, sources, data
    
    def save_metrics(self, data, filename = "qwen2.5-coder32b_mbpp_plus_results"):
        """Save metrics to excel file."""
        # Save data to Excel
        excel_wr.write_data_excel(data, f"{filename}.xlsx")
        print(f"Added prompt in {filename}")