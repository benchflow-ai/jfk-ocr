import os
import time
import json
import re

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_PATH = "./data/jfk_text/"
loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()
print("Finished loading documents, total:", len(documents))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY),
    retriever=retriever
)

evaluator = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

def evaluate_answer(query, expected, qa_answer):
    eval_prompt = f"""
You are an expert evaluator. Please score the following QA result on a scale from 1 to 10, where 10 means the answer is perfect and 1 means it is completely incorrect.

Question: {query}
Expected Answer: {expected}
QA System's Answer: {qa_answer}

Please provide only the final score as an integer between 1 and 10.
    """
    score_response = evaluator.invoke(eval_prompt)
    
    if hasattr(score_response, "content"):
        score_text = score_response.content
    else:
        score_text = str(score_response)
    match = re.search(r"\d+", score_text)
    if match:
        score = int(match.group())
    else:
        score = 0
    return score

benchmark_file = "data/jfk_qa.json"
with open(benchmark_file, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

results = []
total_score = 0

for sample in benchmark_data:
    query = sample["query"]
    expected = sample.get("expected_answer", "")
    
    start_time = time.time()
    qa_answer = qa_chain.invoke(query)
    end_time = time.time()
    response_time = end_time - start_time
    
    score = evaluate_answer(query, expected, qa_answer)
    
    result = {
        "query": query,
        "expected_answer": expected,
        "qa_answer": qa_answer,
        "response_time": response_time,
        "score": score
    }
    results.append(result)
    total_score += score
    
    print(f"Query: {query}")
    print(f"Expected Answer: {expected}")
    print(f"QA System's Answer: {qa_answer}")
    print(f"Score: {score}")
    print("-" * 50)

average_score = total_score / len(benchmark_data)
print("Final Average Score:", average_score)

with open("benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
