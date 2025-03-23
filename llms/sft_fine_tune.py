import os
import glob
import json
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from langchain_openai import ChatOpenAI

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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


DATA_PATH = "../data/jfk_text/"

def load_text_files(directory):
    texts = []
    for file_path in glob.glob(os.path.join(directory, "**/*.md"), recursive=True):
        print(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts

texts = load_text_files(DATA_PATH)
print("Finished loading documents, total:", len(texts))

data = {"text": texts}
dataset = Dataset.from_dict(data)
print(dataset)
dataset = dataset.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./supervised_finetune_output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=2,
    fp16=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)


print("Starting supervised fine-tuning...")
trainer.train()
trainer.save_model("./supervised_finetuned_deepseek")
print("Supervised fine-tuning completed and model saved.")


qa_file = "../data/jfk_qa.json"
with open(qa_file, "r", encoding="utf-8") as f:
    qa_samples = json.load(f)

results = []
total_score = 0

for sample in qa_samples:
    query = sample["query"]
    expected = sample.get("expected_answer", "")
    
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256)
    qa_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    score = evaluate_answer(query, expected, qa_answer)
    results.append({
        "query": query,
        "expected_answer": expected,
        "qa_answer": qa_answer,
        "score": score
    })
    total_score += score
    print(f"Query: {query}")
    print(f"Expected: {expected}")
    print(f"Answer: {qa_answer}")
    print(f"Score: {score}")
    print("-" * 50)

average_score = total_score / len(qa_samples)
print("Final Average Score:", average_score)

with open("supervised_benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)