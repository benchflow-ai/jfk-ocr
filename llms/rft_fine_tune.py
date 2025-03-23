import os
import json
import re
import time
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from langchain_community.chat_models import ChatOpenAI

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
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

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
ref_model = create_reference_model(model)

qa_file = "data/jfk_qa.json"
with open(qa_file, "r", encoding="utf-8") as f:
    qa_samples = json.load(f)

prompts = [sample["query"] for sample in qa_samples]
expected_answers = [sample.get("expected_answer", "") for sample in qa_samples]

def compute_rewards(batch_prompts, batch_generated, batch_expected):
    rewards = []
    for query, generated, expected in zip(batch_prompts, batch_generated, batch_expected):
        reward = evaluate_answer(query, expected, generated)
        rewards.append(reward)
    return rewards

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    ppo_epochs=4,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

num_epochs = 3

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    for i in range(0, len(prompts), ppo_config.batch_size):
        batch_prompts = prompts[i:i+ppo_config.batch_size]
        batch_expected = expected_answers[i:i+ppo_config.batch_size]
        
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**batch_inputs, max_length=256)
        batch_generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        rewards = compute_rewards(batch_prompts, batch_generated, batch_expected)
        
        stats = ppo_trainer.step(batch_prompts, batch_generated, rewards)
        print(f"Batch {(i // ppo_config.batch_size)+1}: Rewards: {rewards}")
    print(f"Epoch {epoch+1} completed.")

model.save_pretrained("./ppo_finetuned_deepseek")
print("Reinforcement fine-tuning completed and model saved.")