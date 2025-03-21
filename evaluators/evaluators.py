import json
from evaluators.sentence_alignment import align_sentences
from rapidfuzz import fuzz
from nltk.tokenize import word_tokenize
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer

# 1. OCR Accuracy (Sentence-level Edit Distance)
def sentence_edit_distance(gt_sentence, pred_sentence):
    return 1 - (fuzz.ratio(gt_sentence, pred_sentence) / 100.0)

def evaluate_ocr_accuracy(gt_sentences, pred_sentences):
    start_time = time.time()
    aligned_pairs = align_sentences(gt_sentences, pred_sentences)
    total_distance = sum(sentence_edit_distance(gt, pred) for gt, pred in aligned_pairs)
    avg_distance = total_distance / len(aligned_pairs)
    end_time = time.time()
    print(f"ocr accuracy time taken: {end_time - start_time:.4f} seconds")
    return avg_distance

# 2. Sentence-grams and flag % differences
def evaluate_sentence_gram_difference(gt_sentences, pred_sentences, n=2):
    start_time = time.time()
    aligned_sentences = align_sentences(gt_sentences, pred_sentences)

    def get_ngrams(sentences, n):
        ngrams = set()
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            ngrams.update(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        return ngrams
    gt_aligned_sentences = [gt for gt, _ in aligned_sentences if gt]
    pred_aligned_sentences = [pred for _, pred in aligned_sentences if pred]
    gt_ngrams = get_ngrams(gt_aligned_sentences, n)
    pred_ngrams = get_ngrams(pred_aligned_sentences, n)
    intersection = gt_ngrams & pred_ngrams
    difference_ratio = 1 - (len(intersection) / max(len(gt_ngrams), 1))
    end_time = time.time()
    print(f"sentence gram difference (n={n}) time taken: {end_time - start_time:.4f} seconds")

    return difference_ratio

# 3. Reading order accuracy
def evaluate_reading_order_accuracy(gt_sentences, pred_sentences, n=2):
    start_time = time.time()
    # Note: nltk.word_tokenize will split the punctuation into two tokens, it will affect the accuracy
    vectorizer = CountVectorizer(analyzer='word', tokenizer=word_tokenize, token_pattern=None, ngram_range=(n, n), binary=True)
    all_sentences = gt_sentences + pred_sentences
    vectorizer.fit(all_sentences)

    gt_vec = vectorizer.transform(gt_sentences)
    pred_vec = vectorizer.transform(pred_sentences)

    gt_sum = np.array(gt_vec.sum(axis=0)).flatten()
    pred_sum = np.array(pred_vec.sum(axis=0)).flatten()

    correct_order = np.sum((gt_sum > 0) & (pred_sum > 0))
    total_ngrams = np.sum(gt_sum > 0)

    accuracy = correct_order / max(total_ngrams, 1)
    end_time = time.time()
    print(f"reading order accuracy time taken: {end_time - start_time:.4f} seconds")
    return accuracy

def evaluate_hallucination_rate(gt_sentences, pred_sentences):
    start_time = time.time()
    aligned_pairs = align_sentences(gt_sentences, pred_sentences)
    
    total_pred_words = 0
    total_hallucinated = 0
    
    for gt, pred in aligned_pairs:
        if pred is None:
            continue
        gt_tokens = set(word_tokenize(gt))
        pred_tokens = word_tokenize(pred)
        
        total_pred_words += len(pred_tokens)
        hallucinated = sum(1 for word in pred_tokens if word not in gt_tokens)
        total_hallucinated += hallucinated
    
    hallucination_ratio = total_hallucinated / max(total_pred_words, 1)
    end_time = time.time()
    print(f"hallucination rate time taken: {end_time - start_time:.4f} seconds")
    return hallucination_ratio

def evaluate_all(gt_file, pred_file):
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    results = {}
    for gt_doc in gt_data:
        doc_id = gt_doc['id']
        gt_sentences = gt_doc['sentences']
        pred_sentences = next(doc['sentences'] for doc in pred_data if doc['id'] == doc_id)
        
        results[doc_id] = {
            'ocr_accuracy': evaluate_ocr_accuracy(gt_sentences, pred_sentences),
            'sentence_gram_difference': evaluate_sentence_gram_difference(gt_sentences, pred_sentences),
            # 'table_extraction_accuracy': evaluate_table_extraction_accuracy([], []),  # Placeholder
            # 'llm_extraction_accuracy': evaluate_llm_extraction_accuracy({}, {}),  # Placeholder
            # 'checkbox_accuracy': evaluate_checkbox_accuracy([], []),  # Placeholder
            'reading_order_accuracy': evaluate_reading_order_accuracy(gt_sentences, pred_sentences),
            'hallucination_rate': evaluate_hallucination_rate(gt_sentences, pred_sentences),
            # 'dropped_content': evaluate_dropped_content(gt_sentences, pred_sentences),
            # 'determinism': evaluate_determinism([pred_sentences]*10)  # Placeholder: 应传入多次运行的真实预测结果
        }
    return results
