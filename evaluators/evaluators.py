import json
from evaluators.sentence_alignment import align_sentences
from rapidfuzz import fuzz
import numpy as np
from collections import Counter

# 1. OCR Accuracy (Sentence-level Edit Distance)
def sentence_edit_distance(gt_sentence, pred_sentence):
    return 1 - (fuzz.ratio(gt_sentence, pred_sentence) / 100.0)

def evaluate_ocr_accuracy(gt_sentences, pred_sentences):
    aligned_pairs = align_sentences(gt_sentences, pred_sentences)
    total_distance = sum(sentence_edit_distance(gt, pred) for gt, pred in aligned_pairs)
    avg_distance = total_distance / len(aligned_pairs)
    return avg_distance

# 2. Sentence-grams and flag % differences
def evaluate_sentence_gram_difference(gt_sentences, pred_sentences, n=2):
    def get_ngrams(sentences, n):
        ngrams = set()
        for sentence in sentences:
            tokens = sentence.split()
            ngrams.update(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        return ngrams

    gt_ngrams = get_ngrams(gt_sentences, n)
    pred_ngrams = get_ngrams(pred_sentences, n)
    intersection = gt_ngrams & pred_ngrams
    difference_ratio = 1 - (len(intersection) / max(len(gt_ngrams), 1))
    return difference_ratio

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
            # 'reading_order_accuracy': evaluate_reading_order_accuracy(gt_sentences, pred_sentences),
            # 'hallucination_rate': evaluate_hallucination_rate(gt_sentences, pred_sentences),
            # 'dropped_content': evaluate_dropped_content(gt_sentences, pred_sentences),
            # 'determinism': evaluate_determinism([pred_sentences]*10)  # Placeholder: 应传入多次运行的真实预测结果
        }
    return results
