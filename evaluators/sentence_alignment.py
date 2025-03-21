from rapidfuzz import fuzz
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import os

def align_sentences(gt_sentences, pred_sentences):
    n_gt, n_pred = len(gt_sentences), len(pred_sentences)
    cost_matrix = np.zeros((n_gt, n_pred))

    for i, gt_sentence in enumerate(gt_sentences):
        for j, pred_sentence in enumerate(pred_sentences):
            similarity = fuzz.ratio(gt_sentence, pred_sentence)
            cost_matrix[i, j] = -similarity

    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    aligned = []
    matched_pred = set()
    for i in range(n_gt):
        if i in gt_indices:
            pred_idx = pred_indices[np.where(gt_indices == i)[0][0]]
            if pred_idx not in matched_pred:
                aligned.append((gt_sentences[i], pred_sentences[pred_idx]))
                matched_pred.add(pred_idx)
            else:
                aligned.append((gt_sentences[i], None))
        else:
            aligned.append((gt_sentences[i], None))
    
    os.makedirs('results', exist_ok=True)
    with open('results/aligned.json', 'w') as f:
        json.dump(aligned, f)
    return aligned
