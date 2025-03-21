from evaluators import evaluate_all

def main():
    results = evaluate_all('data/ground_truth.json', 'data/ocr_result.json')
    for doc_id, metrics in results.items():
        print(f"Document ID: {doc_id}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2%}")


if __name__ == "__main__":
    main()
