from reducto import Reducto
from pathlib import Path
from nltk.tokenize import sent_tokenize
import json
import os
import re

client = Reducto(api_key=os.getenv("REDUCTO_API_KEY"))

def extract_contents(result):
    contents = []
    for chunk in result.chunks:
        for block in chunk.blocks:
            if block.type.lower() == "table":
                continue
            contents.append(block.content)
    return contents

def clean_sentence(sentence):
    return re.sub(r'\s+', ' ', sentence).strip()

def process_pdf(file_path):
    upload = client.upload(file=Path(file_path))
    result_data = client.parse.run(document_url=upload)
    all_contents = extract_contents(result_data.result)

    return {
        "id": Path(file_path).stem,
        "sentences": [clean_sentence(sentence) for content in all_contents for sentence in sent_tokenize(content)]
    }

pdf_dir = Path("../data")
pdf_files = list(pdf_dir.glob("*.pdf"))

ocr_results = [process_pdf(str(pdf_file)) for pdf_file in pdf_files]

output_path = "../data/ocr_result.json"
with open(output_path, "w") as f:
    json.dump(ocr_results, f, indent=4)

for result in ocr_results:
    print(f"Processed: {result['id']}")
    for sentence in result["sentences"]:
        print(sentence)
