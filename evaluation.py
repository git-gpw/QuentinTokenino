import os
import json
import time
from google import genai
from agent import run_cinematic_pipeline

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

EVAL_DATASET = [
    {"user_plot": "Two hitmen talk about fast food burgers in a diner before an execution.", "true_plagiarism": True},
    {"user_plot": "A spaceship computer system malfunctions and decides to eliminate the astronauts.", "true_plagiarism": True},
    {"user_plot": "A corporate accountant travels back in time to medieval Italy to sell smartphones.", "true_plagiarism": False}
]

def run_evaluation_suite():
    print("Launching Advanced Semantic RAG Evaluation via google-genai SDK...\n")
    total = len(EVAL_DATASET)
    parsed = 0
    
    for idx, case in enumerate(EVAL_DATASET):
        try:
            raw_output = run_cinematic_pipeline(case["user_plot"])
            data = json.loads(raw_output)
            parsed += 1
            print(f"Case {idx+1} - Matched: '{data['matched_movie']}' | Score: {data['similarity_score']:.4f} | Plagiarism: {data['detected_plagiarism']}")
        except Exception as e:
            print(f"Failed case {idx+1}: {e}")
        time.sleep(4)
        
    print(f"\nEvaluation Complete. System JSON Compliance: {(parsed/total)*100:.2f}%")

if __name__ == "__main__":
    run_evaluation_suite()