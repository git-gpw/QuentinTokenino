import os
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from schema import MovieAnalysis

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def compute_cosine_similarity(vector_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    dot_product = np.dot(matrix_b, vector_a)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(matrix_b, axis=1)
    return dot_product / (norm_a * norm_b)

def run_cinematic_pipeline(user_plot: str, csv_path: str = "movies_dataset.csv", npy_path: str = "movies_embeddings.npy") -> str:
    if not os.path.exists(csv_path) or not os.path.exists(npy_path):
        raise FileNotFoundError("Missing database components. Run generation scripts first.")
        
    df = pd.read_csv(csv_path)
    dataset_embeddings = np.load(npy_path)
    
    query_response = client.models.embed_content(
        model="text-embedding-004",
        contents=user_plot
    )
    user_vector = np.array(query_response.embeddings[0].values, dtype=np.float32)
    
    similarity_scores = compute_cosine_similarity(user_vector, dataset_embeddings)
    max_idx = np.argmax(similarity_scores)
    max_score = float(similarity_scores[max_idx])
    
    matched_entry = df.iloc[max_idx]
    assigned_director = matched_entry['director']
    matched_movie = matched_entry['title']
    
    prompt = f"""
    You are an expert film critic and academic professor in cinematography.
    
    USER INPUT CONCEPT:
    "{user_plot}"
    
    ADVANCED SEMANTIC RETRIEVAL RESULTS:
    - Closest semantic match: "{matched_movie}"
    - Automatically routed director target: {assigned_director}
    - Calculated mathematical similarity score: {max_score:.4f}
    
    STRICT GENERATION INSTRUCTIONS:
    1. Set 'detected_plagiarism' to True if the similarity score is >= 0.75. Otherwise, set it to False.
    2. Rewrite the user's plot inside 'rewritten_plot' adapting the narrative structure to match the style of {assigned_director}.
    3. Document your technical adaptation choices inside the 'stylistic_notes' field.
    """
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MovieAnalysis,
            temperature=0.1
        )
    )
    return response.text

if __name__ == "__main__":
    sample_input = "An astronaut gets isolated on a desert planet and has to grow food to survive."
    print("Executing Semantic RAG pipeline check...")
    print(run_cinematic_pipeline(sample_input))