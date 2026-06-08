import os
import pandas as pd
import numpy as np
from google import genai

client = genai.Client(api_key="AQ.Ab8RN6KyIiMU4izKiHGpkCIaQZfN5J21LymyxCY2otGh6t9aqA")

def batch_generate_store_embeddings(csv_path: str = "movies_dataset.csv", output_npy: str = "movies_embeddings.npy"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing base dataset: {csv_path}")
        
    df = pd.read_csv(csv_path)
    plots = df['plot'].tolist()
    
    print(f"Generating semantic embeddings for {len(plots)} movies via text-embedding-004...")
    
    # API limit is 100 requests per batch
    chunk_size = 100
    all_embeddings = []
    
    for i in range(0, len(plots), chunk_size):
        chunk = plots[i:i + chunk_size]
        print(f"Processing chunk {(i // chunk_size) + 1} ({len(chunk)} items)...")
        
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=chunk
        )
        
        chunk_vectors = [e.values for e in response.embeddings]
        all_embeddings.extend(chunk_vectors)
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    np.save(output_npy, embeddings)
    print(f"Successfully stored vector matrix inside '{output_npy}' (Shape: {embeddings.shape})")

if __name__ == "__main__":
    batch_generate_store_embeddings()