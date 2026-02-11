import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import os

def main():
    print("🟢 STARTING RAG MEMORY INGESTION...")

    # 1. SETUP PATHS
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Locates the CSV file generated in Step 1
    CSV_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", "ai_data.csv"))

    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: {CSV_PATH} not found. Run '01_ai_factory.py' first.")
        return

    # 2. CONNECT TO QDRANT
    # Requires qdrant.exe to be running in a separate terminal
    print("🔌 Connecting to Qdrant Database...")
    try:
        client = QdrantClient(url="http://localhost:6333")
        client.get_collections() # Test connection
        print("✅ Connected to Qdrant!")
    except Exception:
        print("❌ CONNECTION FAILED!")
        print("⚠️ Ensure 'qdrant.exe' is running in another terminal window.")
        return

    # 3. CREATE COLLECTION (The Database Table)
    collection_name = "supermarket_db"
    
    # Refresh the collection to avoid duplicate or old data
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"🗑️ Deleted old '{collection_name}' collection.")
    
    # size=384 matches the output of the 'all-MiniLM-L6-v2' model
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"🆕 Created empty '{collection_name}'.")

    # 4. LOAD EMBEDDING MODEL (The Translator)
    print("🧠 Loading Embedding Model (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # 5. UPLOAD DATA
    print("🚀 Uploading Memory...")
    df = pd.read_csv(CSV_PATH)
    points = []

    for idx, row in df.iterrows():
        # --- THE SYNC FIX ---
        # We use just the class name (e.g., 'RedApple') as the primary search text
        # This makes it easier for YOLO's output to find a match
        search_text = str(row['class'])
        
        # Convert text into a math vector (list of 384 numbers)
        vector = encoder.encode(search_text).tolist()
        
        # Create a Data Point with search vector and readable payload
        point = PointStruct(
            id=idx, 
            vector=vector, 
            payload={
                "product_name": row['class'],
                "description": row['description'],
                "price": row['price'],
                "image_path": row['path']
            }
        )
        points.append(point)

    # Push all points to the database
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"🎉 SUCCESS! Uploaded {len(points)} memories to Qdrant.")
    print("✅ Your AI can now 'remember' product details perfectly.")

if __name__ == '__main__':
    main()