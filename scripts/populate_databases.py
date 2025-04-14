# scripts/populate_databases.py
import json
from PIL import Image
import os
import numpy as np
from qdrant_client import QdrantClient, models
import pymongo
from transformers import CLIPProcessor, CLIPModel
import torch
import uuid # Import the uuid library

# --- Configuration ---
# (Keep config the same)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "products"
EMBEDDING_DIM = 512

MONGO_META_URI = "mongodb://admin:password@localhost:27017/"
MONGO_META_DB_NAME = "product_metadata"
MONGO_META_COLLECTION_NAME = "products"

METADATA_FILE = "/home/aakash/Desktop/ai-infra-challenge-gradio/sample_data/metadata.json"
IMAGE_DIR = "/home/aakash/Desktop/ai-infra-challenge-gradio/sample_data/images/"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- ---

print(f"Using device: {DEVICE}")
# (Keep model loading the same)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
model.eval()

# (Keep client connections the same)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
mongo_client = pymongo.MongoClient(MONGO_META_URI)
mongo_db = mongo_client[MONGO_META_DB_NAME]
mongo_collection = mongo_db[MONGO_META_COLLECTION_NAME]

def get_image_embedding(image_path):
    # (Keep embedding generation the same - including normalization)
    try:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def populate():
    # (Keep collection creation and MongoDB clearing the same)
    try:
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
        )
        print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created/recreated.")
    except Exception as e:
        print(f"Failed to create Qdrant collection: {e}. Trying to continue...")
    try:
        delete_result = mongo_collection.delete_many({})
        print(f"MongoDB collection '{MONGO_META_COLLECTION_NAME}' cleared ({delete_result.deleted_count} documents removed).")
    except Exception as e:
        print(f"Warning: Failed to clear MongoDB collection: {e}")

    # (Keep metadata loading the same)
    metadata_path = os.path.abspath(METADATA_FILE)
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found at {metadata_path}")
        return
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    print(f"Loaded {len(metadata_list)} items from {metadata_path}")

    qdrant_points = []
    mongo_docs = []
    processed_count = 0
    id_map = {} # Optional: map original ID to UUID if needed later

    # 4. Process each product
    for i, item in enumerate(metadata_list):
        original_product_id = item.get('product_id') # Keep the original ID
        image_filename = item.get('image_filename')

        if not original_product_id or not image_filename:
            print(f"Warning: Skipping item {i+1} due to missing 'product_id' or 'image_filename'. Data: {item}")
            continue

        image_path = os.path.abspath(os.path.join(os.path.dirname(metadata_path), IMAGE_DIR , image_filename))
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {original_product_id}: {image_path}")
            continue

        print(f"Processing {original_product_id} ({i+1}/{len(metadata_list)})... Image: {image_path}")

        embedding = get_image_embedding(image_path)
        if embedding is None:
            print(f"Warning: Failed to generate embedding for {original_product_id}")
            continue

        # --- FIX: Generate UUID for Qdrant ID ---
        qdrant_id = str(uuid.uuid4()) # Generate a unique UUID string
        id_map[original_product_id] = qdrant_id # Store mapping if needed
        # --- End Fix ---

        qdrant_points.append(models.PointStruct(
            id=qdrant_id, # Use the generated UUID
            vector=embedding.tolist(),
            payload={"product_id": original_product_id} # Store original ID in payload
        ))

        mongo_doc = item.copy()
        # Ensure the document added to mongo uses the original_product_id
        # No change needed here if item already contains the correct product_id
        mongo_docs.append(mongo_doc)
        processed_count += 1

    # (Keep bulk insert logic the same)
    if qdrant_points:
        print(f"Upserting {len(qdrant_points)} points into Qdrant...")
        try:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=qdrant_points,
                wait=True
            )
            print("Qdrant upsert complete.")
        except Exception as e:
            print(f"ERROR: Qdrant upsert failed: {e}", exc_info=True) # Log full traceback

    if mongo_docs:
        print(f"Inserting {len(mongo_docs)} documents into MongoDB...")
        try:
            mongo_collection.insert_many(mongo_docs)
            print("MongoDB insertion complete.")
        except Exception as e:
            print(f"ERROR: MongoDB insertion failed: {e}", exc_info=True) # Log full traceback

    mongo_client.close()
    print(f"Database population finished. Processed {processed_count} valid items.")

if __name__ == "__main__":
    populate()