import gradio as gr
import aiohttp # Use async HTTP client
import qdrant_client
import pymongo
from PIL import Image
import numpy as np
from transformers import CLIPProcessor # Processor includes tokenizer
import io
import os
import json
from datetime import datetime
import traceback
import time
import asyncio # Import asyncio

# --- Configuration ---
# (Configuration remains the same)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = "products"

MONGO_META_URI = os.getenv("MONGO_META_URI", "mongodb://admin:password@localhost:27017/")
MONGO_META_DB_NAME = "product_metadata"
MONGO_META_COLLECTION_NAME = "products"

MONGO_LOG_URI = os.getenv("MONGO_LOG_URI", "mongodb://admin:password@localhost:27018/")
MONGO_LOG_DB_NAME = "service_logs"
MONGO_LOG_COLLECTION_NAME = "matching_logs"

TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "http://localhost:8000")
TRITON_IMAGE_MODEL_NAME = "clip_image_trt"
TRITON_TEXT_MODEL_NAME = "clip_text_trt"
TRITON_MODEL_VERSION = "1"

CLIP_PROCESSOR_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
TOP_K = 5

GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")

# --- Initialize Clients ---
# Qdrant Client
try:
    qdrant_cli = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    print(f"Attempting Qdrant connection check to {QDRANT_HOST}:{QDRANT_PORT}...")
    # FIX 1: Use a different connection check - try getting collection info
    # This will raise an exception if connection fails or collection doesn't exist (we handle the latter)
    try:
        qdrant_cli.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print("Qdrant connection successful (Collection found).")
    except Exception as collection_ex:
         # If the error is specifically about the collection not found, connection is likely OK
         # Check the type of exception qdrant_client raises for "not found"
         # For now, assume any exception here means potential connection issue *or* collection missing
         print(f"Qdrant: Collection '{QDRANT_COLLECTION_NAME}' not found or other connection issue: {collection_ex}. Assuming connection OK if Qdrant is running.")
         # We proceed, assuming the collection exists or populate script will create it.

except Exception as e:
    print(f"ERROR connecting to Qdrant during client initialization: {e}. Check hostname/port and if Qdrant is running.")
    qdrant_cli = None # Set to None if init fails

# MongoDB Clients (Keep the same robust init)
try:
    mongo_meta_client = pymongo.MongoClient(MONGO_META_URI, serverSelectionTimeoutMS=5000)
    mongo_meta_db = mongo_meta_client[MONGO_META_DB_NAME]
    mongo_meta_collection = mongo_meta_db[MONGO_META_COLLECTION_NAME]
    mongo_meta_client.server_info()
    print("MongoDB Metadata connection successful.")
except Exception as e:
    print(f"ERROR connecting to MongoDB Metadata: {e}")
    mongo_meta_collection = None # Set to None

try:
    mongo_log_client = pymongo.MongoClient(MONGO_LOG_URI, serverSelectionTimeoutMS=5000)
    mongo_log_db = mongo_log_client[MONGO_LOG_DB_NAME]
    mongo_log_collection = mongo_log_db[MONGO_LOG_COLLECTION_NAME]
    mongo_log_client.server_info()
    print("MongoDB Logging connection successful.")
except Exception as e:
    print(f"ERROR connecting to MongoDB Logging: {e}")
    mongo_log_collection = None # Set to None

# Load CLIP Processor
try:
    processor = CLIPProcessor.from_pretrained(CLIP_PROCESSOR_NAME)
    print("CLIP Processor/Tokenizer loaded successfully.")
except Exception as e:
    print(f"ERROR loading CLIP Processor/Tokenizer: {e}")
    processor = None # Set to None

# --- Helper Functions ---

def log_event(level, message, data=None, error=None):
    """Logs an event to the MongoDB logging collection."""
    # FIX 2: Check if mongo_log_collection is None
    if mongo_log_collection is None:
        print(f"LOG ({level}): {message} | Data: {data} | Error: {error} (Logging DB not available)")
        return

    log_entry = {
        "timestamp": datetime.utcnow(),
        "level": level.upper(),
        "message": message,
        "data": data,
    }
    if error:
        log_entry["error"] = str(error)
        log_entry["traceback"] = traceback.format_exc()
    try:
        # Use the correctly checked collection object
        mongo_log_collection.insert_one(log_entry)
    except Exception as e:
        print(f"CRITICAL: Failed to write log to MongoDB: {e}")

# --- ASYNC Helper Functions for Triton ---
# (get_embedding_from_triton function remains the same as previous async version)
async def get_embedding_from_triton(session: aiohttp.ClientSession, model_name: str, payload: dict):
    """Generic async function to get embedding from a Triton model."""
    log_event("info", f"Sending async request to Triton Model ({model_name})")
    start_time = time.time()
    triton_infer_url = f"{TRITON_HTTP_URL}/v2/models/{model_name}/versions/{TRITON_MODEL_VERSION}/infer"

    try:
        async with session.post(triton_infer_url, json=payload, timeout=30) as response:
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            result = await response.json() # Wait for the JSON response

        # Determine expected output name based on model --- FIX APPLIED HERE ---
        if model_name == TRITON_IMAGE_MODEL_NAME:
            output_name = "image_embedding" # Correct name from image config.pbtxt
        elif model_name == TRITON_TEXT_MODEL_NAME:
            output_name = "text_embedding"  # Correct name from text config.pbtxt
        else:
            # Should not happen if called correctly, but good practice
            raise ValueError(f"Unknown model name '{model_name}' for determining output name.")

        # Find the output dictionary using the CORRECT name
        embedding_output = next(o for o in result['outputs'] if o['name'] == output_name)

        embedding = np.array(embedding_output['data'], dtype=np.float32).flatten()
        duration = time.time() - start_time
        log_event("info", f"Received embedding from Triton {model_name} ({duration:.4f}s)")
        return embedding

    except aiohttp.ClientError as e:
        log_event("error", f"HTTP ClientError contacting Triton ({model_name})", error=e, data={"url": triton_infer_url})
        raise ConnectionError(f"Failed communication with Triton ({model_name}): {e}") from e
    except asyncio.TimeoutError:
         log_event("error", f"Timeout contacting Triton ({model_name})", data={"url": triton_infer_url})
         raise ConnectionError(f"Timeout connecting to Triton ({model_name})") from e
    except StopIteration: # Catch the specific error if the output name is somehow still wrong
        log_event("error", f"Could not find output name '{output_name}' in Triton response", data={"response": result})
        raise ValueError(f"Output tensor '{output_name}' not found in Triton response.")
    except Exception as e:
        # Catch potential parsing errors or other unexpected issues
        log_event("error", f"Failed processing Triton ({model_name}) response", error=e, data={"response": result}) # Log the response result
        raise ValueError(f"Error processing Triton ({model_name}) response: {e}") from e


# --- Synchronous DB Helpers ---
# (search_qdrant and get_metadata_from_mongo remain the same synchronous versions)
def search_qdrant(embedding_list):
    if qdrant_cli is None: # Check if None
        raise ConnectionError("Qdrant client is not available.")
    log_event("info", f"Searching Qdrant for top {TOP_K} matches")
    start_time = time.time()
    try:
        search_result = qdrant_cli.search(
            collection_name=QDRANT_COLLECTION_NAME, query_vector=embedding_list, limit=TOP_K,
        )
        duration = time.time() - start_time
        log_event("info", f"Qdrant search completed ({duration:.4f}s)", data={"hits": len(search_result)})
        return search_result
    except Exception as e:
        log_event("error", "Qdrant search failed", error=e)
        raise ConnectionError(f"Qdrant search failed: {e}") from e

def get_metadata_from_mongo(product_id):
    if mongo_meta_collection is None: # Check if None
        raise ConnectionError("MongoDB Metadata client is not available.")
    log_event("info", f"Querying MongoDB Metadata for product_id: {product_id}")
    start_time = time.time()
    try:
        metadata = mongo_meta_collection.find_one({"product_id": product_id})
        duration = time.time() - start_time
        if metadata:
            if '_id' in metadata: metadata['_id'] = str(metadata['_id'])
            log_event("info", f"Metadata found for {product_id} ({duration:.4f}s)")
            return metadata
        else:
            log_event("warning", f"No metadata found for product_id: {product_id} ({duration:.4f}s)")
            return None
    except Exception as e:
        log_event("error", f"MongoDB metadata retrieval failed for {product_id}", error=e)
        return None

# --- Main Gradio Processing Function (ASYNC) ---
async def find_product_matches(input_image: Image.Image, input_text: str):
    """
    Main ASYNC function called by Gradio. Handles image/text input, calls Triton async.
    """
    overall_start_time = time.time()
    query_type = None
    query_data = None

    # Determine input type
    if input_image is not None:
        query_type = "image"
        query_data = input_image
        log_event("info", "Processing IMAGE input.") # Call log_event AFTER checking mongo_log_collection below
    elif input_text and input_text.strip():
        query_type = "text"
        query_data = input_text.strip()
        log_event("info", f"Processing TEXT input: '{query_data}'") # Call log_event AFTER checking mongo_log_collection below
    else:
        return "Please provide either an image or text input.", None

    # FIX 3: Check required clients using 'is not None'
    # Check if clients initialized correctly
    if not all([
        qdrant_cli is not None,
        mongo_meta_collection is not None,
        mongo_log_collection is not None, # Crucially, check the log collection itself here
        processor is not None
    ]):
         # Log directly to print as mongo_log_collection might be None
         err_msg = "System Initialization Error: Clients (Qdrant, Mongo Meta, Mongo Log, Processor) failed to initialize properly. Check console logs at startup."
         print(f"ERROR: {err_msg}")
         # Optionally try logging if the log db is available
         if mongo_log_collection is not None:
             log_event("error", err_msg)
         return err_msg, None

    # --- Now safe to call log_event ---
    # Log the input type determined earlier
    if query_type == "image":
        log_event("info", "Processing IMAGE input confirmed.")
    elif query_type == "text":
        log_event("info", f"Processing TEXT input confirmed: '{query_data}'")
    # ---

    try:
        embedding = None
        payload = None
        model_name = None

        # 1. Preprocess (Sync)
        # (Preprocessing logic remains the same)
        if query_type == "image":
            log_event("info", "Preprocessing image...")
            start_preprocess = time.time()
            query_data = query_data.convert("RGB")
            inputs = processor(images=query_data, return_tensors="np", padding=True)
            pixel_values = inputs['pixel_values'].astype(np.float32)
            log_event("info", f"Preprocessing done ({time.time() - start_preprocess:.4f}s)")
            # Prepare payload for image model
            payload = {
                "inputs": [{"name": "pixel_values", "shape": list(pixel_values.shape), "datatype": "FP32", "data": pixel_values.tolist()}],
                # FIX 3: Match image config output name
                "outputs": [{"name": "image_embedding"}]
            }
            model_name = TRITON_IMAGE_MODEL_NAME

        elif query_type == "text":
            log_event("info", "Tokenizing text...")
            start_preprocess = time.time()
            # NOTE: Ensure tokenizer padding/truncation matches the fixed dims [77] in config.pbtxt
            # Max length 77 is standard for CLIP.
            inputs = processor(text=query_data, return_tensors="np", padding="max_length", truncation=True, max_length=77)
            input_ids = inputs['input_ids'].astype(np.int64)
            # attention_mask = inputs['attention_mask'].astype(np.int64) # Not needed according to config
            log_event("info", f"Tokenizing done ({time.time() - start_preprocess:.4f}s)")
            # Prepare payload for text model
            payload = {
                "inputs": [
                    # FIX 2: Only send input_ids
                    {"name": "input_ids", "shape": list(input_ids.shape), "datatype": "INT64", "data": input_ids.tolist()},
                    # REMOVED attention_mask input
                ],
                 # FIX 1: Match text config output name
                "outputs": [{"name": "text_embedding"}]
            }
            model_name = TRITON_TEXT_MODEL_NAME

        if payload is None or model_name is None:
            raise ValueError("Failed to prepare payload for Triton.")

        # 2. Get Embedding from Triton (Async)
        async with aiohttp.ClientSession() as session:
             embedding = await get_embedding_from_triton(session, model_name, payload) # Gets the raw embedding

        if embedding is None:
             raise ValueError("Failed to generate embedding for the input.")

        # --- FIX: NORMALIZE THE EMBEDDING FROM TRITON ---
        log_event("info", "Normalizing embedding received from Triton...")
        norm = np.linalg.norm(embedding)
        if norm < 1e-6: # Avoid division by zero for zero vectors
            log_event("warning", "Received near-zero embedding from Triton. Using as is.")
            normalized_embedding = embedding
        else:
            normalized_embedding = embedding / norm

        # 3. Search Vector DB (Sync)
        search_results = search_qdrant(normalized_embedding.tolist())

        # 4. Retrieve Metadata & Format Output (Sync)
        # (Formatting logic remains the same)
        output_text = f"Found {len(search_results)} potential matches for your {query_type} query:\n\n"
        matched_images_data = []

        if not search_results:
            output_text += f"No similar products found for the {query_type} query."
        else:
            for i, hit in enumerate(search_results):
                # 'hit' is a ScoredPoint object
                qdrant_point_id = hit.id # This is the UUID generated during population
                score = hit.score
                payload = hit.payload # Get the payload dictionary

                # --- FIX: Get original product_id FROM PAYLOAD ---
                original_product_id = payload.get("product_id") if payload else None
                # --- End Fix ---

                output_text += f"**Match {i+1} (Score: {score:.4f})**\n"
                # Display Qdrant's UUID if desired, or just the original ID
                # output_text += f"- Qdrant ID: {qdrant_point_id}\n"

                if original_product_id:
                    output_text += f"- Product ID: {original_product_id}\n"
                    # Use the original_product_id to query MongoDB
                    metadata = get_metadata_from_mongo(original_product_id)
                    if metadata:
                        output_text += f"- Name: {metadata.get('name', 'N/A')}\n"
                        output_text += f"- Category: {metadata.get('category', 'N/A')}\n"
                        output_text += f"- Price: {metadata.get('price', 'N/A')}\n"
                        img_filename = metadata.get('image_filename', None)
                        if img_filename:
                            output_text += f"- Source Image File: {img_filename}\n"
                            image_path = os.path.abspath(os.path.join('sample_data', 'images', img_filename))
                            if os.path.exists(image_path):
                                 matched_images_data.append((image_path, f"Match {i+1}: {metadata.get('name', '')} ({score:.2f})"))
                            else:
                                 log_event("warning", f"Image file not found for gallery display: {image_path}")
                    else:
                        output_text += "- Metadata: Not found in MongoDB for this ID.\n"
                else:
                     output_text += "- Product ID: Not found in Qdrant payload.\n" # Should not happen if population worked
                output_text += "---\n"

        overall_duration = time.time() - overall_start_time
        log_event("info", f"Processing completed ({overall_duration:.4f}s)")

        return output_text, matched_images_data if matched_images_data else None

    # Catch specific errors from async calls
    except ConnectionError as e:
         log_event("error", "Connection error during processing", error=e)
         return f"Error: Could not connect to a required service (Triton/Qdrant/MongoDB). Details: {e}", None
    except ValueError as e: # Catch Triton parsing errors etc.
         log_event("error", "Value error during processing", error=e)
         return f"Error: Problem processing data or response. Details: {e}", None
    except Exception as e:
        log_event("error", "Unexpected error in Gradio processing function", error=e)
        return f"An unexpected error occurred: {e}", None

# --- Define Gradio Interface ---
# (Interface definition remains the same)
with gr.Blocks() as demo:
    gr.Markdown("# AI Product Matcher Demo (Async - Image & Text)")
    gr.Markdown("Upload an image OR enter text description to find similar items.")
    example_images_path = [
        os.path.abspath(os.path.join("sample_data", "images", img))
        for img in ["6440.jpg", "2826.jpg", "4523.jpg"]
        if os.path.exists(os.path.abspath(os.path.join("sample_data", "images", img)))
    ]
    example_texts = ["blue shoes", "nike shoes", "grey sandals"]
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Product Image")
            gr.Markdown("### OR")
            input_text = gr.Textbox(label="Enter Text Description", placeholder="e.g., red running shoes")
            submit_button = gr.Button("Find Matches")
            if example_images_path:
                gr.Examples(
                    examples=example_images_path, inputs=input_image, outputs=[],
                    fn=find_product_matches, cache_examples=False, label="Example Images (Click to Run)"
                )
            gr.Examples(
                examples=example_texts, inputs=input_text, outputs=[],
                fn=find_product_matches, cache_examples=False, label="Example Texts (Click to Run)"
            )
        with gr.Column(scale=2):
            results_gallery = gr.Gallery(label="Matched Product Images", visible=True, columns=3, rows=2, object_fit="contain", height="auto") # Set style in constructor
            results_markdown = gr.Markdown(label="Matching Results")
    submit_button.click(
        fn=find_product_matches,
        inputs=[input_image, input_text],
        outputs=[results_markdown, results_gallery]
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("Attempting to launch Gradio interface...")
    print(f"Ensure backend services are running via docker-compose.")
    print(f"Gradio server binding to: {GRADIO_SERVER_NAME}")
    # Use server_name from env var for Docker binding
    # Set queue(concurrency_count=...) for Gradio if needed to handle more concurrent users making async calls
    demo.queue() # Enable queue for better handling of async calls
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=7860)
