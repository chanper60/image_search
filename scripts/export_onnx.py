# export_onnx.py
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
ONNX_EXPORT_DIR = "onnx_models"
TEXT_ONNX_PATH = os.path.join(ONNX_EXPORT_DIR, "clip_text_model.onnx")
IMAGE_ONNX_PATH = os.path.join(ONNX_EXPORT_DIR, "clip_image_model.onnx")

# --- Ensure Export Directory Exists ---
os.makedirs(ONNX_EXPORT_DIR, exist_ok=True)

# --- Device Selection ---
# For ONNX export, CPU is often sufficient and sometimes avoids GPU memory issues
# during the export process itself. The final engine will run on GPU via Triton.
device = "cpu"
logging.info(f"Using device: {device} for ONNX export")

# --- Load Model and Processor ---
try:
    logging.info(f"Loading CLIP model '{MODEL_NAME}'...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval() # Set to eval mode
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    logging.info("Model and processor loaded.")
except Exception as e:
    logging.error(f"Error loading model: {e}", exc_info=True)
    exit()

# --- Export Text Model ---
logging.info("Exporting Text Model to ONNX...")
try:
    # Get sample text inputs matching CLIP's expected format (tokenized IDs)
    # Max length for standard CLIP is 77
    text_input_sample = ["a sample text"]
    text_inputs = processor(text=text_input_sample, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(device)
    input_ids = text_inputs['input_ids'] # Shape: (batch_size, sequence_length=77)

    # Define input/output names for the ONNX graph
    text_input_names = ["input_ids"]
    text_output_names = ["text_embedding"]

    # Define dynamic axes if needed (batch size is usually dynamic)
    dynamic_axes = {'input_ids': {0: 'batch_size'}, # Batch size is dynamic
                    'text_embedding': {0: 'batch_size'}} # Batch size is dynamic

    # Perform ONNX export for the text model part
    # We need to wrap the text model's forward pass for export
    class TextWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids):
            # Directly call the underlying text model or get_text_features
            # Note: get_text_features might do projection + normalization
            # For ONNX export, often better to export the core transformer output
            # and do projection/normalization later if needed, or try exporting get_text_features directly
            text_outputs = self.model.text_model(input_ids=input_ids)
            # Get pooled output (often CLS token representation before projection)
            # pooled_output = text_outputs.pooler_output
            # OR get the features after projection
            text_features = self.model.get_text_features(input_ids=input_ids) # Try exporting this directly
            return text_features

    text_model_wrapper = TextWrapper(model).to(device).eval()

    torch.onnx.export(
        text_model_wrapper,
        input_ids, # Example input tensor
        TEXT_ONNX_PATH,
        input_names=text_input_names,
        output_names=text_output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14, # Use a reasonable opset version; 14+ is good for recent features
        export_params=True,
        do_constant_folding=True,
    )
    logging.info(f"Text model exported successfully to {TEXT_ONNX_PATH}")

except Exception as e:
    logging.error(f"Error exporting text model: {e}", exc_info=True)

# --- Export Image Model ---
logging.info("Exporting Image Model to ONNX...")
try:
    # Get sample image inputs matching CLIP's expected format (processed pixels)
    # Standard CLIP input size is 224x224
    # Create a dummy image tensor: (batch_size, channels, height, width)
    # The processor handles the actual transformation (resize, crop, normalize)
    # We need the shape *after* processing.
    image_input_sample = torch.randn(1, 3, 224, 224).to(device) # Example: 1 image, 3 channels, 224x224 pixels

    # Define input/output names
    image_input_names = ["pixel_values"]
    image_output_names = ["image_embedding"]

    # Define dynamic axes (batch size)
    dynamic_axes = {'pixel_values': {0: 'batch_size'}, # Batch size is dynamic
                    'image_embedding': {0: 'batch_size'}} # Batch size is dynamic

    # Perform ONNX export for the vision model part
    class VisionWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, pixel_values):
            # Directly call get_image_features
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            return image_features

    vision_model_wrapper = VisionWrapper(model).to(device).eval()

    torch.onnx.export(
        vision_model_wrapper,
        image_input_sample, # Example input tensor
        IMAGE_ONNX_PATH,
        input_names=image_input_names,
        output_names=image_output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14, # Match opset version
        export_params=True,
        do_constant_folding=True,
    )
    logging.info(f"Image model exported successfully to {IMAGE_ONNX_PATH}")

except Exception as e:
    logging.error(f"Error exporting image model: {e}", exc_info=True)

logging.info("ONNX export process finished.")