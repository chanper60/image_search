# Command to build the TEXT model TensorRT engine (FP16)
trtexec --onnx=onnx_models/clip_text_model.onnx \
        --saveEngine=tensorrt_engines/clip_text_fp16.plan \
        --fp16 \
        --workspace=4096 \
        --minShapes=input_ids:1x77 \
        --optShapes=input_ids:8x77 \
        --maxShapes=input_ids:32x77 \
        --verbose # Add --verbose for detailed output during build

# Command to build the IMAGE model TensorRT engine (FP16)
trtexec --onnx=onnx_models/clip_image_model.onnx \
        --saveEngine=tensorrt_engines/clip_image_fp16.plan \
        --fp16 \
        --workspace=4096 \
        --minShapes=pixel_values:1x3x224x224 \
        --optShapes=pixel_values:8x3x224x224 \
        --maxShapes=pixel_values:16x3x224x224 \
        --verbose # Add --verbose for detailed output during build
