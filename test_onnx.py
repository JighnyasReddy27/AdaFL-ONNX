import onnxruntime as ort
import numpy as np
import time

def test_onnx_model(onnx_path):
    print(f"Loading ONNX model from {onnx_path}...")
    
    # Enable available providers (CUDA if available, otherwise CPU)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Initialize the ONNX Runtime session
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        print("ONNX model loaded successfully.")
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return
        
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input Name: {input_name}, Input Shape: {input_shape}")
    
    # Create dummy numpy input
    print("Generating dummy input (1, 3, 1024, 1024)...")
    dummy_input = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    try:
        outputs = session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
        output_data = outputs[0]
        print(f"Inference successful in {end_time - start_time:.4f} seconds.")
        print(f"Output Shape: {output_data.shape}")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == '__main__':
    test_onnx_model('onnx_model/adaifl.onnx')
