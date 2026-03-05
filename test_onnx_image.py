import onnxruntime as ort
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def preprocess_image(img_path):
    # Match test.py logic
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 1024x1024
    img = cv2.resize(img, (1024, 1024))
    
    # Normalize (0-1) and transpose to (C, H, W)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def test_onnx_image(onnx_path, img_path, output_path):
    print(f"Loading ONNX model from {onnx_path}...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    print(f"Preprocessing image: {img_path}")
    input_data = preprocess_image(img_path)
    
    input_name = session.get_inputs()[0].name
    print("Running inference...")
    outputs = session.run(None, {input_name: input_data})
    
    mask_pred = outputs[0]
    
    # Apply sigmoid as per test.py
    mask_pred = sigmoid(mask_pred)
    
    # Post-process mask
    mask = (mask_pred > 0.5).astype(np.float32) * 255
    mask = np.squeeze(mask)
    
    # Save mask
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, mask, cmap='gray')
    print(f"Result saved to: {output_path}")

if __name__ == '__main__':
    onnx_path = 'onnx_model/adaifl.onnx'
    img_path = 'test_images/img_1.jpg'
    output_path = 'results/onnx_img_1.png'
    test_onnx_image(onnx_path, img_path, output_path)
