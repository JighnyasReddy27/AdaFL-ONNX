import torch
import os
import sys

# Force UTF-8 encoding for stdout to prevent PyTorch logger from crashing on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
import models.modules.fi_attn as fi_attn
fi_attn.EXPORT_MODE = True
from models.net import AdaIFL

def export_to_onnx(model_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the model
    model = AdaIFL().to(device)
    
    print(f"Loading checkpoint from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # The original test script loads into a DataParallel wrapper.
    # We will strip the 'module.' prefix so we can load it into the base model directly.
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'adaifl.onnx')
    
    # Create a dummy input tensor matching the expected input shape for AdaIFL
    # (1 batch, 3 channels, 1024 height, 1024 width)
    print("Creating dummy input tensor...")
    dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
    
    print("Starting ONNX export...")
    try:
        torch.onnx.export(
            model,                       # model being run
            dummy_input,                 # model input (or a tuple for multiple inputs)
            output_path,                 # where to save the model (can be a file or file-like object)
            export_params=True,          # store the trained parameter weights inside the model file
            opset_version=17,            # the ONNX version to export the model to (requires 17 for some modern ops)
            do_constant_folding=True,    # whether to execute constant folding for optimization
            input_names=['input'],       # the model's input names
            output_names=['output'],     # the model's output names
            dynamic_axes={               # variable length axes
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Successfully exported model to {output_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == '__main__':
    model_path = 'AdaIFL_v0.pth'
    output_dir = 'onnx_model'
    export_to_onnx(model_path, output_dir)
