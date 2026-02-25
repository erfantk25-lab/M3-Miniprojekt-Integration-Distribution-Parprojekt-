<<<<<<< HEAD
import torch    
import torch.onnx
import os
from src.model import SimpleCNN

def export_model():
    # Create an instance of the model based on the architecture defined in src/model.py
    model = SimpleCNN()

    # Load the trained model weights from the K2 assignment
    weights_path = "model_weights.pth"
 
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found.")
        return
    
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor with the same shape as the model's expected input
    dummy_input = torch.randn(1, 3, 32, 32) 

    # Export the model to ONNX format
    onnx_path = 'model.onnx'
    print(f"Exporting model to '{onnx_path}'...")

    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                        export_params=True,
                        opset_version=18,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'])
    print(f"Model successfully exported to '{onnx_path}'.")

    if os.path.exists(onnx_path):
        print(f"ONNX model file '{onnx_path}' created successfully.")
    else:
        print(f"Error: Failed to create ONNX model file '{onnx_path}'.")
if __name__ == "__main__":
    export_model()
=======
import torch as torch
import sys
sys.path.append('./src')
from model import simpleCNN

#Ladda modellen
model = simpleCNN()
model.load_state_dict(torch.load('./model_weights.pth'))
>>>>>>> 11f98a281f8a4293ada0600065f5f9b55a5b7e5e
