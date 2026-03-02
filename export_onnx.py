<<<<<<< HEAD
import torch    
=======
import torch
>>>>>>> d6597a46b59309c669f9834e5fc80735a72a9042
import torch.onnx
import os
# Import the model architecture from the src folder
from src.model import SimpleCNN
#bra kod!
def export_model():
<<<<<<< HEAD
    # Create an instance of the model based on the architecture defined in src/model.py
    model = SimpleCNN()

    # Load the trained model weights from the K2 assignment
    weights_path = "model_weights.pth" # ska vara src/model_weights.pth 
 
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found.")
        return
=======
    """
    This script exports the trained PyTorch model to ONNX format.
    This is a requirement for the M3 project (Model Integration).
    """
>>>>>>> d6597a46b59309c669f9834e5fc80735a72a9042
    
    # 1. Initialize the model architecture
    model = SimpleCNN()
    
    # 2. Define the path to the pre-trained weights from K2
    weights_path = "model_weights.pth"
    
    # Check if the weights file exists in the root directory
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found. Please ensure the file is in the root folder.")
        return

    # Load the weights into the model (mapping to CPU for compatibility)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    # 3. Set the model to evaluation mode
    model.eval()

    # 4. Create a dummy input tensor matching the CIFAR-10 shape (1, 3, 32, 32)
    dummy_input = torch.randn(1, 3, 32, 32)

<<<<<<< HEAD
    # Export the model to ONNX format
    onnx_path = 'model.onnx' #bättre att spara i model/ för att hålla ordning
    print(f"Exporting model to '{onnx_path}'...")
=======
    # 5. Export the model to ONNX format
    onnx_file = "model.onnx"
    print(f"Exporting model to '{onnx_file}'...")
    
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_file, 
            export_params=True,       # Store the trained weights inside the file
            opset_version=18,         # Using version 18 for best compatibility with Python 3.13
            do_constant_folding=True, # Optimize the model graph
            input_names=['input'],    # Name the input node
            output_names=['output']   # Name the output node
        )
        
        print(f"SUCCESS: Model successfully exported to '{onnx_file}'.")
        
    except Exception as e:
        print(f"An error occurred during the ONNX export: {e}")
>>>>>>> d6597a46b59309c669f9834e5fc80735a72a9042

if __name__ == "__main__":
    export_model()
<<<<<<< HEAD

=======
>>>>>>> d6597a46b59309c669f9834e5fc80735a72a9042
