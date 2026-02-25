from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np


# Intialize the FastAPI app
app = FastAPI(
title = "CIFAR-10 Image classification API",
description = "En API som använder en ONNX-modell för att klassificera CIFAR-10 bilder.",
version = "1.0.0"
)

# Load the ONNX model session globally to reuse across requests
try:
    session = ort.InferenceSession("model.onnx")
    print("ONNX model loaded successfully.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")


# Define the input data model using Pydantic
class imagedata(BaseModel):
    data: list [float]


@app.get("/")
def read_root():
    return {"message": "Welcome to the CIFAR-10 Image Classification API! Use the /predict endpoint to classify images."}   

@app.post("/predict")
async def predict(image: imagedata):
    """Endpoint to receive image data and return the predicted class. Expects a JSON payload with a 'data' field containing a list of 3072 floats (1x3x32x32).
    """
    try:
        # 1. Convert the input list to a NumPy array and reshape it to match the model's expected input shape
        input_data = np.array(image.data, dtype=np.float32)

        # 2. Validate the input size(3072 values for 1x3x32x32)
        if input_data.size != 3072:
            raise ValueError(f"Invalid input size. Expected 3072 floats for a 1x3x32x32 image, but got {input_data.size}.")
        
        # 3. Reshape the input data to (1, 3, 32, 32)
        input_tensor = input_data.reshape(1, 3, 32, 32)

        # 4. Get the indesx of the input node from the ONNX model
        input_name = session.get_inputs()[0].name
        output_name = session.run(None, {input_name: input_tensor})

        # 5. Get the predicted class index
        prediction = int(np.argmax(output_name[0]))
        
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
        predicted_label = class_names[prediction]

        return {
            "prediction_index": prediction,
            "label": predicted_label,  
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)