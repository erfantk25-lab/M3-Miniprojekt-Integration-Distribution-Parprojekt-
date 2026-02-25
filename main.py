from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()
#Load ONNX-modellen
model = ort.InferenceSession("model/simplecnn.onnx")

@app.get("/")
def root():
    return {"message": "SimpleCNN ONNX Model API - CIFAR-10"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: dict):
    # Try-except for error handling
    try:
        if "image" not in data:
            return {"error": "Missing 'image' key in data"}
        
        # Receiving image from json
        image = data["image"]
        
        if len(image) != 3072:
            return {"error": f"Invalid image size. Expected 3072, got {len(image)}"}
        
        # Convert to the correct format [1, 3, 32, 32]
        input_array = np.array(image).reshape(1, 3, 32, 32).astype(np.float32)
        
        # Run the model
        results = model.run(None, {"input": input_array})
        predictions = results[0][0]
        
        class_id = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        return {
            "class_id": class_id,
            "confidence": confidence
        }
    
    except Exception as e:
        return {"error": str(e)}
