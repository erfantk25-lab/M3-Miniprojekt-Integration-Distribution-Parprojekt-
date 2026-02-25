from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()
#Ladda ONNX-modellen
model = ort.InferenceSession("model/simplecnn.onnx")

@app.get("/")
def root():
    return {"message": "SimpleCNN ONNX Model API - CIFAR-10"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: list):
    #Tar emot bild från json
    image = data["image"]
    # Omvandla till rätt format [1, 3, 32, 32]
    input_array = np.array(image).reshape(1, 3, 32, 32).astype(np.float32)

    #kör modellen
    results = model.run(None, {"input": input_array})
    predictions = results[0][0]

    class_id = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return{
        "class_id": class_id,
        "confidence": confidence
    }

