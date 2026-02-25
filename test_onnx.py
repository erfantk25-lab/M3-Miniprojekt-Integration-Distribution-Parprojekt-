import onnxruntime as ort
import numpy as np

def test_model():
    print("Testing ONNX model...")
    try:
        
        session = ort.InferenceSession("model.onnx")
        
       
        input_name = session.get_inputs()[0].name
        dummy_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
        
       
        result = session.run(None, {input_name: dummy_data})
        
        print("SUCCESS: The ONNX model is working!")
        print(f"Output shape: {result[0].shape} (Should be 1, 10)")
        
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    test_model()