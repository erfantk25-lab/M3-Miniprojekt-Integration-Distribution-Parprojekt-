import json
import numpy as np
from PIL import Image

def convert_image(image_path):
    img = Image.open(image_path).resize((32, 32))
    
    img_array = np.array(img).astype(np.float32) / 255.0
    
    img_array = img_array.transpose(2, 0, 1)
    
    flat_list = img_array.flatten().tolist()
    
    return json.dumps({"data": flat_list})

print(convert_image("test.jpg"))