import torch as torch
import sys
sys.path.append('./src')
from model import simpleCNN

#Ladda modellen
model = simpleCNN()
model.load_state_dict(torch.load('./model_weights.pth'))
