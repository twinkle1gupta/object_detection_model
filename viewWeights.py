import torch

state_dict = torch.load("yolo_vgg16.pth", map_location="cpu")

print("Top-level type:", type(state_dict))
if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

for key in state_dict.keys():
    print(key)