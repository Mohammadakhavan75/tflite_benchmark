import torch
from torchvision import models


# model_path = "./best_params.pt"
model = models.resnet34(pretrained=False)
num_ftrs = model.heads[-1].in_features
model.heads[-1] = torch.nn.Linear(num_ftrs, 5)

# state_dict = torch.load(model_path)
# model.load_state_dict(state_dict)

input_names = [ "actual_input" ]
output_names = [ "output" ]

torch.onnx.export(model, torch.randn(1, 3, 224, 224), "vit.onnx", verbose=False, input_names=input_names, output_names=output_names, export_params=True)
