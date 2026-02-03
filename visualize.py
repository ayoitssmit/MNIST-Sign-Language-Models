import torch
import os
from torchsummary import summary
from torchviz import make_dot

# ----------------------------------
# IMPORT YOUR MODEL CLASS
# ----------------------------------
from custom_cnn import CustomCNN

# ----------------------------------
# CONFIGURATION
# ----------------------------------
DEVICE = torch.device("cpu")      # use CPU for visualization
INPUT_SHAPE = (1, 28, 28)         # (C, H, W)
MODEL_PATH = "CNN.pth"
OUTPUT_DIR = "outputs"

# create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------
# LOAD MODEL AND WEIGHTS
# ----------------------------------
model = CustomCNN(num_classes=26).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded successfully")

# ----------------------------------
# PRINT MODEL SUMMARY
# ----------------------------------
print("\n🔹 MODEL SUMMARY\n")
summary(model, INPUT_SHAPE)

# ----------------------------------
# CREATE DUMMY INPUT
# ----------------------------------
dummy_input = torch.randn(1, *INPUT_SHAPE).to(DEVICE)

# ----------------------------------
# FORWARD PASS
# ----------------------------------
output = model(dummy_input)

# ----------------------------------
# VISUALIZE ARCHITECTURE + DATA FLOW
# ----------------------------------
graph = make_dot(
    output,
    params=dict(model.named_parameters()),
    show_attrs=False,
    show_saved=False
)

graph.render(
    os.path.join(OUTPUT_DIR, "cnn_architecture"),
    format="png"
)

print("✅ Architecture image saved at: outputs/cnn_architecture.png")

# ----------------------------------
# EXPORT MODEL TO ONNX (FOR NETRON)
# ----------------------------------
torch.onnx.export(
    model,
    dummy_input,
    os.path.join(OUTPUT_DIR, "cnn_model.onnx"),
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("✅ ONNX model saved at: outputs/cnn_model.onnx")
print("\n🎉 Visualization complete!")
