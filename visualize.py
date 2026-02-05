import torch
import torch.nn as nn
import os
from torchsummary import summary
from torchviz import make_dot
from torchvision.models import mobilenet_v2

# ----------------------------------
# CONFIGURATION
# ----------------------------------
DEVICE = torch.device("cpu")   # change to "cuda" if needed
INPUT_SHAPE = (3, 64, 64)      # ✅ MobileNetV2 input
MODEL_PATH = "mv2slfinal.pth"  # ✅ correct file
OUTPUT_DIR = "outputs"

NUM_CLASSES = 24

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------
# MODEL DEFINITION (MATCH TRAINING)
# ----------------------------------
class MobileNetV2_SL(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.backbone = mobilenet_v2(weights=None)

        # ✅ MATCHES TRAINING ARCHITECTURE EXACTLY
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ----------------------------------
# LOAD MODEL
# ----------------------------------
model = MobileNetV2_SL(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ MobileNetV2 model loaded successfully")

# ----------------------------------
# MODEL SUMMARY
# ----------------------------------
print("\n🔹 MODEL SUMMARY\n")
summary(model, INPUT_SHAPE)

# ----------------------------------
# DUMMY INPUT
# ----------------------------------
dummy_input = torch.randn(1, *INPUT_SHAPE).to(DEVICE)

# ----------------------------------
# FORWARD PASS
# ----------------------------------
output = model(dummy_input)

# ----------------------------------
# VISUALIZE COMPUTATION GRAPH
# ----------------------------------
graph = make_dot(
    output,
    params=dict(model.named_parameters())
)

graph.render(
    os.path.join(OUTPUT_DIR, "mobilenetv2_architecture"),
    format="png"
)

print("✅ Architecture image saved at: outputs/mobilenetv2_architecture.png")

# ----------------------------------
# EXPORT TO ONNX
# ----------------------------------
torch.onnx.export(
    model,
    dummy_input,
    os.path.join(OUTPUT_DIR, "mobilenetv2_model.onnx"),
    input_names=["input"],
    output_names=["output"],
    opset_version=18,          # ✅ MATCHES PYTORCH
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    do_constant_folding=True
)

print("ONNX model saved at: outputs/mobilenetv2_model.onnx")
print("MobileNetV2 visualization complete!")
