import torch
import torchvision.transforms as transforms
import requests
from PIL import Image
from io import BytesIO
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath('scripts/gen_resnet_model.py')))

hub_dir = os.path.expanduser("~/.cache/torch/hub/akamaster_pytorch_resnet_cifar10_master")
sys.path.insert(0, hub_dir)
from resnet import resnet20

model = resnet20()
checkpoint_path = os.path.join(hub_dir, "pretrained_models", "resnet20-12fca82f.th")
state = torch.load(checkpoint_path, map_location="cpu")
sd = state.get("state_dict", state)
sd = {k.replace("module.", ""): v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()

url = "https://github.com/YoongiKim/CIFAR-10-images/raw/master/test/bird/0000.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(input_tensor)
    
print("resnet20 FP32 Logits:", out)
print("resnet20 FP32 Pred:", out.argmax().item())
