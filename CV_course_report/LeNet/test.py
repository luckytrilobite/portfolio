import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from model import LeNet

device = torch.device('cpu')

# 建立模型並載入權重
output_dim = 10
model = LeNet(output_dim).to(device)
model.load_state_dict(torch.load('./LeNet/model/LeNet.pt', map_location=device))
model.eval()

for i in range(10):
    img = Image.open(f'./LeNet/images/number_{i}.png').convert('L')
    img = img.resize((28, 28), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()

    print("各數字預測機率：")
    for i, p in enumerate(probs[0]):
        print(f"{i}: {p.item()*100:.2f}%")
    print(f"\n最終預測結果：{pred}")

    input("Press Enter to continue...")
