import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import ResNet18
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型
model = ResNet18(num_classes=2).to(device)
model.load_state_dict(torch.load("./cat_dog/model/resnet18.pth", map_location=device))
model.eval()

# 圖片預處理
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 測試多張圖片
image_folder = "./cat_dog/images/"
image_files = ["cat01.jpg", "dog01.jpg"]

# 類別對應
classes = ["Cat", "Dog"]

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    img = Image.open(img_path).convert("RGB")
    img_tensor = test_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = F.softmax(output, dim=1)  # 計算每個類別的機率
        pred_class = prob.argmax(dim=1).item()  # 取最高機率的類別

    # 輸出結果
    print(f"Image: {img_name}")
    for i, c in enumerate(classes):
        print(f"  {c} probability: {prob[0,i]:.4f}")
    print(f"  Predicted class: {classes[pred_class]}\n")
