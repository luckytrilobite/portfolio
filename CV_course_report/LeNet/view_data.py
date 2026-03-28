import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# === 資料轉換 ===
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# === 載入資料 ===
train_data = datasets.EMNIST(
    root='./LeNet/data',
    split='digits',
    train=True,
    transform=data_transform,
    download=False  # 已經下載好了
)

# === 查看資料集大小 ===
print("訓練集大小：", len(train_data))
print("類別數量：", len(train_data.classes))
print("類別列表：", train_data.classes)

# === 查看前幾張圖 ===
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    img, label = train_data[i]
    axes[i // 5, i % 5].imshow(img.squeeze(), cmap='gray')
    axes[i // 5, i % 5].set_title(f"Label: {label}")
    axes[i // 5, i % 5].axis('off')

plt.show()
