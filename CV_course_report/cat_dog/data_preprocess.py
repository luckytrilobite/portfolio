import os, shutil, random

random.seed(42)

# 原始資料夾
src_folder = r"C:\VScode\python\AI_practice\cat_dog\data\kagglecatsanddogs_5340\PetImages"

# 拆分後資料夾
train_folder = r"C:\VScode\python\AI_practice\cat_dog\data\train"
val_folder   = r"C:\VScode\python\AI_practice\cat_dog\data\val"

for label in ["Cat", "Dog"]:
    label_folder = os.path.join(src_folder, label)          # 原始資料夾子路徑
    files = os.listdir(label_folder)                        # 取得該類別所有檔案
    files = [f for f in files if f.lower().endswith((".jpg", ".png"))]  # 過濾圖片
    random.shuffle(files)

    n_train = int(len(files) * 0.8)

    # 訓練集
    for f in files[:n_train]:
        os.makedirs(os.path.join(train_folder, label), exist_ok=True)
        shutil.copy(os.path.join(label_folder, f), os.path.join(train_folder, label, f))

    # 驗證集
    for f in files[n_train:]:
        os.makedirs(os.path.join(val_folder, label), exist_ok=True)
        shutil.copy(os.path.join(label_folder, f), os.path.join(val_folder, label, f))
