#Team Members:
#Osman Emre Kaya
# Hamza Çiçek 
# Furkan Arslan
# Ayhan Tan Açar 
#-----------------------
# Average accuracy = 0.96

# Gerekli Kütüphanelerin İçe Aktarılması
import torchvision
from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import random
import time
from PIL import Image

# -----------------------------------------------
# 1. Veri Dizini
# -----------------------------------------------
data_dir = "/content/gdrive/MyDrive/ML/dataset/"

# -----------------------------------------------
# 2. Veriyi Yükleme ve Dönüştürme
# -----------------------------------------------

# Temel (Base) Dönüşümler: Boyutlandırma, Tensor'e çevirme, Normalizasyon
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Augmentasyon Dönüşümleri: Rastgele yatay çevirme, döndürme, renk jitter, rastgele yeniden boyutlandırma, rastgele gri tonlama
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                           saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1)
])

# Orijinal Dataset'i Yükleme
original_dataset = datasets.ImageFolder(data_dir, transform=base_transform)
print("Sınıflar:", original_dataset.classes)

# -----------------------------------------------
# 3. Custom AugmentedDataset Sınıfı
# -----------------------------------------------
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, augmentation_transform, augmentations_per_image=10):
        """
        Args:
            original_dataset (Dataset): Orijinal ImageFolder dataset'i.
            augmentation_transform (transforms.Compose): Uygulanacak augmentasyon dönüşümleri.
            augmentations_per_image (int): Her orijinal resim için oluşturulacak augmentasyon sayısı.
        """
        self.original_dataset = original_dataset
        self.augmentation_transform = augmentation_transform
        self.augmentations_per_image = augmentations_per_image
        self.base_transform = original_dataset.transform  # Orijinal dönüşüm

    def __len__(self):
        return len(self.original_dataset) * self.augmentations_per_image

    def __getitem__(self, idx):
        original_idx = idx // self.augmentations_per_image
        augmentation_idx = idx % self.augmentations_per_image
        image, label = self.original_dataset[original_idx]

        if augmentation_idx > 0:
            # Orijinal görüntüyü yeniden yükle
            image_path, _ = self.original_dataset.samples[original_idx]
            image = self.original_dataset.loader(image_path)
            # Augmentasyon dönüşümlerini uygula
            image = self.augmentation_transform(image)
            # Temel dönüşümleri uygula
            image = self.base_transform(image)

        return image, label

# -----------------------------------------------
# 4. Augmented Dataset Oluşturma
# -----------------------------------------------
augmentations_per_image = 10  # 10x veri artırımı için
augmented_dataset = AugmentedDataset(original_dataset, augmentation_transform, augmentations_per_image)
print(f"Orijinal veri sayısı: {len(original_dataset)}")
print(f"Augmented veri sayısı: {len(augmented_dataset)}")

# Tüm Etiketleri Toplama
all_labels = [label for _, label in augmented_dataset]

# -----------------------------------------------
# 5. Model ve SVM Eğitimi İçin Ayarlar
# -----------------------------------------------
n_tests = 5
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b0(pretrained=True)
model.classifier = torch.nn.Identity()  # Özellik çıkarımı için son katmanı kaldırıyoruz
model.to(device)
model.eval()

# -----------------------------------------------
# 6. Zaman Takibi Başlatma
# -----------------------------------------------
start_time = time.time()

# Ortalama Doğruluk için Değişken
avg_accuracy = 0

# -----------------------------------------------
# 7. 5 Çalıştırma İçin Döngü
# -----------------------------------------------
for test_run in range(n_tests):
    print(f"\nTest Çalıştırması: {test_run + 1}")

    # Veriyi Bölme
    train_indices, test_indices = train_test_split(
        range(len(augmented_dataset)),
        test_size=0.33,
        random_state=random.randint(0, 250),
        stratify=all_labels
    )

    train_dataset = Subset(augmented_dataset, train_indices)
    test_dataset = Subset(augmented_dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # -----------------------------------------------
    # 8. Özellik Çıkarımı Fonksiyonu
    # -----------------------------------------------
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for inputs, lbls in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                features.append(outputs.cpu().numpy())
                labels.append(lbls.numpy())
        return np.vstack(features), np.hstack(labels)

    # Özellikler Çıkarılıyor
    print("Özellikler çıkarılıyor...")
    train_features, train_labels = extract_features(train_loader)
    test_features, test_labels = extract_features(test_loader)

    # Özellikleri Ölçeklendirme
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # SVM Eğitimi
    print("SVM eğitiliyor...")
    svm = SVC(kernel='linear', C=1)
    svm.fit(train_features, train_labels)

    # Test ve Doğruluk
    print("Model test ediliyor...")
    predictions = svm.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Doğruluk: {accuracy:.4f}")
    avg_accuracy += accuracy

# -----------------------------------------------
# 9. Ortalama Doğruluk ve Çalışma Süresi
# -----------------------------------------------
avg_accuracy /= n_tests
print(f"\nOrtalama Test Doğruluğu: {avg_accuracy:.4f}")

end_time = time.time()
print(f"\nToplam Çalışma Süresi: {end_time - start_time:.2f} saniye")