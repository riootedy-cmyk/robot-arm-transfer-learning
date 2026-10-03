# Robot Arm Transfer Learning Project

## 1. Tujuan project

Project ini membandingkan tiga pendekatan klasifikasi objek berwarna merah, hijau, dan biru menggunakan model ResNet-18 dengan PyTorch:

1. Feature Extraction
2. Partial Fine-Tuning
3. Training from Scratch

Tujuan utama adalah melihat performa model pada dataset 3 kelas berdasarkan akurasi, precision, recall, F1-score, confusion matrix, serta latency inferensi.

## 2. Dataset

Dataset yang digunakan adalah dataset publik/eksternal yang berisi objek dengan warna merah, hijau, dan biru. Dataset tidak dibuat secara buatan dan tidak diklaim berasal dari robot arm jika bukan dari sumber aslinya.

Struktur dataset yang harus dipersiapkan:

```bash
dataset_raw/
├── red/
├── green/
├── blue/
```

Setelah preprocessing, hasil split akan dibuat ke:

```bash
dataset/
├── train/
│   ├── red/
│   ├── green/
│   └── blue/
└── val/
    ├── red/
    ├── green/
    └── blue/
```

## 3. Struktur folder

```bash
robot-arm-transfer-learning-p2/
├── dataset_raw/
│   ├── red/
│   ├── green/
│   └── blue/
├── dataset/
│   ├── train/
│   │   ├── red/
│   │   ├── green/
│   │   └── blue/
│   └── val/
│       ├── red/
│       ├── green/
│       └── blue/
├── models/
├── results/
├── src/
│   ├── prepare_dataset.py
│   ├── train_feature_extraction.py
│   ├── train_finetuning.py
│   ├── train_scratch.py
│   ├── evaluate.py
│   ├── inference.py
│   └── __init__.py
├── requirements.txt
├── README.md
├── main.py
└── .gitignore
```

## 4. Preprocessing

Preprocessing yang diterapkan:

- Resize semua gambar ke 224x224
- Penerapan normalization ImageNet:
  - mean = [0.485, 0.456, 0.406]
  - std = [0.229, 0.224, 0.225]
- Split train/validation tanpa data leakage
- Augmentasi hanya diterapkan pada set training
- Jumlah gambar setiap kelas ditampilkan sebelum dan sesudah split

## 5. Arsitektur ResNet-18

Model yang digunakan adalah ResNet-18 dari torchvision. Arsitektur ini sesuai untuk klasifikasi gambar 3 kelas. Ketiga pendekatan yang dibandingkan adalah:

- Feature Extraction: backbone dibekukan, classifier dilatih
- Partial Fine-Tuning: sebagian akhir backbone dibuka, lalu classifier dilatih
- Scratch: semua parameter dilatih dari awal tanpa pre-trained weights

## 6. Feature Extraction

Pada pendekatan ini:

- ResNet-18 pretrained ImageNet digunakan
- Seluruh backbone dibekukan
- Layer classifier terakhir diganti menjadi 3 kelas
- Hanya classifier yang dilatih
- Model disimpan ke `models/resnet18_feature_extraction.pth`

## 7. Partial Fine-Tuning

Pada pendekatan ini:

- ResNet-18 pretrained ImageNet digunakan
- Kebanyakan backbone dibekukan
- Layer akhir `layer4` dan classifier dibuka untuk fine-tuning
- Model disimpan ke `models/resnet18_partial_finetuning.pth`

## 8. Scratch

Pada pendekatan ini:

- Arsitektur ResNet-18 yang sama digunakan
- Tidak memakai pretrained ImageNet weights
- Semua parameter dilatih dari awal
- Model disimpan ke `models/resnet18_scratch.pth`

## 9. Cara install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pada Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 10. Cara menjalankan training

### Persiapan dataset

```bash
python src/prepare_dataset.py --raw-root dataset_raw --output-root dataset --split-ratio 0.8
```

### Training Feature Extraction

```bash
python src/train_feature_extraction.py --data-dir dataset --epochs 15 --batch-size 32 --lr 1e-3 --seed 42
```

### Training Partial Fine-Tuning

```bash
python src/train_finetuning.py --data-dir dataset --epochs 20 --batch-size 32 --lr 3e-4 --seed 42
```

### Training Scratch

```bash
python src/train_scratch.py --data-dir dataset --epochs 25 --batch-size 32 --lr 1e-3 --seed 42
```

### Atau jalankan semua training sekaligus

```bash
python main.py --prepare --train-feature --train-finetune --train-scratch
```

## 11. Cara melakukan inference

```bash
python src/inference.py --image path/to/image.jpg --model feature_extraction
```

Pilihan model:

- `feature_extraction`
- `partial_finetuning`
- `scratch`

Contoh output:

```bash
Predicted class: RED
Confidence: 97.52%
```

## 12. Cara membaca hasil

Setelah training dan evaluasi selesai, hasil tersimpan di folder `results/`.

Hasil yang dibuat meliputi:

- training loss curve
- validation loss curve
- training accuracy curve
- validation accuracy curve
- confusion matrix
- JSON metrics
- comparison table CSV

Untuk evaluasi model:

```bash
python src/evaluate.py --model-path models/resnet18_feature_extraction.pth --data-dir dataset --model-type feature_extraction
python src/evaluate.py --model-path models/resnet18_partial_finetuning.pth --data-dir dataset --model-type partial_finetuning
python src/evaluate.py --model-path models/resnet18_scratch.pth --data-dir dataset --model-type scratch
```

Untuk membuat tabel perbandingan:

```bash
python main.py --compare
```

## 13. Keterbatasan eksperimen

- Hasil sangat tergantung pada kualitas dataset yang digunakan
- Akurasi dapat bervariasi berdasarkan pembagian data train/validation
- Hyperparameter belum tentu optimal untuk semua dataset
- Latensi inferensi bergantung pada hardware dan ukuran batch
- Model tidak dapat menghasilkan hasil yang valid sebelum dataset benar-benar dipersiapkan

## 14. Urutan perintah yang direkomendasikan

```bash
cd robot-arm-transfer-learning-p2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_dataset.py --raw-root dataset_raw --output-root dataset --split-ratio 0.8
python src/train_feature_extraction.py --data-dir dataset --epochs 15 --batch-size 32 --lr 1e-3 --seed 42
python src/train_finetuning.py --data-dir dataset --epochs 20 --batch-size 32 --lr 3e-4 --seed 42
python src/train_scratch.py --data-dir dataset --epochs 25 --batch-size 32 --lr 1e-3 --seed 42

python src/evaluate.py --model-path models/resnet18_feature_extraction.pth --data-dir dataset --model-type feature_extraction
python src/evaluate.py --model-path models/resnet18_partial_finetuning.pth --data-dir dataset --model-type partial_finetuning
python src/evaluate.py --model-path models/resnet18_scratch.pth --data-dir dataset --model-type scratch
python main.py --compare

python src/inference.py --image path/to/image.jpg --model feature_extraction
```

## 15. Catatan penting

- Jangan menghapus folder `dataset_raw/`
- Jangan mengarang hasil eksperimen
- Hasil akhir hanya boleh ditulis setelah program benar-benar dijalankan
- Jika dependency error terjadi, baca pesan error dan lakukan install package yang sesuai

