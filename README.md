# 🍎🍊 Apple vs Orange Classifier

Aplikasi web sederhana untuk klasifikasi gambar apel dan jeruk menggunakan deep learning, dibangun dengan Streamlit dan TensorFlow.

## 🔗 Demo

Coba langsung di: appleorange-classifier.streamlit.app

## ✨ Fitur

- Upload gambar (PNG/JPG/JPEG) dan dapatkan prediksi secara instan
- Pilihan 2 model yang bisa dibandingkan:
  - **Custom CNN** — dibangun dari nol (4 layer Conv2D + MaxPooling2D)
  - **Transfer Learning (MobileNetV2)** — memanfaatkan pretrained model
- Menampilkan confidence score dan perbandingan probabilitas Apple vs Orange
- Riwayat prediksi selama sesi berlangsung
- Tab informasi detail tentang model yang digunakan

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — framework web app
- [TensorFlow/Keras](https://www.tensorflow.org/) — deep learning
- NumPy & Pillow — pemrosesan gambar

## 📁 Struktur Project

```
Apple_Orange/
├── app.py                          # Aplikasi utama Streamlit
├── custom_cnn_apple_orange.h5      # Model Custom CNN
├── mobilenetv2_apple_orange.h5     # Model MobileNetV2
├── requirements.txt                # Dependencies
└── README.md
```

## 🚀 Menjalankan secara lokal

1. Clone repository ini:
```bash
git clone https://github.com/rajabhibhawa/Apple_Orange.git
cd Apple_Orange
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:
```bash
streamlit run app.py
```

4. Buka `http://localhost:8501` di browser.

## 🧠 Cara Kerja Model

Gambar yang diupload akan:
1. Di-resize ke ukuran 160x160 piksel
2. Dinormalisasi (nilai piksel 0-1)
3. Diproses oleh model yang dipilih
4. Menghasilkan skor probabilitas untuk kelas Apple dan Orange

Model menggunakan output biner (sigmoid), di mana skor mendekati 1 berarti Orange dan mendekati 0 berarti Apple.

## ⚠️ Batasan

Model ini hanya dilatih untuk mengenali **dua kelas**: apple dan orange. Gambar buah atau objek lain di luar kedua kelas ini akan tetap diklasifikasikan sebagai salah satu dari keduanya (dengan confidence yang biasanya rendah).

## 📌 Embed ke Website Lain

Aplikasi ini bisa disematkan ke halaman web lain menggunakan iframe:

```html
<iframe 
  src="https://appleorange-3t62pfpqm3ekhexhejpx6d.streamlit.app/?embed=true" 
  width="100%" 
  height="800" 
  style="border:none;">
</iframe>
```

## 📄 Lisensi

Bebas digunakan untuk keperluan pembelajaran.
