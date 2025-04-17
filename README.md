# LAPORAN PROYEK KELAS MACHINE LEARNING EXPERT - IDCAMP X DICODING - MUHAMAD JAFAR RAHADIAN
## **Memprediksi Harga Saham IDX Composite menggunakan Algoritma Neural Network**
<br>

## A. **Latar Belakang**

  Indeks Harga Saham Gabungan (IHSG) merupakan indikator sentral dalam menilai kinerja pasar modal Indonesia, mencerminkan kondisi ekonomi makro, sentimen investor, dan volatilitas pasar yang tinggi. Prediksi IHSG menjadi sangat penting untuk pengambilan keputusan investasi, manajemen risiko, dan kebijakan ekonomi nasional[[1]](https://jom.unsurya.ac.id/index.php/jimen/article/view/34). Beberapa studi menggarisbawahi keunggulan model deep learning untuk forecasting time series finansial seperti LSTM yang mampu mempelajari dependensi jangka panjang dan mengatasi masalah gradien menghilang pada sequence panjang[[2]](https://www.sciencedirect.com/science/article/pii/S0957417423008485), namun ada algoritma lain seperti CNN 1D yang berguna dalam mengekstraksi pola lokal melalui lapisan konvolusi untuk meningkatkan generalisasi model dan GRU menawarkan arsitektur lebih sederhana dengan jumlah parameter lebih sedikit namun kinerjanya sebanding dengan LSTM dalam berbagai kasus prediksi saham[[3]](https://www.mdpi.com/2227-7390/12/23/3738)[[4]](https://etasr.com/index.php/ETASR/article/view/9363/4459). 
  
  Oleh karena itu meskipun banyak studi pada indeks global (misalnya DAX, S&P500), aplikasi deep learning khusus untuk IHSG masih belum sebanyak seperti pada index global, sehingga proyek ini bertujuan untuk membantu menjembatani kesenjangan tersebut dengan menganalisis dan membandingkan performa model LSTM, CNN, dan GRU secara tunggal pada data historis IHSG atau dalam bahasa internasional disebut IDX Composite (kode: ^JKSE).

Daftar Pustaka: <br>
[1] [PENGARUH INFLASI DAN NILAI TUKAR/KURS TERHADAP INDEKS HARGA SAHAM GABUNGAN (IHSG) YANG TERDAFTAR DI BURSA EFEK INDONESIA(BEI) PADA MASA PANDEMI COVID-19 BULAN JANUARI-DESEMBER TAHUN 2020. (2021). Jurnal Inovatif Mahasiswa Manajemen, 1(2), 139-149.   https://doi.org/10.35968/ma9jyn97](https://jom.unsurya.ac.id/index.php/jimen/article/view/34)

[2] [Gülmez, B. (2023). Stock price prediction with optimized deep LSTM network with artificial rabbits optimization algorithm. Expert Systems with Applications, 227(April), 120346. https://doi.org/10.1016/j.eswa.2023.120346](https://www.sciencedirect.com/science/article/pii/S0957417423008485)

[3] [Ye, P., Zhang, H., & Zhou, X. (2024). CNN-CBAM-LSTM: Enhancing Stock Return Prediction Through Long and Short Information Mining in Stock Prediction. Mathematics, 12(23), 3738. https://doi.org/10.3390/math12233738](https://www.mdpi.com/2227-7390/12/23/3738)

[4] [Dwiandiyanta, B. Y., Hartanto, R., & Ferdiana, R. (2025). Optimization of Stock Predictions on Indonesia Stock Exchange: A New Hybrid Deep Learning Method. Engineering, Technology and Applied Science Research, 15(1), 19370–19379. https://doi.org/10.48084/etasr.9363](https://etasr.com/index.php/ETASR/article/view/9363/4459)

## B. **Business Understanding**


### 1. Problem Statements:
- Volatilitas IHSG (^JKSE) yang tinggi membutuhkan perbandingan akurasi dengan model lain.
- Belum banyak yang membahas komparasi model yang fokus pada penerapan model deep learning seperti LSTM, CNN, dan GRU secara individual untuk memprediksi harga penutupan IHSG (^JKSE).

### 2. Goals:
- Menghasilkan model prediksi IHSG yang mampu mengurangi nilai error (MSE, RMSE MAE, MAPE, dan R2) dengan membandingkan model LSTM, CNN, dan GRU secara individual.
- Memberikan insight kuantitatif dan visual terhadap kemampuan masing-masing model dalam menangkap pola harga historis IHSG (^JKSE) dan memprediksi harga di masa depan.
  
### 3. Solution statements:
- Mengimplementasikan dan melatih tiga model deep learning secara terpisah: LSTM, CNN, dan GRU menggunakan dataset historis harga penutupan IHSG (^JKSE).
- Mengukur performa setiap model menggunakan metrik kuantitatif yang objektif seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean absolute percentage error (MAPE), dan R-Squared (R2).

### Tambahan - Hardware:
- Processor: Ryzen 7 5700X
- Ram: 32GB DDR4
- GPU: RTX 3060 12GB GDDR6

### Tambahan - Software:
- OS: Windows 11 Home 64bit
- Notebook: Google Colab (CPU)
- Docker untuk Local Runtime GPU
- Untuk instalasi API, Framework, ataupun Library dapat dilakukan melalui file requirements.txt


## C. **Data Understanding**

### 1. Data Loading

Data yang digunakan merupakan data historis dari Indeks Harga Saham Gabungan (disingkat IHSG; dalam bahasa Inggris: Indonesia Composite Index, ICI, atau IDX Composite). Data tersebut dapat dilihat secara publik melalui website Yahoo Finance: https://finance.yahoo.com/quote/%5EJKSE/ 
<br>
Untuk mendapatkan data ^JKSE diperlukan API [**yfinance**](https://github.com/ranaroussi/yfinance) dan selanjutnya di convert ke file csv menggunakan Library [**csv**](https://docs.python.org/3/library/csv.html), prosesnya dapat dilihat seperti dibawah:

```python
def fetch_jkse_historical():
    """
    Mengambil seluruh data historis IHSG dari awal hingga hari ini.
    """
    ticker = yf.Ticker("^JKSE")

    # Ambil data maksimum yang tersedia (dari awal hingga hari ini)
    df = ticker.history(period="max", actions=False)

    # Reset index dan ubah format tanggal
    df = df.reset_index()
    df = df.rename(columns={'Date': 'date'})

    # Konversi tipe data datetime ke string (opsional)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    return df

if __name__ == "__main__":
    try:
        # Ambil data historis
        historical_data = fetch_jkse_historical()

        # Simpan ke CSV
        historical_data.to_csv("jkse_historical.csv", index=False)
        print(f"Data berhasil disimpan! Jumlah baris: {len(historical_data)}")

    except Exception as e:
        print(f"Terjadi error: {str(e)}")
```
Hasilnya:

![image](https://github.com/user-attachments/assets/da5e07e9-1e80-4b27-9182-aebbce4ffd9d)

### 2. Variabel-variabel pada dataset ^JKSE adalah sebagai berikut:
- date:
- open:
- high:
- low:
- close:
- volume:

** 3. Exploratory Data Analysis 

## D. **Data Preparation**
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## E. **Modeling**
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## F. **Evaluation**
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## G. **Implementasi**
