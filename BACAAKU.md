# LAPORAN PROYEK KELAS MACHINE LEARNING EXPERT - IDCAMP X DICODING - MUHAMAD JAFAR RAHADIAN
## **Memprediksi Harga Saham IDX Composite menggunakan Algoritma Neural Network**
<br>

## **Domain Proyek**

  Indeks Harga Saham Gabungan (IHSG) merupakan indikator sentral dalam menilai kinerja pasar modal Indonesia, mencerminkan kondisi ekonomi makro, sentimen investor, dan volatilitas pasar yang tinggi. Prediksi IHSG menjadi sangat penting untuk pengambilan keputusan investasi, manajemen risiko, dan kebijakan ekonomi nasional[[1]](https://jom.unsurya.ac.id/index.php/jimen/article/view/34). Beberapa studi menggarisbawahi keunggulan model deep learning untuk forecasting time series finansial seperti LSTM yang mampu mempelajari dependensi jangka panjang dan mengatasi masalah gradien menghilang pada sequence panjang[[2]](https://www.sciencedirect.com/science/article/pii/S0957417423008485), namun ada algoritma lain seperti CNN 1D yang berguna dalam mengekstraksi pola lokal melalui lapisan konvolusi untuk meningkatkan generalisasi model dan GRU menawarkan arsitektur lebih sederhana dengan jumlah parameter lebih sedikit namun kinerjanya sebanding dengan LSTM dalam berbagai kasus prediksi saham[[3]](https://www.mdpi.com/2227-7390/12/23/3738)[[4]](https://etasr.com/index.php/ETASR/article/view/9363/4459). 
  
  Oleh karena itu meskipun banyak studi pada indeks global (misalnya DAX, S&P500), aplikasi deep learning khusus untuk IHSG masih belum sebanyak seperti pada index global, sehingga proyek ini bertujuan untuk membantu menjembatani kesenjangan tersebut dengan menganalisis dan membandingkan performa model LSTM, CNN, dan GRU secara tunggal pada data historis IHSG atau dalam bahasa internasional disebut IDX Composite (kode: ^JKSE).

Daftar Pustaka: <br>
[1] [PENGARUH INFLASI DAN NILAI TUKAR/KURS TERHADAP INDEKS HARGA SAHAM GABUNGAN (IHSG) YANG TERDAFTAR DI BURSA EFEK INDONESIA(BEI) PADA MASA PANDEMI COVID-19 BULAN JANUARI-DESEMBER TAHUN 2020. (2021). Jurnal Inovatif Mahasiswa Manajemen, 1(2), 139-149.   https://doi.org/10.35968/ma9jyn97](https://jom.unsurya.ac.id/index.php/jimen/article/view/34)

[2] [Gülmez, B. (2023). Stock price prediction with optimized deep LSTM network with artificial rabbits optimization algorithm. Expert Systems with Applications, 227(April), 120346. https://doi.org/10.1016/j.eswa.2023.120346](https://www.sciencedirect.com/science/article/pii/S0957417423008485)

[3] [Ye, P., Zhang, H., & Zhou, X. (2024). CNN-CBAM-LSTM: Enhancing Stock Return Prediction Through Long and Short Information Mining in Stock Prediction. Mathematics, 12(23), 3738. https://doi.org/10.3390/math12233738](https://www.mdpi.com/2227-7390/12/23/3738)

[4] [Dwiandiyanta, B. Y., Hartanto, R., & Ferdiana, R. (2025). Optimization of Stock Predictions on Indonesia Stock Exchange: A New Hybrid Deep Learning Method. Engineering, Technology and Applied Science Research, 15(1), 19370–19379. https://doi.org/10.48084/etasr.9363](https://etasr.com/index.php/ETASR/article/view/9363/4459)

## **Business Understanding**

### **Problem Statements**:
- Pada saat ini jenis investasi banyak ragamnya salah satunya yaitu dibidang Saham, salah satu jenis saham yang sedang viral di indonesia saat ini ada IHSG karena terjadi penurunan terus menerus dan IHSG (^JKSE) juga memiliki tingkat volatilitas yang tinggi, sehingga **menyulitkan investor dan analis dalam memprediksi harga penutupan secara akurat**, yang berdampak pada pengambilan keputusan investasi.
- Di tengah tingginya volatilitas pasar saham, khususnya IHSG, kebutuhan akan sistem prediksi harga yang akurat menjadi semakin mendesak bagi pelaku industri keuangan dan investor untuk mengambil keputusan yang tepat dan mengurangi risiko kerugian. Namun, masih sangat minim riset yang secara spesifik **membandingkan efektivitas berbagai pendekatan Machine Learning, terutama model deep learning seperti LSTM, CNN, dan GRU dalam konteks prediksi harga penutupan IHSG**. Ketiadaan referensi yang kuat dan teruji di lapangan menyebabkan banyak perusahaan dan institusi keuangan belum dapat memanfaatkan potensi teknologi ini secara optimal untuk mendukung strategi investasi dan pengelolaan portofolio.

### **Goals**:
- Menghasilkan model prediksi IHSG yang mampu mengurangi nilai error (MSE, RMSE MAE, MAPE, dan $R^2$) dengan membandingkan model LSTM, CNN, dan GRU secara individual.
- Memberikan insight kuantitatif dan visual terhadap kemampuan masing-masing model dalam menangkap pola harga historis IHSG (^JKSE) dan memprediksi harga di masa depan.
  
### **Solution statements**:
- Mengimplementasikan dan melatih tiga model deep learning secara terpisah: LSTM, CNN, dan GRU menggunakan dataset historis harga penutupan IHSG (^JKSE).
- Mengukur performa setiap model menggunakan metrik kuantitatif yang objektif seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean absolute percentage error (MAPE), dan R-Squared ($R^2$).

*Untuk instalasi API, Framework, ataupun Library dapat dilakukan melalui file requirements.txt

## **Data Understanding**
Data yang saya gunakan merupakan dataset berjenis **TIME SERIES** dari data historis harian dari Indeks Harga Saham Gabungan (disingkat IHSG; dalam bahasa Inggris: Indonesia Composite Index, ICI, atau IDX Composite) dan memiliki kode [ticker](https://academy.binance.com/en/glossary/ticker-symbol) ^JKSE, berikut adalah detail dari dataset yang saya gunakan:

### **Sumber Data**
Data yang saya gunakan bersumber dari **WEBSITE YAHOO FINANCE** serta dapat diakses melalui tautan berikut: https://finance.yahoo.com/quote/%5EJKSE/ <br>

Untuk mendapatkan file dataset diperlukan API `yfinance` versi 0.2.54. kemudian dengan beberapa perintah seperti pada Notebook file dataset dengan format `jkse_historical.csv` disimpan ke folder `/content/` menggunakan library `csv` dan untuk selanjutnya dapat diolah pada proses-proses berikutnya.

### **Feature pada Dataset ^JKSE**
Berikut adalah tabel dari dataset `jkse_historical.csv`, tabel berikut tampil menggunakan perintah `maindf=pd.read_csv('/content/jkse_historical.csv')` dan perintah `maindf`.

|      |       date |        Open |        High |         Low |       Close |    Volume |
|-----:|-----------:|------------:|------------:|------------:|------------:|----------:|
|   0  | 1990-04-06 |  641.244019 |  641.244019 |  641.244019 |  641.244019 |         0 |
|   1  | 1990-04-09 |  633.457336 |  633.457336 |  633.457336 |  633.457336 |         0 |
|   2  | 1990-04-10 |  632.061340 |  632.061340 |  632.061340 |  632.061340 |         0 |
|   3  | 1990-04-11 |  634.668274 |  634.668274 |  634.668274 |  634.668274 |         0 |
|   4  | 1990-04-12 |  639.589111 |  639.589111 |  639.589111 |  639.589111 |         0 |
|  ... |        ... |         ... |         ... |         ... |         ... |       ... |
| 8534 | 2025-04-21 | 6450.313965 | 6472.538086 | 6406.803223 | 6445.966797 | 108855100 |
| 8535 | 2025-04-22 | 6455.079102 | 6538.321777 | 6428.104980 | 6538.266113 | 150312500 |
| 8536 | 2025-04-23 | 6589.327148 | 6642.915039 | 6588.256836 | 6634.376953 | 184488000 |
| 8537 | 2025-04-24 | 6671.067871 | 6697.900879 | 6585.456055 | 6613.478027 | 159090300 |
| 8538 | 2025-04-25 | 6660.619141 | 6683.360840 | 6640.777832 | 6678.915039 | 169030400 |

Berdasarkan pada tabel diatas dapat diketahui bahwa ada beberapa feature dari dataset `jkse_historical.csv` :
- date: Tanggal (dimulai dari tanggal 6 April 1990 hingga hari dimana data tersebut diambil, dalam kasus ini pada tanggal 25 April 2025)
- open: Harga Pembukaan pada Hari itu 
- high: Harga Tertinggi pada Hari itu
- low: Harga Terendah pada Hari itu
- close: Harga Penutupan pada Hari itu
- volume: Jumlah saham yang diperdagangkan pada hari itu

### **Jumlah Baris & Kolom**
Untuk dapat mengetahui jumlah baris dan kolom, anda dapat menggunakan perintah `maindf.shape` yang dimana pada studi kasus di hari data ini diambil maka akan menghasilkan output berupa `(8539, 6)`.<br>

Berdasarkan pada tuple `(8539, 6)`, secara keseluruhan data berjumlah 8539 dengan 6 Feature seperti yang sudah dijelaskan pada bagian **Feature pada Dataset ^JKSE**.
### **Informasi Tipe Data**
### **Statistik Deskriptif Data**
### **Kondisi Data (Missing Value, Outlier, dan Duplikat)**
### **Distribusi Feature Harga menggunakan Histogram**
### **Visualisasi Distribusi Harga menggunakan Density Plot**
### **Visualisasi Time Series pada Close Price**
### **Correlation Matrix dan Scatter Plots**

## **Data Preparation**
Berikut adalah beberapa tahapan yang dilakukan untuk merapihkan data agar data siap untuk diproses pada tahap modeling:
### **Penganganan Missing Value, Outlier, dan Duplikat**
### **Encoding Feature**
### **Split Dataset**
### **Standarisasi/Normalisasi Data**

## **Modeling**
Berikut adalah model yang saya gunakan beserta cara kerja dari model tersebut:
### **LSTM (Long Short Term Memory)**
### **CNN 1D (Convolutional Neural Network 1 Dimensi)**
### **GRU (Gated Recurrent Unit)**

## **Evaluasi**
### **Penjelasan Matriks Evaluasi MSE**
### **Penjelasan Matriks Evaluasi RMSE**
### **Penjelasan Matriks Evaluasi MAE**
### **Penjelasan Matriks Evaluasi MAPE**
### **Penjelasan Matriks Evaluasi $R^2$**
### **Evaluasi Model LSTM**
### **Evaluasi Model CNN 1D**
### **Evaluasi Model GRU**
### **Kesimpulan**
### **Percobaan Implementasi**
