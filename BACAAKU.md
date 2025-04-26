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

Untuk mendapatkan file dataset diperlukan API `yfinance` versi 0.2.54. kemudian dengan beberapa perintah seperti pada Notebook, file dataset dengan format `jkse_historical.csv` disimpan ke folder `/content/` menggunakan library `csv` dan untuk selanjutnya dapat diolah pada proses-proses berikutnya.

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
Untuk dapat mengetahui jumlah baris dan kolom, anda dapat menggunakan perintah `maindf.shape` yang dimana pada studi kasus di hari data ini diambil maka akan menghasilkan output berupa `(8539, 6)`. Berdasarkan pada tuple `(8539, 6)`, secara keseluruhan data berjumlah 8539 dengan 6 feature, untuk penjelasan setiap feature dapat anda baca pada bagian **Feature pada Dataset ^JKSE**.

### **Informasi Tipe Data**
Untuk dapat mengetahui informasi tipedata, anda dapat menggunakan `maindf.info()` dan akan menghasilkan output yaitu:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8539 entries, 0 to 8538
Data columns (total 6 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   date    8539 non-null   object 
 1   Open    8539 non-null   float64
 2   High    8539 non-null   float64
 3   Low     8539 non-null   float64
 4   Close   8539 non-null   float64
 5   Volume  8539 non-null   int64  
dtypes: float64(4), int64(1), object(1)
memory usage: 400.4+ KB
```
Berdasarkan pada output diatas dapat disimpulkan beberapa hal, seperti:
- Terdapat 1 kolom dengan tipe object, yaitu: date. Kolom ini merupakan categorical features (fitur non-numerik).<br>
- Terdapat 4 kolom numerik dengan tipe data float64 yaitu: open, high, low, dan close. Ini merupakan fitur numerik.<br>
- Terdapat 1 kolom numerik dengan tipe data int64, yaitu: volume.
- Keenam data tersebut tidak memiliki null.
- Ukuran memory yang digunakan sekitar 400.4 KB

### **Statistik Deskriptif Data**
Untuk dapat mengetahui statistik deskriptif pada dataset, anda dapat menggunakan perintah `maindf.describe()` dan berikut hasilnya:

|       |        Open |        High |         Low |       Close |       Volume |
|------:|------------:|------------:|------------:|------------:|-------------:|
| count | 8539.000000 | 8539.000000 | 8539.000000 | 8539.000000 | 8.539000e+03 |
|  mean | 2816.056560 | 2831.658894 | 2798.305307 | 2815.865948 | 5.125379e+07 |
|  std  | 2475.366799 | 2487.356968 | 2461.514572 | 2474.303709 | 2.047740e+08 |
|  min  |  223.240311 |  223.240311 |  223.240311 |  223.240311 | 0.000000e+00 |
|  25%  |  504.779843 |  507.779735 |  500.437519 |  504.496353 | 2.729900e+06 |
|  50%  | 1784.227596 | 1794.452148 | 1770.622127 | 1784.933594 | 2.092240e+07 |
|  75%  | 5139.862047 | 5165.633861 | 5120.314961 | 5142.301025 | 5.386560e+07 |
|  max  | 7904.395020 | 7910.556152 | 7853.353027 | 7905.390137 | 9.788202e+09 |

Berdasarkan tabel diatas yang terdiri dari count, mean, std, min, 25%, 50%, 75%, dan max. Penjelasannya adalah sebagai berikut:
- count: Jumlah data valid (non-null).
- mean: Rata-rata nilai.
- std: Standar deviasi (ukuran sebaran data).
- min/max: Nilai minimum dan maksimum.
- quartil (25%, 50%, 75%): Batas distribusi data (median = 50%).

Kesimpulan statistik Deskriptif:
-  8.539 data untuk setiap kolom (Open, High, Low, Close, Volume). Artinya, dataset mencakup 8.539 periode perdagangan (dalam hari menurut frekuensi data).
-  Harga rata-rata IHSG berkisar di 2.800-an, dengan harga tertinggi (High) cenderung lebih tinggi daripada terendah (Low), sesuai ekspektasi pasar.
-  Volume perdagangan rata-rata 51,2 juta unit, tetapi perlu diwaspadai karena ada nilai ekstrem (menandakan outlier).
-  Volatilitas tinggi (std dev besar), menunjukkan fluktuasi signifikan dalam sejarah IHSG. Serta penyebaran pada feature volume sangat lebar (std dev 204 juta vs mean 51 juta), mengindikasikan adanya outlier.
-  Harga terendah IHSG (Open, High, Low, dan Close) pernah menyentuh harga sekitar 223. mungkin terjadi pada periode krisis.
-  Jarak Q1(25%) ke Q3(75%) ekstrem (504 ke 5.139), menunjukkan pertumbuhan jangka panjang atau inflasi harga saham.
-  Harga tertinggi IHSG pernah mencapai 7.904–7.910.

### **Kondisi Data (Missing Value, Duplikat, dan Outlier)**
Berikut adalah beberapa kondisi data yang dapat dilihat baik dalam bentuk tabel maupun gambar:
- Missing Value<br>
  Dengan menggunakan perintah `maindf.isnull().sum()` tidak ditemukannya missing value pada dataset, berikut buktinya:

  |        | 0 |
  |-------:|--:|
  |  Open  | 0 |
  |  High  | 0 |
  |   Low  | 0 |
  |  Close | 0 |
  | Volume | 0 |
  
- Data Duplikat<br>
  Karena data time series itu berdasarkan pada rentang waktu, dan pada studi kasus ini pada rentang tanggal harian maka tidak akan ditemukannya duplikat data tanggal (saya menggunakan `['counts'] > 1` untuk memeriksa tanggal yang sama dengan jumlah lebih dari satu),   berikut buktinya:
  | date | counts |
  |-----:|-------:|
  |      |        |
  
- Outlier<br>

  ![image](https://github.com/user-attachments/assets/9933cf06-f31e-42d1-b7da-ac4388029bfb)

  Dapat dilihat pada boxplot diatas bahwa hanya pada **FEATURE VOLUME** terjadinya peristiwa outliers sedangkan pada feature lain aman tanpa terjadinya outliers. Outlier sendiri adalah observasi yang terletak pada jarak abnormal dari nilai lain dalam sampel acak dari suatu populasi dalam data. Boxplot diatas hanya berfungsi untuk menampilkan apakah ada **OUTLIERS** pada data **TANPA PENGAMBILAN KEPUTUSAN APAPUN**, untuk data cleansing akan dilakukan pada tahapan **DATA PREPARATION**.
  
### **Distribusi Feature Harga menggunakan Histogram**
Terlihat pada gambar berikut bahwa distribusi data pada keseluruhan(5) feature terdapat perbedaan. Untuk open, high, low, close memiliki rentang nilai yang sama sedangkan volume memiliki nilai distribusi yang berbeda sendiri (Menandakan tingginya outliers dan sudah terbukti saat menggunakan boxplot pada tahapan sebelumnya).
<br>

![image](https://github.com/user-attachments/assets/a144e813-70ae-4d78-adaa-a97cac459dd6)

### **Visualisasi Distribusi Harga menggunakan Density Plot**
Sama seperti pada histogram, terlihat pada gambar berikut bahwa density data pada keseluruhan(5) Feature terdapat perbedaan. Untuk open, high, low, close memiliki rentang nilai yang sama sedangkan volume memiliki nilai pola density yang berbeda sendiri.
<br>

![image](https://github.com/user-attachments/assets/c1d4ed07-655f-4957-bbef-7727574fe4d0)

### **Visualisasi Time Series pada Close Price**
Gambar berikut adalah bentuk pola time series dari dataset IHSG sejak 1990 hingga hari ini, dapat dilihat juga pola serta fluktiasinya yang cukup tinggi.
<br>

![image](https://github.com/user-attachments/assets/257fef97-36ae-4fa3-8f75-5aedb6667ef6)

### **Correlation Matrix dan Scatter Plots**
- Correlation Matrix<br>
  Pada Correlation Matrix berikut terlihat sangat amat jelas bahwa feature close, low, high, dan open memiliki perbedaan yang hampir **TIDAK ADA** sedangkan volume yang berbeda sendiri dengan angka yang jauh berbeda. Hal ini membuktikan bahwa korelasi antara feature volume dengan feature lainnya berbeda sehingga dapat di buang (**DROP**) di proses-proses selanjutnya.

  ![image](https://github.com/user-attachments/assets/d9a81f6e-a3c3-4035-98aa-b0e95d69e914)

- Scatter Plots<br>
  ![image](https://github.com/user-attachments/assets/26ac6517-67de-4d12-a389-98f23b6a5fde)

Berdasarkan kedua plot tersebut, dapat ditarik kesimpulan bahwa **Karena Volume tidak memiliki korelasi yang kuat dengan data lainnya maka tidak dipilih**, kemudian **Close yang akan dipilih karena dari keempat parameter lainnya hasilnya hampir sama**.

## **Data Preparation**
Berikut adalah beberapa tahapan yang dilakukan untuk merapihkan data agar data siap untuk diproses pada tahap modeling:
### **Penganganan Missing Value, Outlier, dan Duplikat**
- Missing Value<br>
  Dengan menggunakan metode [interpolasi](https://medium.com/@aseafaldean/time-series-data-interpolation-e4296664b86), tidak ditemukan adanya missing value pada dataset. Maka dari itu dapat langsung ke tahap data cleansing selanjutnya, **BUKTINYA DAPAT ANDA LIHAT PADA TAHAPAN "Kondisi Data (Missing Value, Duplikat, dan Outlier)"**
  
- Outlier
  Dengan menggunakan metore [Interquartil Range (IQR)](https://www.stat.cmu.edu/~hseltman/309/Book/Book.pdf), outlier pada feature volume dapat diselesaikan (**ANDA DAPAT MELIHAT BUKTI FEATURE VOLUME TERDAPAT OUTLIER PADA TAHAP "Kondisi Data (Missing Value, Duplikat, dan Outlier)"**). Berikut setelah penanganannya:<br>
  ![image](https://github.com/user-attachments/assets/28742ecd-0135-4d6b-908a-1c00639d8ff2)

- Duplikat
  Berdasarkan pengecekan pada tahapan **Kondisi Data (Missing Value, Duplikat, dan Outlier)**, tidak ditemukannya ada data duplikat maka dari itu data yang sudah bersih (`maindf_cleaned`) dapat diproses pada tahapan selanjutnya.
  
### **Encoding Feature**
Berdasarkan proses **Correlation Matrix dan Scatter Plots**, feature yang akan dipilih dan dijadikan untuk testing serta training adalah feature close. **ANDA BISA MEMBACA LAGI PADA TAHAPAN **Correlation Matrix dan Scatter Plots** KENAPA FEATURE CLOSE YANG DIPILIH**, kemudian 1000 data terbaru akan dipilih untuk data test dan data train. Kenapa hanya 1000 data terbaru yang dipilih untuk training dan testing?? Saya hanya memilih 1000 data terbaru agar kinerja model tetap optimal dan tidak terbebani. Hal itu disebabkan karena beberapa model yang saya gunakan tidak mampu membaca data time series yang sangat panjang. Berikut adalah data hasil Encoding Feature **TANPA BERMAKSUD** untuk melakukan visualisasi data, menampilkan data, dan eksplorasi pemahamaan data lainya seperti pada catatan review:

|      |       date |       Close |
|-----:|-----------:|------------:|
| 7522 | 2021-03-02 | 6359.205078 |
| 7523 | 2021-03-03 | 6376.756836 |
| 7524 | 2021-03-04 | 6290.798828 |
| 7525 | 2021-03-05 | 6258.749023 |
| 7526 | 2021-03-08 | 6248.464844 |
|  ... |        ... |         ... |
| 8517 | 2025-04-21 | 6445.966797 |
| 8518 | 2025-04-22 | 6538.266113 |
| 8519 | 2025-04-23 | 6634.376953 |
| 8520 | 2025-04-24 | 6613.478027 |
| 8521 | 2025-04-25 | 6678.915039 |

### **Split Dataset**
Data kemudian akan dipisahkan menjadi data training dan data testing, dengan proporsi 80% untuk data pelatihan dan 20% untuk data pengujian, penggunaan konsep data pelatihan dan data pengujian menggunakan materi yang ada pada buku yang dikarang oleh [(Triayudi)](https://anyflip.com/tdezn/iggg/basic/151-200) pada halaman 153-155. Berikut adalah hasilnya **TANPA BERMAKSUD** untuk melakukan visualisasi data, menampilkan data, dan eksplorasi pemahamaan data lainya seperti pada catatan review:<br>
```
Training data shape: (800, 1)
Testing data shape: (200, 1)
```

### **Standarisasi/Normalisasi Data**
Setelah data dibagi kedalam data training dan data testing, selanjutnya akan dilakukan proses standarisasi menggunakan Min-Max Scaler. Standarisasi dilakukan untuk meminimalisir error, dengan diubahnya data menjadi nilai interval 0 dan 1 menggunakan rumus matematika. Berikut adalah hasilnya **TANPA BERMAKSUD** untuk melakukan visualisasi data, menampilkan data, dan eksplorasi pemahamaan data lainya seperti pada catatan review:
```
Scaled Training data shape: (800, 1)
Scaled Testing data shape: (200, 1)
```

### **Ubah Matriks Data**
Mengubah data kedalam matriks 3 dimensi (`[samples, time steps, features]`) pada setiap model, mengambil dataset (dengan kolom 'Close') dan parameter time_step (jumlah lag), lalu melalui looping, ia membuat input sequence (dataX) yang berisi sekuens nilai sepanjang time_step dan output/target (dataY) yang berisi nilai berikutnya setelah sequence tersebut. Hasilnya adalah dua array numpy: dataX (berisi kumpulan sequences untuk training) dan dataY (berisi nilai target yang sesuai), di mana struktur ini memungkinkan model untuk mempelajari pola temporal. Berikut adalah hasilnya **TANPA BERMAKSUD** untuk melakukan visualisasi data, menampilkan data, dan eksplorasi pemahamaan data lainya seperti pada catatan review:

![image](https://github.com/user-attachments/assets/dc9f562c-ab99-4a75-a744-bac9f94e211d)


## **Modeling**
Berikut adalah model yang saya gunakan beserta cara kerja dari model tersebut:

### **LSTM (Long Short Term Memory)**
Model LSTM bekerja dengan cara memanfaatkan tiga gerbang (input, forget, dan output) untuk mengontrol aliran informasi dalam sel memornya, memungkinkannya mempelajari dependensi jangka panjang dalam data deret waktu seperti harga saham. Untuk memprediksi harga indeks ^JKSE (IHSG) menggunakan hanya data close, model bekerja dengan memproses data deret waktu secara berurutan. Dengan menggunakan mekanisme gerbang (input, forget, output) untuk memilih informasi mana yang disimpan atau dibuang, sehingga mampu mengingat pola jangka panjang seperti tren harian dalam pergerakan harga.<br>

Berikut Hyperparameter yang digunakan pada model LSTM:
- Neuron: `100`
- Activation: `relu`
- Optimizer: `Adam`  
- Loss: `MSE`
- Model: `Sequential`  
- Epochs: `128`  
- Batch Size: `16`

Berikut adalah kekurangan dan kelebihan model LSTM:
- Kekurangan:<br>
  -- Membutuhkan waktu pelatihan lebih lama dibanding GRU atau CNN karena kompleksitas strukturnya.<br>
  -- Jumlah unit LSTM, learning rate, dan ukuran window harus di-tune dengan hati-hati.<br>

- Kelebihan:<br>
  -- Forget gate membantu menghindari noise dalam data saham yang fluktuatif.<br>
  -- Dirancang khusus untuk data deret waktu, sehingga cocok untuk prediksi harga saham.<br>

Kesimpulannya adalah ...

### **CNN 1D (Convolutional Neural Network 1 Dimensi)**
Model GRU (Gated Recurrent Unit) adalah varian yang lebih sederhana dari LSTM dengan hanya dua gerbang (update dan reset), yang menggabungkan fungsi input dan forget gate menjadi satu, tetap mampu menangkap pola temporal tetapi dengan komputasi lebih efisien, cocok untuk mengenali pola pergerakan harga harian tanpa redundansi.<br>

Berikut Hyperparameter yang digunakan pada model CNN 1D:
- Layer: `64`
- MaxPool: `2`
- Neuron: `50`
- Activation: `relu`
- Optimizer: `Adam`  
- Loss: `MSE`
- Model: `Sequential`  
- Epochs: `128`  
- Batch Size: `32`

Berikut adalah kekurangan dan kelebihan dari Model CNN 1D:
- Kekurangan:<br>
  -- CNN tidak memiliki memori internal karena butuh window-based approach untuk menghubungkan waktu.<br>
  -- Ukuran kernel harus disesuaikan dengan pola data.<br>
  
- Kelebihan:<br>
  -- Komputasinya cepat karena paralelisasi lebih baik dibanding RNN (LSTM & GRU : Turunan RNN).<br>
  -- Arsitektur fleksibel sehingga dapat dikombinasikan dengan pooling layers untuk ekstraksi fitur hierarkis.<br>

Kesimpulannya adalah ...

### **GRU (Gated Recurrent Unit)**
Model CNN 1D mengaplikasikan filter konvolusi pada data `close` untuk mendeteksi pola lokal (misalnya, kenaikan/penurunan 5 hari berturut-turut) melalui operasi sliding window, lalu hasil ekstraksi fiturnya digunakan untuk prediksi. Model CNN 1D lebih cepat tetapi terbatas pada pola jangka pendek.<br>

Berikut Hyperparameter yang digunakan pada model GRU:
- Neuron: `100`
- Activation: `relu`
- Optimizer: `Adam`  
- Loss: `MSE`
- Model: `Sequential`  
- Epochs: `128`  
- Batch Size: `16`

Berikut adalah kekurangan dan kelebihan dari Model GRU:
- Kekurangan:<br>
  -- Kurang optimal untuk menangkap tren jangka panjang karena keterbatasan Memori Jangka Panjang.<br>
  -- **Dokumentasi dan studi kasus lebih sedikit dibanding LSTM**.<br>

- Kelebihan:<br>
  -- Efisiensi Komputasi: Struktur lebih sederhana (2 gate vs 3 gate pada LSTM) sehingga pelatihan lebih cepat.<br>
  -- Kinerja pada Data Kecil: Lebih robust terhadap overfitting pada dataset terbatas.<br>
  -- Menangnai Pola Jangka Pendek: Efektif untuk prediksi harian/mingguan dengan fluktuasi cepat.<br>

Kesimpulannya adalah ...

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
