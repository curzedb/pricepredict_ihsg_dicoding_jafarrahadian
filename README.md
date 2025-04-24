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
- Pada saat ini jenis investasi banyak ragamnya salah satunya yaitu dibidang Saham, salah satu jenis saham yang sedang viral di indonesia saat ini ada IHSG karena terjadi penurunan terus menerus dan IHSG (^JKSE) juga memiliki tingkat volatilitas yang tinggi, sehingga menyulitkan investor dan analis dalam memprediksi harga penutupan secara akurat, yang berdampak pada pengambilan keputusan investasi.
- Di tengah tingginya volatilitas pasar saham, khususnya IHSG, kebutuhan akan sistem prediksi harga yang akurat menjadi semakin mendesak bagi pelaku industri keuangan dan investor untuk mengambil keputusan yang tepat dan mengurangi risiko kerugian. Namun, masih sangat minim riset yang secara spesifik membandingkan efektivitas berbagai pendekatan Machine Learning, terutama model deep learning seperti LSTM, CNN, dan GRU, dalam konteks prediksi harga penutupan IHSG. Ketiadaan referensi yang kuat dan teruji di lapangan menyebabkan banyak perusahaan dan institusi keuangan belum dapat memanfaatkan potensi teknologi ini secara optimal untuk mendukung strategi investasi dan pengelolaan portofolio.

### **Goals**:
- Menghasilkan model prediksi IHSG yang mampu mengurangi nilai error (MSE, RMSE MAE, MAPE, dan $R^2$) dengan membandingkan model LSTM, CNN, dan GRU secara individual.
- Memberikan insight kuantitatif dan visual terhadap kemampuan masing-masing model dalam menangkap pola harga historis IHSG (^JKSE) dan memprediksi harga di masa depan.
  
### **Solution statements**:
- Mengimplementasikan dan melatih tiga model deep learning secara terpisah: LSTM, CNN, dan GRU menggunakan dataset historis harga penutupan IHSG (^JKSE).
- Mengukur performa setiap model menggunakan metrik kuantitatif yang objektif seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean absolute percentage error (MAPE), dan R-Squared ($R^2$).

*Untuk instalasi API, Framework, ataupun Library dapat dilakukan melalui file requirements.txt

## **Data Understanding**

### **Data Loading**

Data yang digunakan merupakan data historis harian dari Indeks Harga Saham Gabungan (disingkat IHSG; dalam bahasa Inggris: Indonesia Composite Index, ICI, atau IDX Composite). Data tersebut dapat dilihat secara publik melalui website Yahoo Finance: https://finance.yahoo.com/quote/%5EJKSE/ 
<br>
Untuk mendapatkan data ticker ^JKSE diperlukan API [**yfinance**](https://github.com/ranaroussi/yfinance) dan selanjutnya di convert ke file csv menggunakan Library [**csv**](https://docs.python.org/3/library/csv.html), prosesnya dapat dilihat seperti dibawah:

  ```python
  def fetch_jkse_historical():
      """
      Mengambil seluruh data historis IHSG dari awal hingga hari ini.
      """
      ticker = yf.Ticker("^JKSE")
  
      # Ambil data maksimum yang tersedia (dari awal hingga hari ini)
      df = ticker.history(period="max", actions=False)
  ```

Hasilnya:

![image](https://github.com/user-attachments/assets/da5e07e9-1e80-4b27-9182-aebbce4ffd9d) <br>

Ticker tersebut dapat diganti berdasarkan jenis saham yang akan digunakan untuk dilatih datanya, yang dalam studi kasus ini menggunakan saham IHSG (^JKSE).

#### Variabel-variabel pada dataset ^JKSE adalah sebagai berikut:
- date: Tanggal 
- open: Harga Pembukaan pada Hari itu
- high: Harga Tertinggi pada Hari itu
- low: Harga Terendah pada Hari itu
- close: Harga Penutupan pada Hari itu
- volume: saham yang diperdagangkan pada hari itu

### **Exploratory Data Analysis**
EDA terdiri dari beberapa tahapan, diantaranya:

#### **Deskripsi Variabel**
- Data pertama dan terakhir<br>
import data jkse_historical.csv yang sudah di scapping kemudian import dengan menggunakan perintah
  ```python
  maindf=pd.read_csv('/content/jkse_historical.csv')
  maindf
  ```
maka hasilnya:

  |      |       date |        Open |        High |         Low |       Close |    Volume |
  |-----:|-----------:|------------:|------------:|------------:|------------:|----------:|
  |   0  | 1990-04-06 |  641.244019 |  641.244019 |  641.244019 |  641.244019 |         0 |
  |   1  | 1990-04-09 |  633.457336 |  633.457336 |  633.457336 |  633.457336 |         0 |
  |   2  | 1990-04-10 |  632.061340 |  632.061340 |  632.061340 |  632.061340 |         0 |
  |   3  | 1990-04-11 |  634.668274 |  634.668274 |  634.668274 |  634.668274 |         0 |
  |   4  | 1990-04-12 |  639.589111 |  639.589111 |  639.589111 |  639.589111 |         0 |
  |  ... |        ... |         ... |         ... |         ... |         ... |       ... |
  | 8533 | 2025-04-17 | 6407.020996 | 6438.269043 | 6384.285156 | 6438.269043 | 131124200 |
  | 8534 | 2025-04-21 | 6450.313965 | 6472.538086 | 6406.803223 | 6445.966797 | 108855100 |
  | 8535 | 2025-04-22 | 6455.079102 | 6538.321777 | 6428.104980 | 6538.266113 | 150312500 |
  | 8536 | 2025-04-23 | 6589.327148 | 6642.915039 | 6588.256836 | 6634.376953 | 184488000 |
  | 8537 | 2025-04-24 | 6671.067871 | 6697.900879 | 6585.456055 | 6613.478027 |         0 |

  8538 rows × 6 columns

Data terlama ada pada tanggal 6 april 1990 dan data terbaru ada pada tanggal 24 april 2025 dimana program ini sedang dijalankan.

- Menampilkan informasi tipe data<br>
dengan menggunakan perintah `maindf_info()` hasil dari tipe data pada dataset adalah sebagai berikut:
  ```
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 8538 entries, 0 to 8537
  Data columns (total 6 columns):
   #   Column  Non-Null Count  Dtype  
  ---  ------  --------------  -----  
   0   date    8538 non-null   object 
   1   Open    8538 non-null   float64
   2   High    8538 non-null   float64
   3   Low     8538 non-null   float64
   4   Close   8538 non-null   float64
   5   Volume  8538 non-null   int64  
  dtypes: float64(4), int64(1), object(1)
  memory usage: 400.3+ KB
  ```
Penjelasannya adalah sebagai berikut:<br>
  -- Terdapat 1 kolom dengan tipe object, yaitu: date. Kolom ini merupakan categorical features (fitur non-numerik).<br>
  -- Terdapat 4 kolom numerik dengan tipe data float64 yaitu: open, high, low, dan close. Ini merupakan fitur numerik.<br>
  -- Terdapat 1 kolom numerik dengan tipe data int64, yaitu: volume.
  
- Menampilkan statistik deskriptif<br>

  |       |        Open |        High |         Low |       Close |       Volume |
  |------:|------------:|------------:|------------:|------------:|-------------:|
  | count | 8538.000000 | 8538.000000 | 8538.000000 | 8538.000000 | 8.538000e+03 |
  |  mean | 2815.606271 | 2831.207769 | 2797.855263 | 2815.413494 | 5.122137e+07 |
  |  std  | 2475.162009 | 2487.153272 | 2461.307385 | 2474.095334 | 2.047795e+08 |
  |  min  |  223.240311 |  223.240311 |  223.240311 |  223.240311 | 0.000000e+00 |
  |  25%  |  504.756587 |  507.769979 |  500.358530 |  504.488350 | 2.727875e+06 |
  |  50%  | 1783.673543 | 1794.278099 | 1770.574081 | 1784.646057 | 2.087685e+07 |
  |  75%  | 5139.551470 | 5165.338076 | 5119.448730 | 5141.749268 | 5.383542e+07 |
  |  max  | 7904.395020 | 7910.556152 | 7853.353027 | 7905.390137 | 9.788202e+09 |   

Terdiri dari count, mean, std, min, 25%, 50%, 75%, dan max. Penjelasannya adalah sebagai berikut:<br>
  -- count: Jumlah data valid (non-null).<br>
  -- mean: Rata-rata nilai.<br>
  -- std: Standar deviasi (ukuran sebaran data).<br>
  -- min/max: Nilai minimum dan maksimum.<br>
  -- quartil (25%, 50%, 75%): Batas distribusi data (median = 50%).<br>

Kegunaan:<br>
  -- Analisis Awal: Cek sebaran data (apakah ada outlier atau nilai ekstrem).<br>
  -- Validasi Data: Bandingkan statistik dengan ekspektasi (misal: volume perdagangan tidak mungkin negatif).<br>
  -- Persiapan Pemodelan: Pahami karakteristik data sebelum normalisasi/transformasi.<br>

(Kolom non-numerik seperti date otomatis diabaikan.)

#### **Univariate Analysis**

- Visualisasi boxplot untuk melihat outliers dalam data
  ![image](https://github.com/user-attachments/assets/5cfc6d95-90c7-4341-98cf-2819e7b3cbe4)

  Boxplot diatas hanya berfungsi untuk menampilkan apakah ada **OUTLIERS** pada data **TANPA PENGAMBILAN KEPUTUSAN APAPUN**, untuk data cleansing akan dilakukan pada tahapan **DATA PREPARATION**.

  Outlier sendiri adalah observasi yang terletak pada jarak abnormal dari nilai lain dalam sampel acak dari suatu populasi dalam data.

  Dapat dilihat pada boxplot diatas bahwa hanya pada feature volume terjadinya peristiwa outliers sedangkan pada feature lain aman tanpa terjadinya outliers.

- Distribusi fitur harga menggunakan histogram<br>
  ![image](https://github.com/user-attachments/assets/65ee7bcf-0666-4b2e-843a-75930113ecb8)


    Terlihat pada gambar diatas bahwa distribusi data pada keseluruhan(5) Feature terdapat perbedaan. Untuk open, high, low, close memiliki rentang nilai yang sama sedangkan volume memiliki nilai distribusi yang berbeda sendiri.<br>
    **KEPUTUSAN UNTUK DROP FEATURE VOLUME AKAN DILAKUKAN PADA PROSES-PROSES SETELAHNYA**

- Visualisasikan distribusi harga menggunakan Density Plot<br>
  ![image](https://github.com/user-attachments/assets/96a10b52-980d-41b0-bfed-0d30fa3664c9)


    Sama seperti pada histogram, terlihat pada gambar diatas bahwa density data pada keseluruhan(ke-5) Feature terdapat perbedaan. Untuk open, high, low, close memiliki rentang nilai yang sama sedangkan volume memiliki nilai pola density yang berbeda sendiri.<br>
    **KEPUTUSAN UNTUK DROP FEATURE VOLUME AKAN DILAKUKAN PADA PROSES-PROSES SETELAHNYA**
  
- Visualisasi Time Series pada Close Price<br>
  ![image](https://github.com/user-attachments/assets/cb819fd8-9395-4a95-9b07-9c42296e9848)


    Gambar diatas adalah bentuk pola time series dari dataset IHSG sejak 1990 hingga hari ini, dapat dilihat juga pola serta fluktiasinya yang cukup tinggi.

#### **Multivariate Analysis**
Melakukan Correlation Matrix dan Scatter Plots untuk memeriksa korelasi antar feature.<br>

![image](https://github.com/user-attachments/assets/26aea0af-e310-422b-af56-bbedb43f13ed)
<br>
![image](https://github.com/user-attachments/assets/2c95fcc6-b47f-46e7-afee-2b1fc540f92c)


Pada Correlation Matrix diatas terlihat sangat amat jelas bahwa feature close, low, high, dan open memiliki perbedaan yang hampir **TIDAK ADA** sedangkan volume yang berbeda sendiri dengan angka yang jauh berbeda. Hal ini membuktikan bahwa korelasi antara feature volume dengan feature lainnya berbeda sehingga dapat di buang (**DROP**) di proses-proses selanjutnya.
Kesimpulan:
- **Karena Volume tidak memiliki korelasi yang kuat dengan data lainnya maka tidak dipilih**. (DROP)
- **Close yang akan dipilih karena dari keempat parameter lainnya hasilnya hampir sama**.

## **Data Preparation**

### **Penanganan Missing Values & Outliers**
- penanganan Missing Values menggunakan Interpolasi<br>
  berdasarkan metode [ini](https://medium.com/@aseafaldean/time-series-data-interpolation-e4296664b86), berikut hasilnya:<br>

  ![image](https://github.com/user-attachments/assets/652e1646-aa59-4f44-82d2-738f7f1e1384)

Berdasarkan penggunaan metode interpolasi dataset yang digunakan tidak menunjukkan adanya Missing value, buktinya dapat dilihat pada gambar diatas. 

- Penanganan outliers menggunakan metode IQR<br>
Ditemukan outliers pada feature volume, berikut adalah bukti pembersihan outliers pada feature volume:<br>

  ![image](https://github.com/user-attachments/assets/4eefbbc5-4aa7-4b73-bb45-45467847bce5)


- Visualisasi outliers menggunakan boxplot<br>
  -- boxplot keseluruhan feature: <br>
        ![image](https://github.com/user-attachments/assets/dcbb982c-a69c-4e67-ab6b-8ec85d8c620b)
    <br>
    Seperti yang terlihat pada boxplot diatas **membuktikan** bahwasanya data pada seluruh(5) feature sudah bersih dan **TIDAK ADA OUTLIERS** karena sudah dilakukan proses pembersihan menggunakan IQR pada proses sebelumnya.

### **Encoding Feature**
Encoding Feature dilakukan dengan memilah Feature mana yang akan dipilih untuk dijadikan data training dan data testing, pada studi kasus kali ini Feature yang akan dipilih adalah **close** yang merupakan harga penutupan saham. Dipilihnya Feature **close** karena berdasarkan Correlation Matrix dan Scatter Plots feature **close** memiliki persamaan dengan 3 Feature numerik lainnya, sementara itu feature **date** akan dijadikan acuan tanggal dan feature **volume** akan dibuang(drop). Berikut hasilnya:<br>

  |      |       date |       Close |
  |-----:|-----------:|------------:|
  |   0  | 1990-04-06 |  641.244019 |
  |   1  | 1990-04-09 |  633.457336 |
  |   2  | 1990-04-10 |  632.061340 |
  |   3  | 1990-04-11 |  634.668274 |
  |   4  | 1990-04-12 |  639.589111 |
  |  ... |        ... |         ... |
  | 8533 | 2025-04-17 | 6438.269043 |
  | 8534 | 2025-04-21 | 6445.966797 |
  | 8535 | 2025-04-22 | 6538.266113 |
  | 8536 | 2025-04-23 | 6634.376953 |
  | 8537 | 2025-04-24 | 6613.478027 |

  8538 rows × 2 columns

setelah itu untuk meringankan kerja model, akan dipilih 1000 data terbaru yang akan dijadikan untuk data testing dan data training, berikut adalah hasilnya:<br>

  |      |       date |       Close |
  |-----:|-----------:|------------:|
  | 7538 | 2021-03-01 | 6338.513184 |
  | 7539 | 2021-03-02 | 6359.205078 |
  | 7540 | 2021-03-03 | 6376.756836 |
  | 7541 | 2021-03-04 | 6290.798828 |
  | 7542 | 2021-03-05 | 6258.749023 |
  |  ... |        ... |         ... |
  | 8533 | 2025-04-17 | 6438.269043 |
  | 8534 | 2025-04-21 | 6445.966797 |
  | 8535 | 2025-04-22 | 6538.266113 |
  | 8536 | 2025-04-23 | 6634.376953 |
  | 8537 | 2025-04-24 | 6613.478027 |

  1000 rows × 2 columns

### **Split Dataset**
Data kemudian akan dipisahkan menjadi data training dan data testing, dengan proporsi 80% untuk data pelatihan dan 20% untuk data pengujian, penggunaan konsep data pelatihan dan data pengujian menggunakan materi yang ada pada buku yang dikarang oleh [(Triayudi)](https://anyflip.com/tdezn/iggg/basic/151-200) pada halaman 153-155. berikut adalah hasilnya:<br>
```
Training data shape: (800, 1)
Testing data shape: (200, 1)
```


### **Standarisasi**
Setelah data dibagi kedalam data training dan data testing, selanjutnya akan dilakukan proses standarisasi menggunakan Min-Max Scaler. Standarisasi dilakukan untuk meminimalisir error, dengan diubahnya data menjadi nilai interval 0 dan 1 menggunakan rumus matematika. Berikut adalah hasilnya:

```
Scaled Training data shape: (800, 1)
Scaled Testing data shape: (200, 1)
```

#### Standarisasi data Training: 

  |      |    Close |
  |-----:|---------:|
  | 7538 | 0.345500 |
  | 7539 | 0.357871 |
  | 7540 | 0.368363 |
  | 7541 | 0.316976 |
  | 7542 | 0.297815 |
  |  ... |      ... |
  | 8333 | 0.577699 |
  | 8334 | 0.632939 |
  | 8335 | 0.669201 |
  | 8336 | 0.674694 |
  | 8337 | 0.670831 |

  800 rows × 1 columns

#### Standarisasi data Testing:

  |      |    Close |
  |-----:|---------:|
  | 8338 | 0.684544 |
  | 8339 | 0.721794 |
  | 8340 | 0.778962 |
  | 8341 | 0.824425 |
  | 8342 | 0.815767 |
  |  ... |      ... |
  | 8533 | 0.405137 |
  | 8534 | 0.409739 |
  | 8535 | 0.464918 |
  | 8536 | 0.522375 |
  | 8537 | 0.509881 |

  200 rows × 1 columns

## **Modeling**

### Long-Short Term Memory (LSTM)
Algoritma LSTM digunakan karena kemampuannya dalam mengatasi vanishing & Loss Gradient, adapun kekurangan dan kelebihan dari Algoritma LSTM:
- Kekurangan:<br>
  -- Membutuhkan waktu pelatihan lebih lama dibanding GRU atau CNN karena kompleksitas strukturnya.<br>
  -- Jumlah unit LSTM, learning rate, dan ukuran window harus di-tune dengan hati-hati.<br>

- Kelebihan:<br>
  -- Forget gate membantu menghindari noise dalam data saham yang fluktuatif.<br>
  -- Dirancang khusus untuk data deret waktu, sehingga cocok untuk prediksi harga saham.<br>

#### Transformasi Data
Pertama-tama data diubah kedalam bentuk `[samples, time_steps, features]` menggunakan fungsi seperti yang ada didalam notebook.

#### Bentuk Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#menentukan model, neuron, loss, dan optimizer
model_lstm =Sequential()
model_lstm.add(LSTM(100,input_shape=(None,1),activation="relu"))
model_lstm.add(Dense(1))
model_lstm.compile(loss="mean_squared_error",optimizer="adam")
```
```python
history_lstm = model_lstm.fit(X_train_lstm,y_train_lstm,validation_data=(X_test_lstm,y_test_lstm),epochs=128,batch_size=16,verbose=1)
```
#### Hyperparameter Tuning
Berikut adalah Hyperparameter yang di atur sedemikian rupa agar model mendapatkan hasil terbaiknya:<br>
- Neuron: `100`
- Activation: `relu`
- Optimizer: `Adam`  
- Loss: `MSE`
- Model: `Sequential`  
- Epochs: `128`  
- Batch Size: `16`

#### Kesimpulan
Berdasarkan history training model, algoritma LSTM mampu melakukan train dengan hyperparameter yang sudah ditentukan seperti diatas, untuk hasil Matriks Evaluasi model akan dilakukan pada step Evaluation. **PENENTUAN MODEL TIDAK DAPAT DILAKUKAN JIKA BELUM MENGETAHUI HASIL DARI EVALUASI MODEL**.<br>

Jika sudah maka data dapat diubah kembali ke bentuk awal menggunakan proses denormalisasi dengan inverse Min-Max Scaler.

### Convolutional Neural Network (CNN)
Algoritma CNN digunakan karena kecepatan komputasinya, adapun kekurangan dan kelebihan dari Algoritma CNN:
- Kekurangan:<br>
  -- CNN tidak memiliki memori internal karena butuh window-based approach untuk menghubungkan waktu.<br>
  -- Ukuran kernel harus disesuaikan dengan pola data.<br>
  
- Kelebihan:<br>
  -- Komputasinya cepat karena paralelisasi lebih baik dibanding RNN (LSTM & GRU : Turunan RNN).<br>
  -- Arsitektur fleksibel sehingga dapat dikombinasikan dengan pooling layers untuk ekstraksi fitur hierarkis.<br>
  
#### Transformasi Data
Pertama-tama data diubah kedalam bentuk `[samples, time_steps, features]` menggunakan fungsi seperti yang ada didalam notebook.

#### Bentuk Model
```python
# Define the CNN model
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(1))  # Output layer for regression
model_cnn.compile(optimizer='adam', loss='mse')
model_cnn.summary()
```
```python
# Train the model
history_cnn = model_cnn.fit(X_train_cnn, y_train_cnn, epochs=128, batch_size=32, validation_data=(X_test_cnn, y_test_cnn), verbose=1)
```
#### Hyperparameter Tuning
Berikut adalah Hyperparameter yang di atur sedemikian rupa agar model mendapatkan hasil terbaiknya:<br>
- Layer: `64`
- MaxPool: `2`
- Neuron: `50`
- Activation: `relu`
- Optimizer: `Adam`  
- Loss: `MSE`
- Model: `Sequential`  
- Epochs: `128`  
- Batch Size: `32`

#### Kesimpulan
Berdasarkan history training model, dapat diketahui bahwa algoritma CNN mampu melakukan train dengan hyperparameter yang sudah ditentukan seperti diatas, untuk hasil Matriks Evaluasi model akan dilakukan pada step Evaluation. **PENENTUAN MODEL TIDAK DAPAT DILAKUKAN JIKA BELUM MENGETAHUI HASIL DARI EVALUASI MODEL**.<br>

Jika sudah maka data dapat diubah kembali ke bentuk awal menggunakan proses denormalisasi dengan inverse Min-Max Scaler.

### Gated Recurrent Unit (GRU)
Algoritma GRU digunakan karena strukturnya lebih sederhana daripada LSTM namun performanya dapat sebanding dengan LSTM, adapun kekurangan dan kelebihan dari Algoritma GRU:
- Kekurangan:<br>
  -- Kurang optimal untuk menangkap tren jangka panjang karena keterbatasan Memori Jangka Panjang.<br>
  -- **Dokumentasi dan studi kasus lebih sedikit dibanding LSTM**.<br>

- Kelebihan:<br>
  -- Efisiensi Komputasi: Struktur lebih sederhana (2 gate vs 3 gate pada LSTM) sehingga pelatihan lebih cepat.<br>
  -- Kinerja pada Data Kecil: Lebih robust terhadap overfitting pada dataset terbatas.<br>
  -- Menangnai Pola Jangka Pendek: Efektif untuk prediksi harian/mingguan dengan fluktuasi cepat.<br>

#### Transformasi Data
Pertama-tama data diubah kedalam bentuk `[samples, time_steps, features]` menggunakan fungsi seperti yang ada didalam notebook.

#### Bentuk Model
```python
# Define the GRU model
model_gru = Sequential()
model_gru.add(GRU(100, input_shape=(X_train_gru.shape[1], 1), activation='relu')) # Adjust units as needed
model_gru.add(Dense(1))
model_gru.compile(optimizer='adam', loss='mse')

# Train the GRU model
history_gru = model_gru.fit(X_train_gru, y_train_gru, epochs=128, batch_size=16, validation_data=(X_test_gru,y_test_gru), verbose=1)
```
#### Hyperparameter Tuning
- Neuron: `100`
- Activation: `relu`
- Optimizer: `Adam`  
- Loss: `MSE`
- Model: `Sequential`  
- Epochs: `128`  
- Batch Size: `16`

#### Kesimpulan
Berdasarkan history training model, dapat diketahui bahwa algoritma GRU mampu melakukan train dengan hyperparameter yang sudah ditentukan seperti diatas, untuk hasil Matriks Evaluasi model akan dilakukan pada step Evaluation. **PENENTUAN MODEL TIDAK DAPAT DILAKUKAN JIKA BELUM MENGETAHUI HASIL DARI EVALUASI MODEL**.<br>

Jika sudah maka data dapat diubah kembali ke bentuk awal menggunakan proses denormalisasi dengan inverse Min-Max Scaler.

## **Evaluation**
Matriks Evaluasi yang digunakan pada proyek ini diantaranya ada MSE, RMSE, MAE, MAPE, & R2. berikut adalah penjelasan dari setiap matriks evaluasi yang digunakan:<br>
### MSE
MSE mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. Semakin tinggi nilai MSE maka semakin jauh prediksi model dari nilai aktual, yang berarti akurasi model menurun. Rumusnya: <br>

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

### RMSE
Root Mean Square Error atau disingkat RMSE merupakan hasil dari penjumlahan kuadrat error(Mean Square Error), perbedaan antar nilai asli dengan nilai prediksi akan dibagi dengan hasil penjumlahan yang akan diperoleh dari waktu peramalan. Semakin nilai RMSE mendekati nol, maka semakin baik kualitas hasil prediksi data tersebut, RMSE dirumuskan dengan: <br>

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

### MAE
MAE mengukur rata-rata absolut selisih antara nilai prediksi dan nilai aktual. Nilai MAE yang tinggi menunjukkan adanya deviasi absolut yang besar, sehingga akurasi model menurun. Rumusnya: <br>

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

### MAPE
MAPE mengukur persentase kesalahan relatif terhadap nilai aktual. MAPE bernilai non‑negatif dan nilai terbaik adalah 0.0%, Semakin tinggi nilai MAPE, semakin besar persentase deviasi, yang berarti model kurang akurat. Rumusnya: <br>

$$
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

### $R^2$
R-squared($R^2$) memiliki arti koefisien determinasi, merupakan ukuran uji statistik yang digunakan untuk menilai sejauh mana variabel yang tidak bergantung dalam model tersebut dapat menguraikan varian pada variabel independen. Nilai dari koefisien determinasi berada pada angka antara 0 dan 1, dimana jika nilai menunjukkan angka 1 atau semakin mendekati angka 1 maka hasil prediksi tersebut sepenuhnya cocok dengan data yang ada. Rumusnya: <br>

$$ \large R^2 = 1- \dfrac{SS_{RES}}{SS_{TOT}} = 1 - \dfrac{\sum_i(y_i - \hat y_i)^2}{\sum_i(y_i - \overline y_i)^2} $$

Setelah mengetahui penjelasan dari masing-masing Matriks Evaluasi, selanjutnya adalah hasil evaluasi dari masing-masing model:<br>

### LSTM
#### Plotting Loss dan Validasi Loss model LSTM:
![image](https://github.com/user-attachments/assets/33ef1a94-7ff1-4247-86c3-ceb1640b815b)<br>
Berdasarkan pola kedua kurva tersebut dan menentukan titik optimal dimana validasi loss mulai menurun (indikasi [goodfit](https://www.dicoding.com/blog/overfitting-vs-underfitting-apa-bedanya/)) dengan training loss terus menurun.

#### Hasil Matriks Evaluasi model LSTM:
```
Training Data Metrics LSTM:
MSE: 2506.15
RMSE: 50.06
MAE: 38.88
MAPE: 0.57%
R-squared: 0.98

Testing Data Metrics LSTM:
MSE: 8201.13
RMSE: 90.56
MAE: 68.22
MAPE: 0.98%
R-squared: 0.96
```

### CNN
#### Plotting Loss dan Validasi Loss model CNN:
![image](https://github.com/user-attachments/assets/0b602c74-f2e4-4f35-b635-0fb46f15cc76)<br>
Berdasarkan pola kedua kurva tersebut dan menentukan titik optimal dimana validasi loss mulai menurun (indikasi [goodfit](https://www.dicoding.com/blog/overfitting-vs-underfitting-apa-bedanya/)) dengan training loss terus menurun.

#### Hasil Matriks Evaluasi model CNN:
```
Training Data Metrics CNN:
MSE: 2665.5495
RMSE: 51.6290
MAE: 38.9811
MAPE: 0.58%
R-squared: 0.9821

Testing Data Metrics CNN:
MSE: 10282.0030
RMSE: 101.4002
MAE: 74.7207
MAPE: 1.07%
R-squared: 0.9483
```

### GRU
#### Plotting Loss dan Validasi Loss model GRU:
![image](https://github.com/user-attachments/assets/75cb051f-066a-4b91-b481-c94e79d6025c)<br>
Berdasarkan pola kedua kurva tersebut dan menentukan titik optimal dimana validasi loss mulai menurun (indikasi [goodfit](https://www.dicoding.com/blog/overfitting-vs-underfitting-apa-bedanya/)) dengan training loss terus menurun.

#### Hasil Matriks Evaluasi model GRU:
```
Training Data Metrics GRU:
MSE: 2945.94
RMSE: 54.28
MAE: 42.82
MAPE: 0.63%
R-squared: 0.98

Testing Data Metrics GRU:
MSE: 11011.92
RMSE: 104.94
MAE: 83.44
MAPE: 1.18%
R-squared: 0.94
```

**KESIMPULAN**:

|   | Model |   MSE_Train | RMSE_Train | MAE_Train | MAPE_Train | R2_Train |     MSE_Test |  RMSE_Test |  MAE_Test | MAPE_Test |  R2_Test |
|--:|------:|------------:|-----------:|----------:|-----------:|---------:|-------------:|-----------:|----------:|----------:|---------:|
| 0 |  LSTM | 2506.152450 |  50.061487 | 38.880598 |   0.574729 | 0.983165 |  8201.131900 |  90.560101 | 68.222057 |  0.978184 | 0.958765 |
| 1 |   CNN | 2665.549489 |  51.628960 | 38.981074 |   0.577532 | 0.982094 | 10282.002998 | 101.400212 | 74.720732 |  1.070848 | 0.948302 |
| 2 |   GRU | 2945.942454 |  54.276537 | 42.820113 |   0.631529 | 0.980210 | 11011.920218 | 104.937697 | 83.443109 |  1.176651 | 0.944632 |

**Berdasarkan Hasil Evaluasi Model, LSTM merupakan Model Terbaik untuk studi kasus Time Series univariate.**

**ALASAN LSTM MENJADI MODEL TERBAIK KARENA**:
1. SKOR MSE UNTUK DATA TRAIN DAN DATA TEST MENJADI YANG TERKECIL SEPERTI PADA TABEL DIATAS,
2. LALU SKOR RMSE UNTUK KEDUA DATA JUGA MENJADI YANG TERKECIL DARI KEDUA MODEL LAINNYA,
3. LALU UNTUK MAE DATA TRAIN DAN DATA TEST LSTM UNGGUL DARI KEDUA MODEL LAINNYA DENGAN MENJADI MODEL DENGAN SKOR TERKECIL, 
4. UNTUK MAPE DARI KEDUA DATA LSTM UNGGUL DARI KEDUA MODEL LAINNYA, DAN
5. UNTUK SKOR $R^2$ LSTM LAH YANG PALING UNGGUL DARI KEDUA MODEL LAINNYA DENGAN SKOR $R^2$ PALING MENDEKATI 1.

Maka dari itu Model **LSTM** yang akan dipilih untuk dijadikan implementasi.

## **Implementasi**
### Simpan model ke tf.lite
![image](https://github.com/user-attachments/assets/19018356-aecb-4f2f-828c-cccc7d14cc2a)

### Data Sebelum dan Sesudah Prediksi
![hehe](https://github.com/user-attachments/assets/532725f9-6430-499d-8a32-029b1f42dfe5)

### Contoh penggunaan untuk prediksi 7 hari kedepan
![image](https://github.com/user-attachments/assets/3f9e2ffb-69fe-4a65-b64a-19042e2fe979)

--------------------------------------------------------
