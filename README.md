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
  | 8530 | 2025-04-14 | 6225.336914 | 6404.069824 | 6225.336914 | 6368.517090 | 149701700 |
  | 8531 | 2025-04-15 | 6444.341797 | 6497.532227 | 6395.926758 | 6441.683105 | 147079600 |
  | 8532 | 2025-04-16 | 6461.273926 | 6469.597168 | 6373.790039 | 6400.054199 | 142022700 |
  | 8533 | 2025-04-17 | 6407.020996 | 6438.269043 | 6384.285156 | 6438.269043 | 131124200 |
  | 8534 | 2025-04-21 | 6450.313965 | 6472.538086 | 6410.488770 | 6449.289062 |         0 |

  8534 rows × 6 columns

Data terlama ada pada tanggal 6 april 1990 dan data terbaru ada pada tanggal 21 april 2025 dimana program ini sedang dijalankan.

- Menampilkan informasi tipe data<br>
dengan menggunakan perintah `maindf_info()` hasil dari tipe data pada dataset adalah sebagai berikut:
  ```
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 8535 entries, 0 to 8534
  Data columns (total 6 columns):
   #   Column  Non-Null Count  Dtype  
  ---  ------  --------------  -----  
   0   date    8535 non-null   object 
   1   Open    8535 non-null   float64
   2   High    8535 non-null   float64
   3   Low     8535 non-null   float64
   4   Close   8535 non-null   float64
   5   Volume  8535 non-null   int64  
  dtypes: float64(4), int64(1), object(1)
  memory usage: 400.2+ KB
  ```
Penjelasannya adalah sebagai berikut:<br>
  -- Terdapat 1 kolom dengan tipe object, yaitu: date. Kolom ini merupakan categorical features (fitur non-numerik).<br>
  -- Terdapat 4 kolom numerik dengan tipe data float64 yaitu: open, high, low, dan close. Ini merupakan fitur numerik.<br>
  -- Terdapat 1 kolom numerik dengan tipe data int64, yaitu: volume.
  
- Menampilkan statistik deskriptif<br>

  |       |        Open |        High |         Low |       Close |       Volume |   
  |------:|------------:|------------:|------------:|------------:|-------------:|
  | count | 8535.000000 | 8535.000000 | 8535.000000 | 8535.000000 | 8.535000e+03 |  
  |  mean | 2814.285984 | 2829.873789 | 2796.542485 | 2814.085251 | 5.118739e+07 |  
  |  std  | 2474.594154 | 2486.571812 | 2460.743026 | 2473.515154 | 2.048066e+08 |  
  |  min  |  223.240311 |  223.240311 |  223.240311 |  223.240311 | 0.000000e+00 |  
  |  25%  |  504.601834 |  507.750229 |  500.262051 |  504.475342 | 2.724100e+06 |   
  |  50%  | 1781.983572 | 1792.152300 | 1770.499101 | 1783.909546 | 2.085050e+07 |   
  |  75%  | 5138.040906 | 5164.886486 | 5115.274943 | 5140.824951 | 5.377760e+07 |  
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

#### **Penanganan Missing Values & Outliers**
- penanganan Missing Values menggunakan Interpolasi<br>
  berdasarkan metode [ini](https://medium.com/@aseafaldean/time-series-data-interpolation-e4296664b86), berikut hasilnya:<br>

  ![image](https://github.com/user-attachments/assets/652e1646-aa59-4f44-82d2-738f7f1e1384)

Berdasarkan penggunaan metode interpolasi dataset yang digunakan tidak menunjukkan adanya Missing value, buktinya dapat dilihat pada gambar diatas. 

- Penanganan outliers menggunakan metode IQR<br>
Ditemukan outliers pada feature volume, berikut adalah bukti pembersihan outliers pada feature volume:<br>

  ![image](https://github.com/user-attachments/assets/5cd927d5-fd59-4ba1-b798-015ec657b3a6)

- Visualisasi outliers menggunakan boxplot<br>
  -- boxplot keseluruhan feature: <br>
        ![image](https://github.com/user-attachments/assets/dcbb982c-a69c-4e67-ab6b-8ec85d8c620b)
    <br>
    Seperti yang terlihat pada boxplot diatas **membuktikan** bahwasanya data pada seluruh(5) feature sudah bersih dan **TIDAK ADA OUTLIERS** karena sudah dilakukan proses pembersihan menggunakan IQR pada proses sebelumnya.

#### **Univariate Analysis**
- Distribusi fitur harga menggunakan histogram<br>
![image](https://github.com/user-attachments/assets/ec87a196-408e-4ae6-933a-d1f811c78053)

    Terlihat pada gambar diatas bahwa distribusi data pada keseluruhan(5) Feature terdapat perbedaan. Untuk open, high, low, close memiliki rentang nilai yang sama sedangkan volume memiliki nilai distribusi yang berbeda sendiri.<br>
    **KEPUTUSAN UNTUK DROP FEATURE VOLUME AKAN DILAKUKAN PADA PROSES-PROSES SETELAHNYA**

- Visualisasikan distribusi harga menggunakan Density Plot<br>
![image](https://github.com/user-attachments/assets/aa6fe4ab-db61-4613-a6f7-129ead24ba85)

    Sama seperti pada histogram, terlihat pada gambar diatas bahwa density data pada keseluruhan(ke-5) Feature terdapat perbedaan. Untuk open, high, low, close memiliki rentang nilai yang sama sedangkan volume memiliki nilai pola density yang berbeda sendiri.<br>
    **KEPUTUSAN UNTUK DROP FEATURE VOLUME AKAN DILAKUKAN PADA PROSES-PROSES SETELAHNYA**
  
- Visualisasi Time Series pada Close Price<br>
![image](https://github.com/user-attachments/assets/66e84a9a-b70f-4135-b7d6-74802a04988f)

    Gambar diatas adalah bentuk pola time series dari dataset IHSG sejak 1990 hingga hari ini, dapat dilihat juga pola serta fluktiasinya yang cukup tinggi.
  
- Deskripsi data yang sudah bersih<br>

  |       |        Open |        High |         Low |       Close |       Volume |
  |------:|------------:|------------:|------------:|------------:|-------------:|
  | count | 8535.000000 | 8535.000000 | 8535.000000 | 8535.000000 | 8.535000e+03 |
  |  mean | 2814.285984 | 2829.873789 | 2796.542485 | 2814.085251 | 3.796068e+07 |
  |  std  | 2474.594154 | 2486.571812 | 2460.743026 | 2473.515154 | 4.421105e+07 |
  |  min  |  223.240311 |  223.240311 |  223.240311 |  223.240311 | 0.000000e+00 |
  |  25%  |  504.601834 |  507.750229 |  500.262051 |  504.475342 | 2.724100e+06 |
  |  50%  | 1781.983572 | 1792.152300 | 1770.499101 | 1783.909546 | 2.085050e+07 |
  |  75%  | 5138.040906 | 5164.886486 | 5115.274943 | 5140.824951 | 5.377760e+07 |
  |  max  | 7904.395020 | 7910.556152 | 7853.353027 | 7905.390137 | 1.303578e+08 |

    Statistik Deskriptif diatas pada dasarnya sama seperti Statistik Deskriptif pada tahap **EDA - Deskripsi Variabel**, akan tetapi jumlah angkanya akan sedikit berubah karena sudah dilakukan proses penghilangan missing value dan penghilangan outliers.

#### **Multivariate Analysis**
Melakukan Correlation Matrix dan Scatter Plots untuk memeriksa korelasi antar feature.<br>

![image](https://github.com/user-attachments/assets/6404f053-7f5a-45f6-b6ed-60fc1ef9049d) <br>

![image](https://github.com/user-attachments/assets/28f5431d-f933-4443-b2d6-5c8d4c778b4d)

Pada Correlation Matrix diatas terlihat sangat amat jelas bahwa feature close, low, high, dan open memiliki perbedaan yang hampir **TIDAK ADA** sedangkan volume yang berbeda sendiri dengan angka yang jauh berbeda. Hal ini membuktikan bahwa korelasi antara feature volume dengan feature lainnya berbeda sehingga dapat di buang (**DROP**) di proses-proses selanjutnya.
Kesimpulan:
- **Karena Volume tidak memiliki korelasi yang kuat dengan data lainnya maka tidak dipilih**. (DROP)
- **Close yang akan dipilih karena dari keempat parameter lainnya hasilnya hampir sama**.

## **Data Preparation**
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
  | 8530 | 2025-04-14 | 6368.517090 |
  | 8531 | 2025-04-15 | 6441.683105 |
  | 8532 | 2025-04-16 | 6400.054199 |
  | 8533 | 2025-04-17 | 6438.269043 |
  | 8534 | 2025-04-21 | 6449.289062 |

  8535 rows × 2 columns

setelah itu untuk meringankan kerja model, akan dipilih 1000 data terbaru yang akan dijadikan untuk data testing dan data training, berikut adalah hasilnya:<br>

  |      |       date |       Close |
  |-----:|-----------:|------------:|
  | 7535 | 2021-02-24 | 6251.054199 |
  | 7536 | 2021-02-25 | 6289.645996 |
  | 7537 | 2021-02-26 | 6241.795898 |
  | 7538 | 2021-03-01 | 6338.513184 |
  | 7539 | 2021-03-02 | 6359.205078 |
  |  ... |        ... |         ... |
  | 8530 | 2025-04-14 | 6368.517090 |
  | 8531 | 2025-04-15 | 6441.683105 |
  | 8532 | 2025-04-16 | 6400.054199 |
  | 8533 | 2025-04-17 | 6438.269043 |
  | 8534 | 2025-04-21 | 6449.289062 |

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
| 7535 | 0.293215 |
| 7536 | 0.316286 |
| 7537 | 0.287680 |
| 7538 | 0.345500 |
| 7539 | 0.357871 |
|  ... |      ... |
| 8330 | 0.651338 |
| 8331 | 0.640258 |
| 8332 | 0.582430 |
| 8333 | 0.577699 |
| 8334 | 0.632939 |

800 rows × 1 columns

#### Standarisasi data Testing:

|      |    Close |
|-----:|---------:|
| 8335 | 0.669201 |
| 8336 | 0.674694 |
| 8337 | 0.670831 |
| 8338 | 0.684544 |
| 8339 | 0.721794 |
|  ... |      ... |
| 8530 | 0.363437 |
| 8531 | 0.407178 |
| 8532 | 0.382291 |
| 8533 | 0.405137 |
| 8534 | 0.411725 |

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

#### Plotting Loss dan Validasi Loss
![image](https://github.com/user-attachments/assets/33ef1a94-7ff1-4247-86c3-ceb1640b815b)

#### Kesimpulan
Berdasarkan Plotting Loss & Val_loss dapat diketahui bahwa algoritma LSTM mampu melakukan train dengan hyperparameter yang sudah ditentukan seperti diatas, untuk hasil Matriks Evaluasi model akan dilakukan pada step Evaluation. **PENENTUAN MODEL TIDAK DAPAT DILAKUKAN JIKA BELUM MENGETAHUI HASIL DARI EVALUASI MODEL**.<br>

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

#### Plotting Loss dan Validasi Loss
![image](https://github.com/user-attachments/assets/1e204541-1589-42c3-a2cd-6cb8c5db513f)


#### Kesimpulan
Berdasarkan Plotting Loss & Val_loss dapat diketahui bahwa algoritma CNN mampu melakukan train dengan hyperparameter yang sudah ditentukan seperti diatas, untuk hasil Matriks Evaluasi model akan dilakukan pada step Evaluation. **PENENTUAN MODEL TIDAK DAPAT DILAKUKAN JIKA BELUM MENGETAHUI HASIL DARI EVALUASI MODEL**.<br>

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

#### Plotting Loss dan Validasi Loss
![image](https://github.com/user-attachments/assets/f6e4e4e2-a74a-4931-9aa5-a504fe1f71cb)


#### Kesimpulan
Berdasarkan Plotting Loss & Val_loss dapat diketahui bahwa algoritma GRU mampu melakukan train dengan hyperparameter yang sudah ditentukan seperti diatas, untuk hasil Matriks Evaluasi model akan dilakukan pada step Evaluation. **PENENTUAN MODEL TIDAK DAPAT DILAKUKAN JIKA BELUM MENGETAHUI HASIL DARI EVALUASI MODEL**.<br>

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
```
Training Data Metrics LSTM:
MSE: 2492.08
RMSE: 49.92
MAE: 38.85
MAPE: 0.57%
R-squared: 0.98

Testing Data Metrics LSTM:
MSE: 7978.20
RMSE: 89.32
MAE: 66.26
MAPE: 0.95%
R-squared: 0.96
```

### CNN
```
Training Data Metrics CNN:
MSE: 3126.6025
RMSE: 55.9160
MAE: 43.6916
MAPE: 0.64%
R-squared: 0.9791

Testing Data Metrics CNN:
MSE: 11399.8232
RMSE: 106.7700
MAE: 83.3281
MAPE: 1.18%
R-squared: 0.9408
```

### GRU
```
Training Data Metrics GRU:
MSE: 2513.77
RMSE: 50.14
MAE: 38.85
MAPE: 0.57%
R-squared: 0.98

Testing Data Metrics GRU:
MSE: 8588.93
RMSE: 92.68
MAE: 70.07
MAPE: 1.00%
R-squared: 0.96
```

**KESIMPULAN**:

|   | Model |   MSE_Train | RMSE_Train | MAE_Train | MAPE_Train | R2_Train |     MSE_Test |  RMSE_Test |  MAE_Test | MAPE_Test |  R2_Test |
|--:|------:|------------:|-----------:|----------:|-----------:|---------:|-------------:|-----------:|----------:|----------:|---------:|
| 0 |  LSTM | 2492.075914 |  49.920696 | 38.845927 |   0.574054 | 0.983346 |  7978.197943 |  89.320759 | 66.263884 |  0.948856 | 0.958567 |
| 2 |   GRU | 2513.770906 |  50.137520 | 38.854945 |   0.574631 | 0.983201 |  8588.933387 |  92.676499 | 70.069656 |  0.999125 | 0.955395 |
| 1 |   CNN | 3126.602476 |  55.916031 | 43.691587 |   0.643636 | 0.979105 | 11399.823241 | 106.769955 | 83.328097 |  1.183736 | 0.940797 |

**Berdasarkan Hasil Evaluasi Model, LSTM merupakan Model Terbaik untuk studi kasus Time Series univariate.**

**ALASAN LSTM MENJADI MODEL TERBAIK KARENA**:
1. SKOR MSE UNTUK DATA TRAIN DAN DATA TEST MENJADI YANG TERKECIL SEPERTI PADA TABEL DIATAS,
2. LALU SKOR RMSE UNTUK KEDUA DATA JUGA MENJADI YANG TERKECIL DARI KEDUA MODEL LAINNYA,
3. LALU UNTUK MAE DATA TRAIN DAN DATA TEST LSTM UNGGUL DARI KEDUA MODEL LAINNYA DENGAN MENJADI MODEL DENGAN SKOR TERKECIL, DAN
4. UNTUK MAPE DARI KEDUA DATA LSTM UNGGUL DARI KEDUA MODEL LAINNYA,
5. UNTUK SKOR R2 LSTM LAH YANG PALING UNGGUL DARI KEDUA MODEL LAINNYA DENGAN SKOR R2 PALING MENDEKATI 1.

Maka dari itu Model **LSTM** yang akan dipilih untuk dijadikan implementasi.

## **Implementasi**
### Simpan model ke tf.lite
![image](https://github.com/user-attachments/assets/19018356-aecb-4f2f-828c-cccc7d14cc2a)

### Data Sebelum dan Sesudah Prediksi
![beforeafterihsg](https://github.com/user-attachments/assets/79844f41-43ee-49f0-a97d-2659551a11ba)

### Contoh penggunaan untuk prediksi 7 hari kedepan
![image](https://github.com/user-attachments/assets/7536eb61-a926-4b8b-93cb-dcae1e4777c8)

