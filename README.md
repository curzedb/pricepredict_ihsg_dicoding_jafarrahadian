# LAPORAN PROYEK KELAS MACHINE LEARNING EXPERT - IDCAMP X DICODING - MUHAMAD JAFAR RAHADIAN
## **Memprediksi Harga Saham IDX Composite menggunakan Algoritma Neural Network**
<br>

## A. **Domain Proyek**

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

Data yang digunakan merupakan data historis harian dari Indeks Harga Saham Gabungan (disingkat IHSG; dalam bahasa Inggris: Indonesia Composite Index, ICI, atau IDX Composite). Data tersebut dapat dilihat secara publik melalui website Yahoo Finance: https://finance.yahoo.com/quote/%5EJKSE/ 
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
a
Hasilnya:

![image](https://github.com/user-attachments/assets/da5e07e9-1e80-4b27-9182-aebbce4ffd9d)

### 2. Variabel-variabel pada dataset ^JKSE adalah sebagai berikut:
- date: Tanggal 
- open: Harga Pembukaan pada Hari itu
- high: Harga Tertinggi pada Hari itu
- low: Harga Terendah pada Hari itu
- close: Harga Penutupan pada Hari itu
- volume: saham yang diperdagangkan pada hari itu

### 3. Exploratory Data Analysis
EDA terdiri dari beberapa tahapan, diantaranya:

#### a. Deskripsi Variabel
- Data pertama dan terakhir<br>
import data jkse_historical.csv yang sudah di scapping kemudian import dengan menggunakan perintah
```python
maindf=pd.read_csv('/content/jkse_historical.csv')
maindf
```
maka hasilnya:

|	| date | Open | High | Low |	Close |	Volume |
|-|------|------|------|-----|--------|--------|
| 0 |	1990-04-06 | 641.244019 |	641.244019 | 641.244019 |	641.244019 | 0 |
| 1	| 1990-04-09 | 633.457336	| 633.457336 | 633.457336	| 633.457336 | 0 |
| 2 |	1990-04-10 | 632.061340 |	632.061340 | 632.061340 |	632.061340 | 0 |
| 3 |	1990-04-11 | 634.668274 |	634.668274 | 634.668274 |	634.668274 | 0 |
| 4 |	1990-04-12 | 639.589111	| 639.589111 | 639.589111 |	639.589111 | 0 |
| ... |	... |	... |	... |	... |	... |	... |
| 8529 | 2025-04-11 |	6195.567871 |	6298.777832 |	6148.775879 |	6262.226074 |	117642700 |
| 8530 | 2025-04-14 |	6225.336914 |	6404.069824 |	6225.336914 |	6368.517090 |	149701700 |
| 8531 | 2025-04-15 |	6444.341797 |	6497.532227 |	6395.926758 |	6441.683105 |	147079600 |
| 8532 | 2025-04-16 |	6461.273926 |	6469.597168 |	6373.790039 |	6400.054199 |	142022700 |
| 8533 | 2025-04-17 |	6407.020996 |	6438.269043 |	6384.285156 |	6438.269043 |	0 |

8534 rows × 6 columns

Data terlama ada pada tanggal 6 april 1990 dan data terbaru ada pada tanggal 17 april 2025 dimana program ini sedang dijalankan.

- Menampilkan informasi tipe data<br>
dengan menggunakan perintah `maindf_info()` hasil dari tipe data pada dataset adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/121a7c2b-e370-4b03-bc7f-99515a94f83c)

  
- Menampilkan statistik deskriptif<br>
Terdiri dari count, mean, std, min, 25%, 50%, 75%, dan max. Penjelasannya adalah sebagai berikut:<br>
-- Terdapat 1 kolom dengan tipe object, yaitu: date. Kolom ini merupakan categorical features (fitur non-numerik).<br>
-- Terdapat 4 kolom numerik dengan tipe data float64 yaitu: open, high, low, dan close. Ini merupakan fitur numerik.<br>
-- Terdapat 1 kolom numerik dengan tipe data int64, yaitu: volume.

#### b. Penanganan Missing Values & Outliers
- penanganan Missing Values menggunakan Interpolasi<br>
  berdasarkan metode [ini](https://medium.com/@aseafaldean/time-series-data-interpolation-e4296664b86), berikut hasilnya:<br>

![image](https://github.com/user-attachments/assets/652e1646-aa59-4f44-82d2-738f7f1e1384)

Berdasarkan penggunaan metode interpolasi dataset yang digunakan tidak menunjukkan adanya Missing value, buktinya dapat dilihat pada gambar diatas. 

- Penanganan outliers menggunakan metode IQR<br>
Ditemukan outliers pada feature volume, berikut adalah bukti pembersihan outliers pada feature volume:<br>

![image](https://github.com/user-attachments/assets/5cd927d5-fd59-4ba1-b798-015ec657b3a6)

- Visualisasi outliers menggunakan boxplot<br>
  -- boxplot open price setelah outlier handling: <br>

  ![image](https://github.com/user-attachments/assets/8f91fa59-1b4c-443a-819d-2c1e7f206e5a)

  -- boxplot close price setelah outlier handling:<br>

  ![image](https://github.com/user-attachments/assets/c8cce334-8d9c-4526-996c-c5948091da5b)
  

  -- boxplot high price setelah outlier handling:<br>
  
  ![image](https://github.com/user-attachments/assets/6235c7ad-036b-458e-81a4-c821d6837e26)


  -- boxplot low price setelah outlier handling:<br>

  ![image](https://github.com/user-attachments/assets/7241e495-a3dc-4a87-b303-eaa6566c8398)


  -- boxplot volume price setelah outlier handling:<br>

  ![image](https://github.com/user-attachments/assets/40fe2ff4-25ef-4052-8b1e-492e4a4231e1)


#### c. Univariate Analysis
- Distribusi fitur harga menggunakan histogram<br>
![image](https://github.com/user-attachments/assets/ec87a196-408e-4ae6-933a-d1f811c78053)

- Visualisasikan distribusi harga menggunakan Density Plot<br>
![image](https://github.com/user-attachments/assets/aa6fe4ab-db61-4613-a6f7-129ead24ba85)

- Visualisasi Time Series pada Close Price<br>
![image](https://github.com/user-attachments/assets/66e84a9a-b70f-4135-b7d6-74802a04988f)


- Deskripsi data yang sudah bersih<br>
![image](https://github.com/user-attachments/assets/775d78a7-50e7-4257-9115-b26656aa25ef)


#### d. Multivariate Analysis
Melakukan Correlation Matrix dan Scatter Plots untuk memeriksa korelasi antar feature.<br>

![image](https://github.com/user-attachments/assets/6404f053-7f5a-45f6-b6ed-60fc1ef9049d) <br>

![image](https://github.com/user-attachments/assets/28f5431d-f933-4443-b2d6-5c8d4c778b4d)

Kesimpulan:
- **Karena Volume tidak memiliki korelasi yang kuat dengan data lainnya maka tidak dipilih**. (DROP)
- **Close yang akan dipilih karena dari keempat parameter lainnya hasilnya hampir sama**.

## D. **Data Preparation**
### 1. Encoding Feature
Encoding Feature dilakukan dengan memilah Feature mana yang akan dipilih untuk dijadikan data training dan data testing, pada studi kasus kali ini Feature yang akan dipilih adalah **close** yang merupakan harga penutupan saham. Dipilihnya Feature **close** karena berdasarkan Correlation Matrix dan Scatter Plots feature **close** memiliki persamaan dengan 3 Feature numerik lainnya, sementara itu feature **date** akan dijadikan acuan tanggal dan feature **volume** akan dibuang(drop). Berikut hasilnya:<br>
![image](https://github.com/user-attachments/assets/3d085d91-d246-4ea4-a35d-1373eb755f99)

setelah itu untuk meringankan kerja model, akan dipilih 1000 data terbaru yang akan dijadikan untuk data testing dan data training, berikut adalah hasilnya:<br>
![image](https://github.com/user-attachments/assets/15fb9c5b-760c-4d82-9b18-fc70b418146d)


### 2. Split Dataset
Data kemudian akan dipisahkan menjadi data training dan data testing, dengan proporsi 80% untuk data pelatihan dan 20% untuk data pengujian, penggunaan konsep data pelatihan dan data pengujian menggunakan materi yang ada pada buku yang dikarang oleh [(Triayudi)](https://anyflip.com/tdezn/iggg/basic/151-200) pada halaman 153-155. berikut adalah hasilnya:<br>
![image](https://github.com/user-attachments/assets/8f5d0878-6208-4d99-ab7e-a98db6075c75)


### 3. Standarisasi
Setelah data dibagi kedalam data training dan data testing, selanjutnya akan dilakukan proses standarisasi menggunakan Min-Max Scaler. Standarisasi dilakukan untuk meminimalisir error, dengan diubahnya data menjadi nilai interval 0 dan 1 menggunakan rumus matematika. Berikut adalah hasilnya:
#### a. Standarisasi data Training: 
![image](https://github.com/user-attachments/assets/7d755477-2ba2-4ad5-9313-149a03227c06)


#### b. Standarisasi data Testing:
![image](https://github.com/user-attachments/assets/e517a453-b2c6-4e1b-832b-91f3b0859eae)

Hal yang perlu diperhatikan: Biasanya dilakukan proses PCA(Principal Component Analysis) untuk mereduksi dimensi pada studi kasus Multivariate Time Series, karena pada kali ini yang digunakan adalah Univariate Time Series maka PCA tidak diperlukan.<br>

## E. **Modeling**

### 1. Long-Short Term Memory (LSTM)
Algoritma LSTM digunakan karena kemampuannya dalam mengatasi vanishing & Loss Gradient, adapun kekurangan dan kelebihan dari Algoritma LSTM:
- Kekurangan:<br>
  -- Membutuhkan waktu pelatihan lebih lama dibanding GRU atau CNN karena kompleksitas strukturnya.<br>
  -- Jumlah unit LSTM, learning rate, dan ukuran window harus di-tune dengan hati-hati.<br>

- Kelebihan:<br>
  -- Forget gate membantu menghindari noise dalam data saham yang fluktuatif.<br>
  -- Dirancang khusus untuk data deret waktu, sehingga cocok untuk prediksi harga saham.<br>

#### a. Transformasi Data
Data dibuat untuk disesuaikan dengan model, sebagai berikut:
```python
import numpy as np

# konversikan array nilai menjadi matriks kumpulan data
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset['Close'][i:(i+time_step)].values
        dataX.append(a)
        # Use iloc to access data by position
        dataY.append(dataset['Close'].iloc[i + time_step])
    return np.array(dataX), np.array(dataY)
```
```python
time_step = 10
X_train_lstm, y_train_lstm = create_dataset(train_data_scaled, time_step)
X_test_lstm, y_test_lstm = create_dataset(test_data_scaled, time_step)

print("X_train: ", X_train_lstm.shape)
print("y_train: ", y_train_lstm.shape)
print("X_test: ", X_test_lstm.shape)
print("y_test", y_test_lstm.shape)
```
![image](https://github.com/user-attachments/assets/ac826e33-31de-40b2-9074-3a34dccf58ad)

```python
# membentuk ulang input menjadi [samples, time steps, features] yang diperlukan untuk LSTM
X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0],X_train_lstm.shape[1] , 1)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0],X_test_lstm.shape[1] , 1)

print("X_train: ", X_train_lstm.shape)
print("X_test: ", X_test_lstm.shape)
```
![image](https://github.com/user-attachments/assets/c1dd285b-be95-458b-8324-58dc75e116d1)

#### b. Bentuk Model
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
#### c. Hyperparameter Tuning
Berikut adalah Hyperparameter yang di atur sedemikian rupa agar model mendapatkan hasil terbaiknya:<br>
- Neuron: `100`
- Activation: `relu`
- Optimizer: `Adam`  
- Loss: `MSE`
- Model: `Sequential`  
- Epochs: `128`  
- Batch Size: `16`

#### D. Plotting Loss dan Validasi Loss
![image](https://github.com/user-attachments/assets/efc98012-04d6-476d-beaf-0dfd40b7d386)


#### E. Kesimpulan
Berdasarkan Plotting Loss & Val_loss dapat diketahui bahwa algoritma LSTM mampu melakukan train dengan hyperparameter yang sudah ditentukan seperti diatas, untuk hasil Matriks Evaluasi model akan dilakukan pada step Evaluation. **PENENTUAN MODEL TIDAK DAPAT DILAKUKAN JIKA BELUM MENGETAHUI HASIL DARI EVALUASI MODEL**.<br>

Jika sudah maka data dapat diubah kembali ke bentuk awal menggunakan proses denormalisasi dengan inverse Min-Max Scaler.

### 2. Convolutional Neural Network (CNN)
Algoritma CNN digunakan karena kecepatan komputasinya, adapun kekurangan dan kelebihan dari Algoritma CNN:
- Kekurangan:<br>
  -- CNN tidak memiliki memori internal karena butuh window-based approach untuk menghubungkan waktu.<br>
  -- Ukuran kernel harus disesuaikan dengan pola data.<br>
  
- Kelebihan:<br>
  -- Komputasinya cepat karena paralelisasi lebih baik dibanding RNN (LSTM & GRU : Turunan RNN).<br>
  -- Arsitektur fleksibel sehingga dapat dikombinasikan dengan pooling layers untuk ekstraksi fitur hierarkis.<br>
  
#### a. Transformasi Data
```python
# Create the dataset for CNN
def create_dataset_cnn(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset['Close'][i:(i + time_step)].values
        dataX.append(a)
        dataY.append(dataset['Close'].iloc[i + time_step])
    return np.array(dataX), np.array(dataY)

time_step = 10
X_train_cnn, y_train_cnn = create_dataset_cnn(train_data_scaled, time_step)
X_test_cnn, y_test_cnn = create_dataset_cnn(test_data_scaled, time_step)

# Reshape the input data for CNN (samples, time steps, features)
X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], X_train_cnn.shape[1], 1)
X_test_cnn = X_test_cnn.reshape(X_test_cnn.shape[0], X_test_cnn.shape[1], 1)
```
#### b. Bentuk Model
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
#### c. Hyperparameter Tuning
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

#### D. Plotting Loss dan Validasi Loss
![image](https://github.com/user-attachments/assets/781db13b-09a2-4dae-9495-1a4320287108)


#### E. Kesimpulan
Berdasarkan Plotting Loss & Val_loss dapat diketahui bahwa algoritma CNN mampu melakukan train dengan hyperparameter yang sudah ditentukan seperti diatas, untuk hasil Matriks Evaluasi model akan dilakukan pada step Evaluation. **PENENTUAN MODEL TIDAK DAPAT DILAKUKAN JIKA BELUM MENGETAHUI HASIL DARI EVALUASI MODEL**.<br>

Jika sudah maka data dapat diubah kembali ke bentuk awal menggunakan proses denormalisasi dengan inverse Min-Max Scaler.

### 3. Gated Recurrent Unit (GRU)
Algoritma GRU digunakan karena strukturnya lebih sederhana daripada LSTM namun performanya dapat sebanding dengan LSTM, adapun kekurangan dan kelebihan dari Algoritma GRU:
- Kekurangan:<br>
  -- Kurang optimal untuk menangkap tren jangka panjang karena keterbatasan Memori Jangka Panjang.<br>
  -- **Dokumentasi dan studi kasus lebih sedikit dibanding LSTM**.<br>

- Kelebihan:<br>
  -- Efisiensi Komputasi: Struktur lebih sederhana (2 gate vs 3 gate pada LSTM) sehingga pelatihan lebih cepat.<br>
  -- Kinerja pada Data Kecil: Lebih robust terhadap overfitting pada dataset terbatas.<br>
  -- Menangnai Pola Jangka Pendek: Efektif untuk prediksi harian/mingguan dengan fluktuasi cepat.<br>

#### a. Transformasi Data
```python
# Create the dataset for GRU
def create_dataset_gru(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset['Close'][i:(i + time_step)].values
        dataX.append(a)
        dataY.append(dataset['Close'].iloc[i + time_step])
    return np.array(dataX), np.array(dataY)

time_step = 10
X_train_gru, y_train_gru = create_dataset_gru(train_data_scaled, time_step)
X_test_gru, y_test_gru = create_dataset_gru(test_data_scaled, time_step)

# Reshape the input data for GRU (samples, time steps, features)
X_train_gru = X_train_gru.reshape(X_train_gru.shape[0],X_train_gru.shape[1] , 1)
X_test_gru = X_test_gru.reshape(X_test_gru.shape[0],X_test_gru.shape[1] , 1)
```

#### b. Bentuk Model
```python
# Define the GRU model
model_gru = Sequential()
model_gru.add(GRU(100, input_shape=(X_train_gru.shape[1], 1), activation='relu')) # Adjust units as needed
model_gru.add(Dense(1))
model_gru.compile(optimizer='adam', loss='mse')

# Train the GRU model
history_gru = model_gru.fit(X_train_gru, y_train_gru, epochs=128, batch_size=16, validation_data=(X_test_gru,y_test_gru), verbose=1)
```
#### c. Hyperparameter Tuning
- Neuron: `100`
- Activation: `relu`
- Optimizer: `Adam`  
- Loss: `MSE`
- Model: `Sequential`  
- Epochs: `128`  
- Batch Size: `16`

#### D. Plotting Loss dan Validasi Loss
![image](https://github.com/user-attachments/assets/1b5dee9e-0dd7-4fba-a7c9-fb0c7b5f268a)


#### E. Kesimpulan
Berdasarkan Plotting Loss & Val_loss dapat diketahui bahwa algoritma GRU mampu melakukan train dengan hyperparameter yang sudah ditentukan seperti diatas, untuk hasil Matriks Evaluasi model akan dilakukan pada step Evaluation. **PENENTUAN MODEL TIDAK DAPAT DILAKUKAN JIKA BELUM MENGETAHUI HASIL DARI EVALUASI MODEL**.<br>

Jika sudah maka data dapat diubah kembali ke bentuk awal menggunakan proses denormalisasi dengan inverse Min-Max Scaler.

## F. **Evaluation**
Matriks Evaluasi yang digunakan pada proyek ini diantaranya ada MSE, RMSE, MAE, MAPE, & R2. berikut adalah penjelasan dari setiap matriks evaluasi yang digunakan:<br>
### 1. MSE
MSE mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. Semakin tinggi nilai MSE maka semakin jauh prediksi model dari nilai aktual, yang berarti akurasi model menurun. Rumusnya: <br>
![image](https://github.com/user-attachments/assets/91a25ade-c763-49da-a7d4-9c7209d6031d)

### 2. RMSE
Root Mean Square Error atau disingkat RMSE merupakan hasil dari penjumlahan kuadrat error(Mean Square Error), perbedaan antar nilai asli dengan nilai prediksi akan dibagi dengan hasil penjumlahan yang akan diperoleh dari waktu peramalan. Semakin nilai RMSE mendekati nol, maka semakin baik kualitas hasil prediksi data tersebut, RMSE dirumuskan dengan: <br>
![image](https://github.com/user-attachments/assets/c3f737ab-a417-4237-876f-edea798322ba)

### 3. MAE
MAE mengukur rata-rata absolut selisih antara nilai prediksi dan nilai aktual. Nilai MAE yang tinggi menunjukkan adanya deviasi absolut yang besar, sehingga akurasi model menurun. Rumusnya: <br>
![image](https://github.com/user-attachments/assets/99cf2692-fb9e-4002-9460-6932a664c296)

### 4. MAPE
MAPE mengukur persentase kesalahan relatif terhadap nilai aktual. MAPE bernilai non‑negatif dan nilai terbaik adalah 0.0%, Semakin tinggi nilai MAPE, semakin besar persentase deviasi, yang berarti model kurang akurat. Rumusnya: <br>
![image](https://github.com/user-attachments/assets/eb27413c-9453-43e4-bf50-c80ee8062e1e)

### 5. R2
R-squared(R2) memiliki arti koefisien determinasi, merupakan ukuran uji statistik yang digunakan untuk menilai sejauh mana variabel yang tidak bergantung dalam model tersebut dapat menguraikan varian pada variabel independen. Nilai dari koefisien determinasi berada pada angka antara 0 dan 1, dimana jika nilai menunjukkan angka 1 atau semakin mendekati angka 1 maka hasil prediksi tersebut sepenuhnya cocok dengan data yang ada. Rumusnya: <br>
![image](https://github.com/user-attachments/assets/ac8e92a6-17f9-41ca-b2d5-ed0ff360e288)

Setelah mengetahui penjelasan dari masing-masing Matriks Evaluasi, selanjutnya adalah hasil evaluasi dari masing-masing model:<br>
### 1. LSTM
![image](https://github.com/user-attachments/assets/c16ca91f-630f-4286-8839-4a577a8661ce)

### 2. CNN
![image](https://github.com/user-attachments/assets/8a06252c-9955-4106-9c03-c339d02696a4)

### 3. GRU
![image](https://github.com/user-attachments/assets/3ffb113c-988d-466c-aa4c-5b3057d0e77b)

**KESIMPULAN**:
![image](https://github.com/user-attachments/assets/696ef75e-2b58-434c-9988-a90836313cc2)

Berdasarkan pada matriks evaluasi pada gambar tabel diatas, model yang menunjukkan performa terbaik untuk analisis Time Series studi kasus Univariate feature adalah LSTM dengan skor tipis dari GRU.<br>

**MAKA DARI ITU MODEL YANG AKAN DIPILIH ADALAH LSTM KARENA MEMILIKI PERFORMA TERBAIK BERDASARKAN HASIL MATRIKS EVALUASI PADA GAMBAR TABEL DIATAS**

## G. **Implementasi**
### 1. Simpan model ke tf.lite
![image](https://github.com/user-attachments/assets/19018356-aecb-4f2f-828c-cccc7d14cc2a)

### 2. Data Sebelum dan Sesudah Prediksi
![beforeafterihsg](https://github.com/user-attachments/assets/79844f41-43ee-49f0-a97d-2659551a11ba)

### 3. Contoh penggunaan untuk prediksi 7 hari kedepan
![image](https://github.com/user-attachments/assets/41fe83fc-c195-43de-b0d8-9727323a7855)
