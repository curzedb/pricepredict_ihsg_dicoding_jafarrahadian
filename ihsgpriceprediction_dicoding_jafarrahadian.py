# -*- coding: utf-8 -*-
"""IHSGPricePrediction_Dicoding_Jafarrahadian.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1s_QeN-Y_zhFVkz8XbdzfYHlQNEuidCAo

# A. Business Understanding

Proyek Predictive Analytics: Prediksi Harga IHSG (^JKSE)
Mengikuti Metode CRISP-DM 6 Fase

Problem Steatment:
- Volatilitas IHSG (^JKSE) yang tinggi membutuhkan perbandingan akurasi dengan model lain.
- Belum banyak yang membahas studi komparatif yang fokus pada penerapan model deep learning seperti LSTM, CNN, dan GRU secara individual untuk memprediksi harga penutupan IHSG.

Goals:
- Menghasilkan model prediksi IHSG yang mampu mengurangi nilai error (MSE, RMSE MAE, MAPE, dan R2) dengan membandingkan model LSTM, CNN, dan GRU secara individual.
- Memberikan insight kuantitatif dan visual terhadap kemampuan masing-masing model dalam menangkap pola harga historis IHSG dan memprediksi harga di masa depan.

Solution statements:
- Mengimplementasikan dan melatih tiga model deep learning secara terpisah: LSTM, CNN, dan GRU menggunakan dataset historis harga penutupan IHSG (^JKSE).
- Mengukur performa setiap model menggunakan metrik kuantitatif yang objektif seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean absolute percentage error (MAPE), dan R-Squared (R2).

Tambahan - Hardware:
- Processor: Ryzen 7 5700X
- Ram: 32GB DDR4
- GPU: RTX 3060 12GB GDDR6

Tambahan - Software:
- OS: Windows 11 Home 64bit
- Notebook: Google Colab (CPU)
- Docker untuk Local Runtime GPU
- Untuk instalasi API, Framework, ataupun Library dapat dilakukan melalui file requirements.txt

## 1. Instalasi Library yang dibutuhkan
"""

!pip install yfinance==0.2.54

"""## 2. Import Library yang akan digunakan"""

import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns

"""# B. Data Understanding

## 1. Mengambil data historis maksimal dari Yahoo Finance (Data Loading)
"""

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

"""## 2. EDA - Deskripsi Variabel"""

# Memuat dataset dari directory /Data
maindf=pd.read_csv('/content/jkse_historical.csv')

maindf #5 data terbaru dan Terlama

maindf.shape # Jumlah data dan parameter data pada dataset

maindf.info() # Informasi mengenai tipe data pada dataset

maindf.describe() # Informasi mengenai Statistik Deskriptif data pada dataset

"""## 3. EDA - Menangani Missing Value dan Outliers"""

# Memeriksa dan menghapus Missing Values
# Jika terdapat Missing Values jumlah data akan berubah

def handle_missing_values(df):
    print("\n=== Penanganan Missing Values ===")

    # Cek jumlah missing values
    print("\nMissing Values Sebelum Penanganan:")
    print(df.isnull().sum())

    # Handle missing values
    if df.isnull().sum().any():
        # Untuk data time series, gunakan interpolasi (https://medium.com/@aseafaldean/time-series-data-interpolation-e4296664b86)
        df_clean = df.interpolate(method='linear')
        print("\nMissing values diatasi dengan interpolasi linear")

        #Jika masih ada missing value setelah interpolasi
        if df_clean.isnull().sum().any():
          df_clean = df_clean.ffill().bfill()
          print("\nSisa Missing values diatasi dengan forward-fill dan back-fill")
    else:
        df_clean = df.copy()
        print("\nTidak ada missing values")

    # Verifikasi setelah penanganan
    print("\nMissing Values Setelah Penanganan:")
    print(df_clean.isnull().sum())

    return df_clean

# Contoh penggunaan
maindf_cleaned = handle_missing_values(maindf)

# Tampilkan data setelah penanganan missing values
maindf_cleaned

def handle_outliers(df, column):
    print(f"\n=== Penanganan Outliers pada kolom '{column}' ===")

    # Metode IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Jumlah outliers: {len(outliers)}")

    # Winsorizing (Batas atas/bawah diganti dengan batas IQR) https://www.stat.cmu.edu/~hseltman/309/Book/Book.pdf halaman 70
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    print("Outliers ditangani dengan Winsorizing")

    return df

# Contoh penggunaan untuk beberapa kolom (sesuaikan dengan kolom yang ingin diproses)
columns_to_check = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in columns_to_check:
    maindf_cleaned = handle_outliers(maindf_cleaned, col)

# Tampilkan data setelah penanganan outlier
maindf_cleaned

# Visualisasi data 'Open' setelah penanganan outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=maindf_cleaned['Open'])
plt.title('Boxplot of Open Price After Outlier Handling')
plt.show()

# Visualisasi data 'Close' setelah penanganan outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=maindf_cleaned['Close'])
plt.title('Boxplot of Close Price After Outlier Handling')
plt.show()

# Visualisasi data 'High' setelah penanganan outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=maindf_cleaned['High'])
plt.title('Boxplot of High Price After Outlier Handling')
plt.show()

# Visualisasi data 'Low' setelah penanganan outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=maindf_cleaned['Low'])
plt.title('Boxplot of Low Price After Outlier Handling')
plt.show()

# Visualisasi data 'Volume' setelah penanganan outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=maindf_cleaned['Volume'])
plt.title('Boxplot of Volume After Outlier Handling')
plt.show()

"""## 4. EDA - Univariate Analysis"""

# Univariate Analysis - Histograms
plt.figure(figsize=(14, 7))

# Histogram 'Open'
plt.subplot(2, 3, 1)
plt.hist(maindf_cleaned['Open'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Open Prices')
plt.xlabel('Open Price')
plt.ylabel('Frequency')

# Histogram 'High'
plt.subplot(2, 3, 2)
plt.hist(maindf_cleaned['High'], bins=30, color='lightcoral', edgecolor='black')
plt.title('Distribution of High Prices')
plt.xlabel('High Price')
plt.ylabel('Frequency')

# Histogram 'Low'
plt.subplot(2, 3, 3)
plt.hist(maindf_cleaned['Low'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Distribution of Low Prices')
plt.xlabel('Low Price')
plt.ylabel('Frequency')

# Histogram 'Close'
plt.subplot(2, 3, 4)
plt.hist(maindf_cleaned['Close'], bins=30, color='gold', edgecolor='black')
plt.title('Distribution of Close Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')

# Histogram 'Volume'
plt.subplot(2, 3, 5)
plt.hist(maindf_cleaned['Volume'], bins=30, color='lightpink', edgecolor='black')
plt.title('Distribution of Volume')
plt.xlabel('Volume')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Visualisasi data 'Open', 'Close', 'High', 'Low', dan 'Volume' setelah penanganan outliers dalam boxplot terpisah
plt.figure(figsize=(14, 7))

plt.subplot(2, 3, 1)
sns.boxplot(x=maindf_cleaned['Open'])
plt.title('Boxplot Open Price')

plt.subplot(2, 3, 2)
sns.boxplot(x=maindf_cleaned['Close'])
plt.title('Boxplot Close Price')

plt.subplot(2, 3, 3)
sns.boxplot(x=maindf_cleaned['High'])
plt.title('Boxplot High Price')

plt.subplot(2, 3, 4)
sns.boxplot(x=maindf_cleaned['Low'])
plt.title('Boxplot Low Price')

plt.subplot(2, 3, 5)
sns.boxplot(x=maindf_cleaned['Volume'])
plt.title('Boxplot Volume')

plt.tight_layout()
plt.show()

# Density Plots
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
sns.kdeplot(maindf_cleaned['Open'], shade=True, color='skyblue')
plt.title('Density Plot of Open Prices')

plt.subplot(2, 3, 2)
sns.kdeplot(maindf_cleaned['High'], shade=True, color='lightcoral')
plt.title('Density Plot of High Prices')

plt.subplot(2, 3, 3)
sns.kdeplot(maindf_cleaned['Low'], shade=True, color='lightgreen')
plt.title('Density Plot of Low Prices')

plt.subplot(2, 3, 4)
sns.kdeplot(maindf_cleaned['Close'], shade=True, color='gold')
plt.title('Density Plot of Close Prices')

plt.subplot(2, 3, 5)
sns.kdeplot(maindf_cleaned['Volume'], shade=True, color='lightpink')
plt.title('Density Plot of Volume')

plt.tight_layout()
plt.show()

# Visualisasikan Close Price Time Series
maindf_cleaned['date'] = pd.to_datetime(maindf_cleaned['date'])

# Set 'date' as index
maindf_cleaned = maindf_cleaned.set_index('date')

# Resample to monthly frequency and take the mean of 'Close' price
monthly_data = maindf_cleaned['Close'].resample('M').mean()

# Create the plot
plt.figure(figsize=(14, 7))
plt.plot(monthly_data.index, monthly_data.values)
plt.xlabel("Date")
plt.ylabel("Average Monthly Close Price")
plt.title("Monthly Average Close Price of IHSG")
plt.grid(True)
plt.show()

# Descriptive Statistics
maindf_cleaned.describe()

"""## 5. EDA - Multivariate Analysis"""

# Correlation Matrix
correlation_matrix = maindf_cleaned[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Stock Prices')
plt.show()

# Scatter Plots
sns.pairplot(maindf_cleaned[['Open', 'High', 'Low', 'Close', 'Volume']])
plt.suptitle('Pairplot of Stock Prices', y=1.02)
plt.show()

"""Kesimpulan:
1. **Karena Volume tidak memiliki korelasi yang kuat dengan data lainnya maka tidak dipilih**. (DROP)
2. **Close yang akan dipilih karena dari keempat parameter lainnya hasilnya hampir sama**.

# C. Data Preparation

## 1. Encoding data dan pilih Feature Date dan Close
"""

# Select 'date' and 'Close' columns
close_stock = maindf_cleaned.copy()
# Reset the index to get 'date' back as a column
selected_data = maindf_cleaned.reset_index()[['date', 'Close']]
selected_data

# Select the first 1000 data points
maindf_cleaned_split = maindf_cleaned[-1000:]
maindf_cleaned_split

"""## 2. Bagi data kedalam data train (80%) dan data test (20%)"""

from sklearn.model_selection import train_test_split

# Select features (Close only)
selected_data = maindf_cleaned_split[['Close']]

# Split the data into training and testing sets (80:20)
train_data, test_data = train_test_split(selected_data, test_size=0.2, random_state=42, shuffle=False)  # Important: shuffle=False for time series data

# Now you have train_data and test_data
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

"""## 3. Standarisasi menggunakan metode Min-Max Scaler"""

from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the training data and transform both training and testing data
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# You can convert them back to dataframes if needed:
train_data_scaled = pd.DataFrame(train_data_scaled, columns=['Close'], index=train_data.index)
test_data_scaled = pd.DataFrame(test_data_scaled, columns=['Close'], index=test_data.index)

print("Scaled Training data shape:", train_data_scaled.shape)
print("Scaled Testing data shape:", test_data_scaled.shape)

train_data_scaled

test_data_scaled

"""## 4. Melakukan Reduksi dimensi dengan Principal Component Analysis (PCA)"""

'''
from sklearn.decomposition import PCA

# Inisialisasi PCA dengan jumlah komponen yang diinginkan (misalnya, 1 komponen)
pca = PCA(n_components=1)  # Sesuaikan jumlah komponen sesuai kebutuhan

# Fit PCA ke data training yang telah di-scaled
pca.fit(train_data_scaled)

# Transformasi data training dan testing menggunakan PCA
train_data_pca = pca.transform(train_data_scaled)
test_data_pca = pca.transform(test_data_scaled)

# Ubah kembali hasil PCA ke DataFrame (opsional)
train_data_pca = pd.DataFrame(data=train_data_pca, columns=['PC1'], index=train_data.index)
test_data_pca = pd.DataFrame(data=test_data_pca, columns=['PC1'], index=test_data.index)

print("PCA Training data shape:", train_data_pca.shape)
print("PCA Testing data shape:", test_data_pca.shape)
'''

"""### *PCA digunakan biasanya pada multivariate Time Series, sedangkan yang saya gunakan adalah Univariate Time Series

# D. Model Development

## 1. Long Short Term Memory (LSTM)
"""

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

time_step = 10
X_train_lstm, y_train_lstm = create_dataset(train_data_scaled, time_step)
X_test_lstm, y_test_lstm = create_dataset(test_data_scaled, time_step)

print("X_train: ", X_train_lstm.shape)
print("y_train: ", y_train_lstm.shape)
print("X_test: ", X_test_lstm.shape)
print("y_test", y_test_lstm.shape)

# membentuk ulang input menjadi [samples, time steps, features] yang diperlukan untuk LSTM
X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0],X_train_lstm.shape[1] , 1)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0],X_test_lstm.shape[1] , 1)

print("X_train: ", X_train_lstm.shape)
print("X_test: ", X_test_lstm.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#menentukan model, neuron, loss, dan optimizer
model_lstm =Sequential()
model_lstm.add(LSTM(100,input_shape=(None,1),activation="relu"))
model_lstm.add(Dense(1))
model_lstm.compile(loss="mean_squared_error",optimizer="adam")

#menentukan epoch dan batch size
history_lstm = model_lstm.fit(X_train_lstm,y_train_lstm,validation_data=(X_test_lstm,y_test_lstm),epochs=128,batch_size=16,verbose=1)

# Proses Plotting Loss dan Validasi Loss
loss = history_lstm.history['loss']
val_loss = history_lstm.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()

#Lakukan prediksi dan periksa Performence metrics
train_predict_lstm=model_lstm.predict(X_train_lstm)
test_predict_lstm=model_lstm.predict(X_test_lstm)
train_predict_lstm.shape, test_predict_lstm.shape

# Ubah kembali ke bentuk Original
train_predict_lstm = scaler.inverse_transform(train_predict_lstm)
test_predict_lstm = scaler.inverse_transform(test_predict_lstm)
original_ytrain_lstm = scaler.inverse_transform(y_train_lstm.reshape(-1,1))
original_ytest_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1,1))

"""## 2. CNN 1D"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

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

# Define the CNN model
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(1))  # Output layer for regression
model_cnn.compile(optimizer='adam', loss='mse')
model_cnn.summary()

# Train the model
history_cnn = model_cnn.fit(X_train_cnn, y_train_cnn, epochs=128, batch_size=32, validation_data=(X_test_cnn, y_test_cnn), verbose=1)

# Proses Plotting Loss dan Validasi Loss
loss = history_cnn.history['loss']
val_loss = history_cnn.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()

#Lakukan prediksi dan periksa Performence metrics
train_predict_cnn=model_cnn.predict(X_train_cnn)
test_predict_cnn=model_cnn.predict(X_test_cnn)
train_predict_cnn.shape, test_predict_cnn.shape

# Ubah kembali ke bentuk Original
train_predict_cnn = scaler.inverse_transform(train_predict_cnn)
test_predict_cnn = scaler.inverse_transform(test_predict_cnn)
original_ytrain_cnn = scaler.inverse_transform(y_train_cnn.reshape(-1,1))
original_ytest_cnn = scaler.inverse_transform(y_test_cnn.reshape(-1,1))

"""## 3. GRU"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

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

# Define the GRU model
model_gru = Sequential()
model_gru.add(GRU(100, input_shape=(X_train_gru.shape[1], 1), activation='relu')) # Adjust units as needed
model_gru.add(Dense(1))
model_gru.compile(optimizer='adam', loss='mse')

# Train the GRU model
history_gru = model_gru.fit(X_train_gru, y_train_gru, epochs=128, batch_size=16, validation_data=(X_test_gru,y_test_gru), verbose=1)

# Plotting Loss and Validation Loss for GRU
loss = history_gru.history['loss']
val_loss = history_gru.history['val_loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss (GRU)')
plt.legend()
plt.show()

# Make predictions
train_predict_gru = model_gru.predict(X_train_gru)
test_predict_gru = model_gru.predict(X_test_gru)

# Inverse transform the predictions and actual values
train_predict_gru = scaler.inverse_transform(train_predict_gru)
test_predict_gru = scaler.inverse_transform(test_predict_gru)
original_ytrain_gru = scaler.inverse_transform(y_train_lstm.reshape(-1,1)) # Use y_train_lstm here
original_ytest_gru = scaler.inverse_transform(y_test_lstm.reshape(-1,1)) # Use y_test_lstm here

"""# E. Evaluasi Model

## 1. Evaluasi Model LSTM
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error_lstm(y_true_lstm, y_pred_lstm):
    y_true_lstm, y_pred_lstm = np.array(y_true_lstm), np.array(y_pred_lstm)
    return np.mean(np.abs((y_true_lstm - y_pred_lstm) / y_true_lstm)) * 100

# Hitung Metrik Evaluasi untuk Data Training LSTM
mse_train_lstm = mean_squared_error(original_ytrain_lstm, train_predict_lstm)
rmse_train_lstm = np.sqrt(mse_train_lstm)
mae_train_lstm = mean_absolute_error(original_ytrain_lstm, train_predict_lstm)
mape_train_lstm = mean_absolute_percentage_error_lstm(original_ytrain_lstm, train_predict_lstm)
r2_train_lstm = r2_score(original_ytrain_lstm, train_predict_lstm)

# Hitung Metrik Evaluasi untuk Data Testing LSTM
mse_test_lstm = mean_squared_error(original_ytest_lstm, test_predict_lstm)
rmse_test_lstm = np.sqrt(mse_test_lstm)
mae_test_lstm = mean_absolute_error(original_ytest_lstm, test_predict_lstm)
mape_test_lstm = mean_absolute_percentage_error_lstm(original_ytest_lstm, test_predict_lstm)
r2_test_lstm = r2_score(original_ytest_lstm, test_predict_lstm)

print("Training Data Metrics LSTM:")
print(f"MSE: {mse_train_lstm:.2f}")
print(f"RMSE: {rmse_train_lstm:.2f}")
print(f"MAE: {mae_train_lstm:.2f}")
print(f"MAPE: {mape_train_lstm:.2f}%")
print(f"R-squared: {r2_train_lstm:.2f}")

print("\nTesting Data Metrics LSTM:")
print(f"MSE: {mse_test_lstm:.2f}")
print(f"RMSE: {rmse_test_lstm:.2f}")
print(f"MAE: {mae_test_lstm:.2f}")
print(f"MAPE: {mape_test_lstm:.2f}%")
print(f"R-squared: {r2_test_lstm:.2f}")

"""## 2. Evaluasi Model CNN 1D"""

def mean_absolute_percentage_error_cnn(y_true_cnn, y_pred_cnn):
    y_true_cnn, y_pred_cnn = np.array(y_true_cnn), np.array(y_pred_cnn)
    return np.mean(np.abs((y_true_cnn - y_pred_cnn) / y_true_cnn)) * 100

# Hitung metrik untuk data training CNN
mse_train_cnn = mean_squared_error(original_ytrain_cnn, train_predict_cnn)
rmse_train_cnn = np.sqrt(mse_train_cnn)
mae_train_cnn = mean_absolute_error(original_ytrain_cnn, train_predict_cnn)
mape_train_cnn = mean_absolute_percentage_error_cnn(original_ytrain_cnn, train_predict_cnn)
r2_train_cnn = r2_score(original_ytrain_cnn, train_predict_cnn)

# Hitung metrik untuk data testing CNN
mse_test_cnn = mean_squared_error(original_ytest_cnn, test_predict_cnn)
rmse_test_cnn = np.sqrt(mse_test_cnn)
mae_test_cnn = mean_absolute_error(original_ytest_cnn, test_predict_cnn)
mape_test_cnn = mean_absolute_percentage_error_cnn(original_ytest_cnn, test_predict_cnn)
r2_test_cnn = r2_score(original_ytest_cnn, test_predict_cnn)

print("\nTraining Data Metrics CNN:")
print(f"MSE: {mse_train_cnn:.4f}")
print(f"RMSE: {rmse_train_cnn:.4f}")
print(f"MAE: {mae_train_cnn:.4f}")
print(f"MAPE: {mape_train_cnn:.2f}%")
print(f"R-squared: {r2_train_cnn:.4f}")

print("\nTesting Data Metrics CNN:")
print(f"MSE: {mse_test_cnn:.4f}")
print(f"RMSE: {rmse_test_cnn:.4f}")
print(f"MAE: {mae_test_cnn:.4f}")
print(f"MAPE: {mape_test_cnn:.2f}%")
print(f"R-squared: {r2_test_cnn:.4f}")

"""## 3. Evaluasi Model GRU"""

def mean_absolute_percentage_error_gru(y_true_gru, y_pred_gru):
    y_true_gru, y_pred_gru = np.array(y_true_gru), np.array(y_pred_gru)
    return np.mean(np.abs((y_true_gru - y_pred_gru) / y_true_gru)) * 100

# Hitung Metrik Evaluasi untuk Data Training GRU
mse_train_gru = mean_squared_error(original_ytrain_gru, train_predict_gru)
rmse_train_gru = np.sqrt(mse_train_gru)
mae_train_gru = mean_absolute_error(original_ytrain_gru, train_predict_gru)
mape_train_gru = mean_absolute_percentage_error_gru(original_ytrain_gru, train_predict_gru)
r2_train_gru = r2_score(original_ytrain_gru, train_predict_gru)

# Hitung Metrik Evaluasi untuk Data Testing GRU
mse_test_gru = mean_squared_error(original_ytest_gru, test_predict_gru)
rmse_test_gru = np.sqrt(mse_test_gru)
mae_test_gru = mean_absolute_error(original_ytest_gru, test_predict_gru)
mape_test_gru = mean_absolute_percentage_error_gru(original_ytest_gru, test_predict_gru)
r2_test_gru = r2_score(original_ytest_gru, test_predict_gru)

print("\nTraining Data Metrics GRU:")
print(f"MSE: {mse_train_gru:.2f}")
print(f"RMSE: {rmse_train_gru:.2f}")
print(f"MAE: {mae_train_gru:.2f}")
print(f"MAPE: {mape_train_gru:.2f}%")
print(f"R-squared: {r2_train_gru:.2f}")

print("\nTesting Data Metrics GRU:")
print(f"MSE: {mse_test_gru:.2f}")
print(f"RMSE: {rmse_test_gru:.2f}")
print(f"MAE: {mae_test_gru:.2f}")
print(f"MAPE: {mape_test_gru:.2f}%")
print(f"R-squared: {r2_test_gru:.2f}")

"""## 4. Kesimpulan"""

# membuat kamus berdasarkan hasil
results = {
    'Model': ['LSTM', 'CNN', 'GRU'],
    'MSE_Train': [mse_train_lstm, mse_train_cnn, mse_train_gru],
    'RMSE_Train': [rmse_train_lstm, rmse_train_cnn, rmse_train_gru],
    'MAE_Train': [mae_train_lstm, mae_train_cnn, mae_train_gru],
    'MAPE_Train': [mape_train_lstm, mape_train_cnn, mape_train_gru],
    'R2_Train': [r2_train_lstm, r2_train_cnn, r2_train_gru],
    'MSE_Test': [mse_test_lstm, mse_test_cnn, mse_test_gru],
    'RMSE_Test': [rmse_test_lstm, rmse_test_cnn, rmse_test_gru],
    'MAE_Test': [mae_test_lstm, mae_test_cnn, mae_test_gru],
    'MAPE_Test': [mape_test_lstm, mape_test_cnn, mape_test_gru],
    'R2_Test': [r2_test_lstm, r2_test_cnn, r2_test_gru],
}

# Membuat dataframe berasarkan result(hasil)
df_results = pd.DataFrame(results)
print(df_results)

# Sortir berdasarkan R2
df_results_sorted = df_results.sort_values(by=['MSE_Test', 'R2_Test'], ascending=[True, True])
print("\nModel terbaik berdasarkan Evaluasi:")
df_results_sorted

"""**Berdasarkan Hasil Evaluasi Model, LSTM merupakan Model Terbaik untuk studi kasus Time Series univariate**

Maka dari itu Model **LSTM** yang akan dipilih untuk dijadikan implementasi.

# F. Implementasi

Simpan Model:
"""

import tensorflow as tf

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model_lstm) # Change to model_lstm

# Enable resource variables (if needed)
converter.experimental_enable_resource_variables = True

# Set supported ops to include Select TF ops (if needed)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('lstm_model.tflite', 'wb') as f:  # Change filename
    f.write(tflite_model)

"""Data Sebelum dan Sesudah Prediksi:"""

import plotly.express as px
from itertools import cycle

look_back = time_step

train_predict_plot = np.empty_like(maindf_cleaned_split[['Close']])
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict_lstm) + look_back, :] = train_predict_lstm
print("Train predicted data: ", train_predict_plot.shape)

test_predict_plot = np.empty_like(maindf_cleaned_split[['Close']])
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict_lstm) + (look_back * 2) + 1:len(maindf_cleaned_split[['Close']]) - 1, :] = test_predict_lstm
print("Test predicted data: ", test_predict_plot.shape)

plotdf = pd.DataFrame({'Date': maindf_cleaned_split.index,
                       'original_close': maindf_cleaned_split['Close'],
                       'train_predicted_close': train_predict_plot.reshape(1, -1)[0].tolist(),
                       'test_predicted_close': test_predict_plot.reshape(1, -1)[0].tolist()})


names = cycle(['Data Harga Asli', 'Data Harga(berdsasarkan data latih)', 'Data Harga(berdsasarkan data Uji)'])

fig = px.line(plotdf, x=plotdf['Date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value': 'Harga IHSG(^JKSE)', 'Date': 'Tanggal'})
fig.update_layout(title_text='Komparasi dari Harga Penutupan asli dengan Harga Penutupan Hasil Prediksi',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

"""Contoh penggunaan untuk prediksi 7 hari kedepan:"""

# Predict the next 7 days using LSTM
x_input = test_data_scaled[-10:].values  # Taking the last 10 days' data
x_input = x_input.reshape(1, -1, 1)  # Reshaping for LSTM input

temp_input = list(x_input[0].flatten())
lst_output = []
n_steps = 10

i = 0
while i < 7:
    if len(temp_input) > 10:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1, 1)
        yhat = model_lstm.predict(x_input, verbose=0)  # Using model_lstm for prediction
        temp_input.extend(yhat[0].flatten().tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model_lstm.predict(x_input, verbose=0)  # Using model_lstm for prediction
        temp_input.extend(yhat[0].flatten().tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

day_new = np.arange(1, 11)
day_pred = np.arange(11, 18)

plt.plot(day_new, scaler.inverse_transform(test_data_scaled[-10:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
plt.title('Prediksi Harga Saham 7 Hari ke Depan (LSTM)')  # Updated title
plt.xlabel('Hari')
plt.ylabel('Harga Saham')
plt.legend(['Data Aktual', 'Prediksi'])
plt.show()

!pip freeze > requirements.txt