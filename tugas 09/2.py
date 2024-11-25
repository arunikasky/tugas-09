from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Mount Google Drive
drive.mount('/content/gdrive')

# Path ke file yang di-mount dari Google Drive
Filedb = '/content/Sinus.txt'  # Ganti dengan path file yang benar
Database = pd.read_csv(Filedb, sep=",", header=0)

# Lihat data
print("----------------------------")
print(Database)

# x data, y target (sesuaikan dengan nama kolom yang ada pada data)
x = Database[['Feature']]  # Ganti dengan nama kolom fitur yang sesuai
y = Database['Target']     # Ganti dengan nama kolom target yang sesuai

# Inisialisasi dan latih model DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state=1)
reg = reg.fit(x, y)

# Display predicted data
xx = np.arange(1, 21, 1)
n = len(xx)
print("xx(i) Decision Tree")
for i in range(n):
    y_dct = reg.predict([[xx[i]]])
    print('{:.2f}'.format(xx[i]), y_dct)

# Plot data yang diprediksi
y_dct2 = reg.predict(x)
plt.figure()
plt.plot(x, y_dct2, color='red')
plt.scatter(x, y, color='blue')
plt.title('Prediksi Data Menggunakan Decision Tree')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Decision Tree', 'data'], loc=2)
plt.show()
