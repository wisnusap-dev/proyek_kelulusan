
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("kelulusan_mahasiswa.csv")
print(df.info())
print(df.head())

print("Missing values per kolom:")
print(df.isnull().sum())

df = df.drop_duplicates()
print("\nJumlah data setelah hapus duplikat:", len(df))

sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.show()

print("\nStatistik deskriptif:")
print(df.describe())

sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi Nilai IPK")
plt.show()

sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("Hubungan IPK dan Waktu Belajar terhadap Kelulusan")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Korelasi antar variabel")
plt.show()

df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

df.to_csv("processed_kelulusan.csv", index=False)

print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Split pertama: Train (70%) dan sisa 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Split kedua: Validation (15%) dan Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

X_train.to_csv("X_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
