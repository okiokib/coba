import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import time
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score
from sklearn.metrics import classification_report

#Membuat Sidebar
with st.sidebar:
    selected = option_menu ('Klasifikasi Penyakit ISPA',
    ['Dataset',
     'Cleaning Data & Normalisasi',
     'Preprocessing',
     'Smote',
     'Klasifikasi KNN',
     'Klasifikasi Naive Bayes',
     'Klasifikasi Decision Tree',
     'Prediksi dengan KNN',
     'Prediksi dengan Naive Bayes',
     'Prediksi dengan Decision Tree'],
    default_index=0)


#Jika memilih menu dataset
if (selected == 'Dataset'):
    st.title('Dataset Penyakit ISPA Puskesmas Tongguh Arosbaya Tahun 2018-2021')
    # Menampilkan dataset dari file CSV
    dataset = pd.read_csv('ispa_data.csv')
    st.write(dataset.head(10))

    st.write("Pada dataset ini terdiri sebanyak 113 data dengan 13 fitur. Adapun fitur-fiturnya yaitu:")
    st.info('''
    1. Nama (Nama Pasein)
    2. Gender (Jenis kelamin pasien (P/L))
    3. Umur (Umur pasien)
    4. Batuk (Gejala pasien)
    5. Demam (Gejala pasien)
    6. Pilek (Gejala pasien)
    7. Sesak Nafas (Gejala pasien)
    8. Radang Tenggorokan (Gejala pasien)
    9. TD sistolik (Tekanan darah sistole)
    10. TD diastolik (Tekanan darah diastole)
    11. ND (Denyut nadi pasien per menit)
    12. RR (Respirasi pasien per menit)
    13. Suhu (suhu pasien)
    
    Selain itu, untuk kategorinya yaitu berupa (Diagnosis Pasien (Pneunomia, Non Pneunomia))
    ''')


#Jika memilih menu Cleaning Data & Normalisasi
if (selected == 'Cleaning Data & Normalisasi'):
    st.title('Cleaning Data & Normalisasi Pada Dataset ISPA')

    dataset = pd.read_csv('ispa_data.csv')

    st.write('''### Cleaning Data (Hapus Fitur)''')
    #Cleaning Data dengan hapus fitur 
    hapus_fitur = pd.DataFrame(dataset)
    kolom_yang_dihapus = ['Nama','TD sistolik (mmhg)', 'TD diastolik (mmhg)', 'ND (x/mnt)']
    hapus_fitur = hapus_fitur.drop(columns=kolom_yang_dihapus)
    # Menampilkan DataFrame setelah menghapus kolom
    st.write(hapus_fitur.head(10))
    st.caption('Pada proses ini dilakukan penghapusan untuk fitur Nama, TD sistolik, TD diastolik, dan ND(Nadi)')

    st.write('''### Normalisasi (Min Max)''')
    kolom_normalisasi = ['Umur', 'RR (x/mnt)', 'S']
    # Fungsi Min-Max Scaling
    def min_max_scaling(column):
        min_val = column.min()
        max_val = column.max()
        return (column - min_val) / (max_val - min_val)
    # Melakukan normalisasi Min-Max Scaling pada kolom yang dipilih
    for kolom in kolom_normalisasi:
        hapus_fitur[kolom] = min_max_scaling(hapus_fitur[kolom])
    st.write(hapus_fitur.head(10))
    st.caption('Pada proses normalisasi ini dilakukan normalisasi pada kolom Umur, RR(Respirasi, dan S(Suhu)')


#Jika memilih menu Preprocessing
if (selected == 'Preprocessing'):
    st.title('Tahapan Preprocessing Pada Dataset ISPA')

    data_normalisasi = pd.read_csv('normalisasi.csv')

    # Membagi dataset menjadi X dan y
    X = data_normalisasi.iloc[:, :-1].values
    y = data_normalisasi.iloc[:, 9].values

    st.write('''### Transformasi ''')
     # Mengkodekan Variabel Independern (X)
    label_encoder = LabelEncoder()
    # Iterasi melalui setiap kolom dalam X
    for i in range(X.shape[1]):
        if isinstance(X[0, i], str):  # Memeriksa apakah kolom ini adalah tipe data string (kategori)
            X[:, i] = label_encoder.fit_transform(X[:, i])
    # Membuat DataFrame Pandas dari hasil transformasi
    X_encoded = pd.DataFrame(X, columns=data_normalisasi.columns[:-1])

    st.write(X_encoded)
    st.caption('Dilakukan transformasi pada fitur Gender')

    # Mengkodekan Variabel Dependen (y)
    le=LabelEncoder()
    transform_y = le.fit_transform(y)
    tabel_y = pd.DataFrame(transform_y, columns=['Target'])
    st.write(tabel_y)
    st.caption('Dilakukan transformasi pada target')

    st.write('''### Dataset Hasil Transformasi dari Fitur X dan y''')
    # Menggabungkan DataFrame X dan y
    final_dataset = pd.concat([X_encoded, tabel_y], axis=1)
    # Menampilkan dataset hasil penggabungan
    st.write(final_dataset)

    st.write('''### Korelasi (Pearson) ''')
    correlation = final_dataset.corr()
    # Menampilkan heatmap menggunakan Matplotlib dan Seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Correlation of Attributes with Class variable')
    # Membuat heatmap
    heatmap = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white', ax=ax)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=30)          
    st.pyplot(fig)


#Jika memilih menu SMOTE
if (selected == 'Smote'):
    st.title('Proses SMOTE')

    data = pd.read_csv('final_dataset.csv')

    #Import Dataset
    X = data.iloc[:, :-1].values
    y = data.iloc[:, 9].values

    st.write("Distribusi kelas sebelum SMOTE: Non Pneumonia = 48 | Pneumonia = 65")
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    column_names = data.columns[:-1].tolist()
    resampled_df = pd.DataFrame(X_resampled, columns=column_names)
    resampled_df['Target'] = y_resampled
    st.write("Distribusi kelas setelah SMOTE: Non Pneumonia = 65 | Pneumonia = 65")
    st.write('''### Dataset SMOTE ''')
    st.write(resampled_df)


#Import Dataset untuk klasifikasi
data_akhir = pd.read_csv('resampled_df.csv')
#Split Dataset
X = data_akhir.iloc[:, :-1].values
y = data_akhir.iloc[:, 9].values
# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)


#Jika memilih menu kalsifikasi KNN
if (selected == 'Klasifikasi KNN'):
    st.title('Klasifikasi dengan Metode K-Nearest Neighbour')

    # Membuat Model KNN
    knn = KNeighborsClassifier(n_neighbors=3, p=1) #p1 = Manhattan Distance
    start_time = time.time()
    # latih the model
    knn.fit(X_train, y_train)
    #Prediksi Model
    y_pred = knn.predict(X_test)


    # Hitung akurasi, presisi, recall, f1-score
    st.write('''### Hasil akurasi, presisi, recall, dan f1-score KNN''')
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Akurasi KNN: {:.2f}%".format(accuracy * 100))
    precision = precision_score(y_test, y_pred, average='weighted')
    st.write("Presisi KNN: {:.2f}%".format(precision * 100))
    recall = recall_score(y_test, y_pred, average='weighted')
    st.write("Recall KNN: {:.2f}%".format(recall * 100))
    f1 = f1_score(y_test, y_pred, average='weighted')
    st.write("f1 score KNN: {:.2f}%".format(f1 * 100))

    st.write('''### Runtime ''')
    end_time = time.time()
    runtime = end_time - start_time
    st.write(f"Runtime K-Nearest Neighbors: {runtime:.2f} seconds")

    st.write('''### Classification Report ''')
    st.text(classification_report(y_test,y_pred))

    st.write('''### Confussion Matrix ''')
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)

    #Menyimpan Model
    filename = 'prediksi_knn.sav'
    pickle.dump(knn, open(filename,'wb'))


#Jika memilih menu klasifikasi Naive Bayes
if (selected == 'Klasifikasi Naive Bayes'):
    st.title('Klasifikasi dengan Metode Naive Bayes')

    # Membuat model
    gnb = GaussianNB()
    start_time = time.time()
    # melatih model
    gnb.fit(X_train, y_train)
    # prediksi
    y_pred = gnb.predict(X_test)

    # Hitung akurasi, presisi, recall, f1-score
    st.write('''### Hasil akurasi, presisi, recall, dan f1-score Naive Bayes''')
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Akurasi Naive Bayes: {:.2f}%".format(accuracy * 100))
    precision = precision_score(y_test, y_pred, average='weighted')
    st.write("Presisi Naive Bayes: {:.2f}%".format(precision * 100))
    recall = recall_score(y_test, y_pred, average='weighted')
    st.write("Recall Naive Bayes: {:.2f}%".format(recall * 100))
    f1 = f1_score(y_test, y_pred, average='weighted')
    st.write("f1 score Naive Bayes: {:.2f}%".format(f1 * 100))

    st.write('''### Runtime ''')
    end_time = time.time()
    runtime = end_time - start_time
    st.write(f"Runtime Naive Bayes: {runtime:.2f} seconds")

    st.write('''### Classification Report ''')
    st.text(classification_report(y_test,y_pred))

    st.write('''### Confussion Matrix ''')
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)

    #Simpan Model
    filename = 'prediksi_naive_bayes.sav'
    pickle.dump(gnb, open(filename,'wb'))


#Jika memilih menu klasifikasi Decision Tree
if (selected == 'Klasifikasi Decision Tree'):
    st.title('Klasifikasi dengan Metode Decision Tree')

    #Membuat Model
    clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=0)
    start_time = time.time()
    #Melatih Model
    clf_en.fit(X_train, y_train)
    #prediksi
    y_pred_en = clf_en.predict(X_test)

    # Hitung akurasi, presisi, recall, f1-score
    st.write('''### Hasil akurasi, presisi, recall, dan f1-score Decision Tree''')
    accuracy = accuracy_score(y_test, y_pred_en)
    st.write("Akurasi Decision Tree: {:.2f}%".format(accuracy * 100))
    precision = precision_score(y_test, y_pred_en, average='weighted')
    st.write("Presisi Decision Tree: {:.2f}%".format(precision * 100))
    recall = recall_score(y_test, y_pred_en, average='weighted')
    st.write("Recall Decision Tree: {:.2f}%".format(recall * 100))
    f1 = f1_score(y_test, y_pred_en, average='weighted')
    st.write("f1 score Decision Tree: {:.2f}%".format(f1 * 100))

    st.write('''### Runtime ''')
    end_time = time.time()
    runtime = end_time - start_time

    st.write(f"Runtime Decision Tree: {runtime:.2f} seconds")

    st.write('''### Classification Report ''')
    st.text(classification_report(y_test,y_pred_en))

    st.write('''### Confussion Matrix ''')
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred_en)
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)

    #Simpan Model
    filename = 'prediksi_decision_tree.sav'
    pickle.dump(clf_en, open(filename,'wb'))


#Jika memilih menu prediksi dengan KNN
if (selected == 'Prediksi dengan KNN'):
    st.title('Prediksi ISPA Dengan Metode K-Nearest Neighbour')
    
    knn_model = pickle.load(open('prediksi_knn.sav', 'rb'))

    Gender = st.text_input ('Input gender : 0 = laki-laki, 1 = perempuan')
    Umur = st.text_input ('Input Umur')
    Batuk = st.text_input ('Input apakah batuk atau tidak : 0 = tidak, 1 = iya')
    Demam = st.text_input ('Input apakah demam atau tidak : 0 = tidak, 1 = iya')
    Pilek = st.text_input ('Input apakah pilek atau tidak : 0 = tidak, 1 = iya')
    Sesak_Napas = st.text_input ('Input apakah sesak napas atau tidak : 0 = tidak, 1 = iya')
    Radang_Tenggorokan = st.text_input ('Input apakah radang tenggorokan atau tidak : 0 = tidak, 1 = iya')
    respirasi = st.text_input ('Input nilai respirasi')
    Suhu = st.text_input ('Input suhu tubuh')

    # Konversi input ke tipe data numerik
    Gender = int(Gender) if Gender else 0
    Umur = int(Umur) if Umur else 0
    Batuk = int(Batuk) if Batuk else 0
    Demam = int(Demam) if Demam else 0
    Pilek = int(Pilek) if Pilek else 0
    Sesak_Napas = int(Sesak_Napas) if Sesak_Napas else 0
    Radang_Tenggorokan = int(Radang_Tenggorokan) if Radang_Tenggorokan else 0
    respirasi = int(respirasi) if respirasi else 0
    Suhu = float(Suhu) if Suhu else 0

    if st.button('prediksi'):
        knn_prediction = knn_model.predict([[Gender, Umur, Batuk, Demam, Pilek, Sesak_Napas, Radang_Tenggorokan, respirasi, Suhu]])

        if knn_prediction[0] == 1:
            ispa_knn = 'Pneumonia'
        else:
            ispa_knn = 'Tidak Pneumonia'
        
        st.success(ispa_knn)


#Jika memilih menu Prediksi dengan Naive Bayes
if (selected == 'Prediksi dengan Naive Bayes'):
    st.title('Prediksi ISPA Dengan Metode Naive Bayes')

    naive_bayes_model = pickle.load(open('prediksi_naive_bayes.sav', 'rb'))

    Gender = st.text_input ('Input gender : 0 = laki-laki, 1 = perempuan')
    Umur = st.text_input ('Input Umur')
    Batuk = st.text_input ('Input apakah batuk atau tidak : 0 = tidak, 1 = iya')
    Demam = st.text_input ('Input apakah demam atau tidak : 0 = tidak, 1 = iya')
    Pilek = st.text_input ('Input apakah pilek atau tidak : 0 = tidak, 1 = iya')
    Sesak_Napas = st.text_input ('Input apakah sesak napas atau tidak : 0 = tidak, 1 = iya')
    Radang_Tenggorokan = st.text_input ('Input apakah radang tenggorokan atau tidak : 0 = tidak, 1 = iya')
    respirasi = st.text_input ('Input nilai respirasi')
    Suhu = st.text_input ('Input suhu tubuh')

    # Konversi input ke tipe data numerik
    Gender = int(Gender) if Gender else 0
    Umur = float(Umur) if Umur else 0
    Batuk = int(Batuk) if Batuk else 0
    Demam = int(Demam) if Demam else 0
    Pilek = int(Pilek) if Pilek else 0
    Sesak_Napas = int(Sesak_Napas) if Sesak_Napas else 0
    Radang_Tenggorokan = int(Radang_Tenggorokan) if Radang_Tenggorokan else 0
    respirasi = int(respirasi) if respirasi else 0
    Suhu = float(Suhu) if Suhu else 0

    if st.button('prediksi'):
        naive_bayes_prediction = naive_bayes_model.predict([[Gender, Umur, Batuk, Demam, Pilek, Sesak_Napas, Radang_Tenggorokan, respirasi, Suhu]])

        if naive_bayes_prediction[0] == 1:
            ispa_naive_bayes = 'Pneumonia'
        else:
            ispa_naive_bayes = 'Tidak Pneumonia'
        
        st.success(ispa_naive_bayes)


if (selected == 'Prediksi dengan Decision Tree'):
    st.title('Prediksi ISPA Dengan Metode Decision Tree')
    
    decision_tree_model = pickle.load(open('prediksi_decision_tree.sav', 'rb'))

    Gender = st.text_input ('Input gender : 0 = laki-laki, 1 = perempuan')
    Umur = st.text_input ('Input Umur')
    Batuk = st.text_input ('Input apakah batuk atau tidak : 0 = tidak, 1 = iya')
    Demam = st.text_input ('Input apakah demam atau tidak : 0 = tidak, 1 = iya')
    Pilek = st.text_input ('Input apakah pilek atau tidak : 0 = tidak, 1 = iya')
    Sesak_Napas = st.text_input ('Input apakah sesak napas atau tidak : 0 = tidak, 1 = iya')
    Radang_Tenggorokan = st.text_input ('Input apakah radang tenggorokan atau tidak : 0 = tidak, 1 = iya')
    respirasi = st.text_input ('Input nilai respirasi')
    Suhu = st.text_input ('Input suhu tubuh')

    # Konversi input ke tipe data numerik
    Gender = int(Gender) if Gender else 0
    Umur = int(Umur) if Umur else 0
    Batuk = int(Batuk) if Batuk else 0
    Demam = int(Demam) if Demam else 0
    Pilek = int(Pilek) if Pilek else 0
    Sesak_Napas = int(Sesak_Napas) if Sesak_Napas else 0
    Radang_Tenggorokan = int(Radang_Tenggorokan) if Radang_Tenggorokan else 0
    respirasi = int(respirasi) if respirasi else 0
    Suhu = float(Suhu) if Suhu else 0

    if st.button('prediksi'):
        decision_tree_prediction = decision_tree_model.predict([[Gender, Umur, Batuk, Demam, Pilek, Sesak_Napas, Radang_Tenggorokan, respirasi, Suhu]])

        if decision_tree_prediction[0] == 1:
            ispa_decision_tree = 'Pneumonia'
        else:
            ispa_decision_tree = 'Tidak Penumonia'
        
        st.success(ispa_decision_tree)
