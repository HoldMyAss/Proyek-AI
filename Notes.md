Ada 2 kode, 'main' tanpa menggunakan filter, 'main2' lagi menggunakan Savitzky-Golay filter. Yang kami masukan ke artikel adalah tanpa menggunakan filter, karena menurut kami data tidak harus di filter terlebih dahulu.

Kami menggunakan library penting seperti NumPy, Pandas, dan PyTorch yang digunakan untuk manipulasi data dan pembangunan model deep learning. File CSV yang berisi data degradasi baterai dimuat menggunakan Pandas. Data tersebut mencakup fitur-fitur penting seperti Kapasitas, Resistansi Internal, Waktu Pengisian, dan State of Health (SoH), yang semuanya merupakan indikator utama dalam menentukan kinerja dan degradasi baterai. Data kemudian dinormalisasi menggunakan MinMaxScaler, karena normalisasi penting untuk mempercepat konvergensi model neural network dan mencegah dominasi fitur tertentu. Data kemudian diubah menjadi tensor PyTorch agar dapat digunakan dalam pelatihan model. Selanjutnya, dibuat urutan data (sequences) dengan panjang tertentu, untuk mempersiapkannya agar cocok digunakan dalam model sekuensial seperti RNN, LSTM, MLP, dan GRU.

Setelah data siap, data dibagi menjadi data pelatihan dan pengujian dengan rasio 80:20. Ini penting untuk memastikan bahwa model diuji pada data yang belum pernah dilihat sebelumnya. Dataset ini kemudian dibungkus menggunakan TensorDataset dan dimasukkan ke dalam DataLoader, yang berfungsi untuk membagi data ke dalam batch dan melakukan pengacakan (shuffling). Ini membantu mempercepat pelatihan dan mengurangi risiko overfitting.

Empat arsitektur neural network yang berbeda didefinisikan untuk dibandingkan.
1. MLP (Multi-Layer Perceptron) adalah jaringan saraf feedforward dengan lapisan penuh, tanpa memori, dan digunakan sebagai model dasar (baseline).
2. RNN (Recurrent Neural Network) menambahkan memori melalui koneksi berulang sehingga dapat mengenali pola dalam data sekuensial. Namun, RNN rentan terhadap masalah vanishing gradient.
3. LSTM (Long Short-Term Memory) adalah versi RNN yang diperbaiki, dengan mekanisme gerbang (gates) yang membuatnya mampu mengingat informasi jangka panjang.
4. GRU (Gated Recurrent Unit) mirip dengan LSTM tetapi lebih sederhana secara komputasi dan sering kali memberikan performa yang sebanding.
Semua model menggunakan lapisan output linier untuk menghasilkan prediksi akhir.

Satu fungsi pelatihan digunakan untuk semua model. Fungsi ini menggunakan Mean Squared Error (MSE) sebagai fungsi kerugian dan Adam optimizer untuk memperbarui bobot model. Pelatihan dilakukan dalam beberapa epoch dan kerugian (loss) dicatat di setiap epoch. Ini memungkinkan untuk memantau konvergensi model. Dengan menggunakan loop pelatihan yang sama, evaluasi antar model menjadi adil dan konsisten.

Setiap model (MLP, RNN, LSTM, GRU) diinisialisasi dan dilatih menggunakan fungsi pelatihan yang telah didefinisikan sebelumnya. Pelatihan dilakukan selama 50 epoch. Ini memberikan setiap model kesempatan untuk mempelajari pola degradasi baterai dari data urutan waktu. Dari hasil pelatihan, kita bisa melihat model mana yang lebih cepat konvergen dan lebih efektif dalam belajar.

Setelah pelatihan, setiap model dievaluasi menggunakan data uji. Hasil prediksi dan nilai aktual dikembalikan ke skala aslinya menggunakan inverse transform dari MinMaxScaler. Evaluasi dilakukan menggunakan beberapa metrik:
1. MAE (Mean Absolute Error) mengukur rata-rata besar kesalahan.
2. RMSE (Root Mean Squared Error) memberikan penalti lebih besar pada kesalahan besar.
3. R-squared (R²) mengukur seberapa baik model menjelaskan variasi dalam data target.
4. Relative Error (RE) menunjukkan seberapa besar kesalahan relatif terhadap nilai sebenarnya.

Terakhir, hasil evaluasi dari semua model dicetak, memungkinkan perbandingan secara langsung. Ini membantu menentukan model mana yang paling akurat dalam memahami pola degradasi baterai dan memprediksi Remaining Useful Life (RUL) secara efektif. Secara teori, model RNN atau LSTM unggul karena kemampuannya dalam memahami dependensi jangka panjang pada data sekuensial.