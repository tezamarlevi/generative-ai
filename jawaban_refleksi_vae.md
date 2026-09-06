# Refleksi Hasil VAE MNIST

Draf refleksi berdasarkan hasil project VAE MNIST 2D, 15 epoch. Model 10D dan autoencoder biasa belum diuji; bagian terkait merupakan hipotesis.

## 1. Interpretasi Latent Space

![Latent space VAE 2D](outputs/latent_space.png)

Saat melihat scatter plot hasil VAE saya, perhatian saya langsung tertuju pada kumpulan digit 1 di bagian atas. Kelompok tersebut cukup jauh dari sebagian besar digit lainnya, walaupun beberapa titik tetap menyebar keluar. Digit 0 cenderung berkumpul di kanan, sedangkan digit 6 banyak berada di bawah. Susunan ini menunjukkan bahwa encoder sudah belajar membedakan sebagian pola tulisan tangan. Namun, saya tidak melihat sepuluh kelompok dengan batas yang bersih. Bagian tengah justru berisi banyak warna yang saling menumpuk.

Menurut saya, label terdistribusi dengan pola tertentu, tetapi tidak teratur dalam arti berurutan dari 0 sampai 9. Saya juga tidak mengharapkan urutan angka seperti itu, karena training hanya memakai gambar sebagai input dan target reconstruction. Model tidak menerima tugas untuk mengurutkan digit ataupun memisahkan kelas. Warna pada scatter plot baru membantu saya membaca hasil encoder setelah training selesai. Karena itu, kedekatan dua titik lebih berkaitan dengan representasi gambar yang model pelajari daripada hubungan angka secara matematis.

Area digit 4 dan 9 menjadi bagian yang paling saya perhatikan. Median kedua kelompok hampir berimpit, dan banyak titiknya menempati wilayah yang sama. Ketika saya membandingkannya dengan reconstruction, contoh digit 4 memang terlihat berubah menyerupai 9. Kedua hasil tersebut konsisten secara visual, meskipun saya belum bisa memastikan bahwa overlap menjadi satu-satunya penyebab kesalahan itu. Di tengah plot, digit 5 dan 8 juga saling berdekatan; lengkungan tulisan yang mirip mungkin membuat encoder kesulitan mempertahankan perbedaannya dalam dua koordinat.

Bagi saya, keterbatasan dua dimensi terasa jelas pada hasil ini. Encoder harus merangkum gambar berisi 784 nilai piksel menjadi representasi yang sangat pendek, sementara KL divergence memberi penalti terhadap posterior yang terlalu jauh dari prior. Saya membaca cluster yang bertumpuk sebagai konsekuensi yang masuk akal dari pembatasan tersebut, bukan langsung sebagai tanda bahwa program gagal.

Sumbu z1 dan z2 sendiri tidak saya anggap sebagai ukuran sifat tulisan tertentu. Saya belum punya bukti bahwa z1 berarti kemiringan atau z2 berarti ketebalan garis. Keduanya bekerja bersama sebagai koordinat latent. Plot ini juga memakai mu, sehingga setiap titik menunjukkan mean posterior suatu gambar, bukan hasil sampling acak. Setelah melihat semua 10.000 titik test, penilaian saya adalah bahwa VAE berhasil membentuk sebagian kelompok visual, tetapi belum memisahkan setiap digit dengan jelas.

## 2. Rekonstruksi Gagal (Failure Case)

![Perbandingan original dan reconstruction, perhatikan digit 4 dan digit 5](outputs/reconstruction.png)

Dua contoh kegagalan ada pada kolom berlabel Digit 4 dan Digit 5. Baris atas adalah original, sedangkan baris bawah adalah reconstruction.

Saya memilih dua contoh kegagalan dari gambar reconstruction, yaitu digit 4 dan digit 5. Keduanya merupakan contoh pertama untuk label masing-masing dalam test set, dengan indeks 4 dan 8 jika penghitungan indeks mulai dari nol. Saya memakai checkpoint setelah 15 epoch dan memberikan mu kepada decoder, sama seperti pada visualisasi reconstruction utama. Jadi, contoh ini berasal dari model yang benar-benar sudah berjalan, tanpa mengganti input agar kesalahannya terlihat lebih mencolok.

Pada contoh digit 4, gambar asli mempunyai bagian atas yang terbuka dan garis tegak di sebelah kanan. Hasil reconstruction justru membentuk bagian atas yang lebih tertutup, sehingga saya lebih mudah membacanya sebagai angka 9. Bagi saya, masalahnya sudah melewati sekadar blur. Bentuk yang membedakan identitas digit ikut berubah. Jika saya hanya melihat output tanpa gambar pembanding, saya tidak akan yakin bahwa input awalnya adalah 4.

Digit 5 mengalami perubahan yang berbeda. Input aslinya memperlihatkan garis mendatar di bagian atas, lalu lengkungan yang menyambung ke bagian bawah. Pada reconstruction, garis atas tersebut tidak lagi jelas dan bentuknya lebih menyerupai dua bagian membulat seperti angka 8. Saya melihat decoder masih menghasilkan gambar yang masuk akal sebagai tulisan tangan, tetapi tidak cukup setia terhadap bentuk input. Hal ini membuat saya membedakan keberhasilan menghasilkan digit dengan keberhasilan merekonstruksi digit yang sama.

Dugaan teknis saya terutama berkaitan dengan kapasitas latent dua dimensi. Encoder mungkin menempatkan variasi tulisan yang berbeda terlalu dekat, sehingga decoder tidak mempunyai informasi yang cukup untuk mempertahankan bagian terbuka pada angka 4 atau garis atas pada angka 5. Scatter plot mendukung dugaan ini karena wilayah 4 dan 9 banyak bertumpuk, sementara 5 berdekatan dengan 8. Saya belum melakukan eksperimen terpisah untuk membuktikan penyebab tersebut.

Objective training juga membantu saya menjelaskan hasilnya. BCE menghitung perbedaan piksel dan tidak memberi penalti khusus ketika bentuk 4 berubah menjadi 9. KL divergence mengatur distribusi latent, tetapi tidak memeriksa identitas digit. Karena gambar ini memakai z = mu, saya tidak bisa menyalahkan satu pengambilan noise yang kebetulan buruk sebagai penyebab langsungnya.

Total training loss memang turun dari 190,5184 menjadi 150,1907. Meski demikian, dua contoh ini mengingatkan saya bahwa penurunan rata-rata loss tidak menjamin setiap reconstruction mempertahankan ciri input. Saya perlu melihat gambar satu per satu untuk menemukan kesalahan yang angka rata-rata sembunyikan.

## 3. Hipotesis Eksperimen: Latent Space 2 menjadi 10 Dimensi

Pembahasan poin 4 sampai 6 berikut berisi prediksi berdasarkan hasil VAE 2D yang sudah berjalan.

## 4. Prediksi Kualitas Rekonstruksi pada Latent Space 10 Dimensi

Saya memperkirakan reconstruction akan lebih baik jika latent space berubah dari 2 menjadi 10 dimensi. Harapan saya terutama berkaitan dengan kemampuan mempertahankan ciri pembeda digit, bukan sekadar menghasilkan tepi yang terlihat lebih tajam. Pada model sekarang, angka 4 kehilangan bagian atas yang terbuka dan angka 5 kehilangan bentuk garis atasnya. Kesalahan tersebut membuat saya menduga bahwa dua koordinat belum cukup untuk menyimpan variasi bentuk yang decoder perlukan.

Dengan sepuluh dimensi, encoder mempunyai ruang representasi yang lebih besar. Saya berharap variasi bentuk angka 4 tidak harus menempati representasi yang terlalu mirip dengan angka 9. Begitu juga dengan angka 5 dan 8, yang pada hasil sekarang tampak berdekatan di latent space. Saya tidak menganggap setiap dimensi baru otomatis menjadi tempat bagi satu sifat tertentu, seperti kemiringan atau ketebalan. Model tetap belajar pembagian informasi melalui objective training, sehingga saya harus memeriksa penggunaannya setelah eksperimen berjalan.

Prediksi saya bukan bahwa semua gambar akan langsung menjadi tajam. Model tetap memakai decoder MLP dan objective BCE bersama KL divergence. Kedua komponen tersebut tetap membatasi cara model mempertahankan informasi. Sebagian dimensi tambahan juga mungkin kurang terpakai apabila encoder menghasilkan posterior yang mendekati prior pada koordinat tersebut. Karena itu, saya lebih yakin memprediksi perbaikan sebagian detail daripada menjanjikan semua kegagalan reconstruction akan hilang.

Saya akan menguji dugaan ini dengan memakai kembali dua gambar gagal tadi, kemudian membandingkan hasil model 2D dan 10D secara berdampingan. Gambar test harus sama agar saya tidak membandingkan satu contoh tulisan yang mudah dengan contoh lain yang lebih sulit. Saya juga akan menjaga learning rate, jumlah epoch, serta pengolahan dataset tetap sama; eksperimen ini perlu berfokus pada perubahan ukuran latent.

Angka reconstruction loss pada test set akan menjadi ukuran pendamping. Saya tidak akan menilai hasil hanya dari total VAE loss karena penjumlahan KL berubah menjadi sepuluh komponen, sehingga membaca perubahan totalnya memerlukan pemeriksaan setiap bagian. Target yang ingin saya lihat cukup konkret: digit 4 tetap terbuka di bagian atas, garis pembentuk angka 5 lebih terjaga, dan reconstruction loss test lebih rendah daripada model dua dimensi. Saat ini semuanya masih berupa hipotesis, karena project ini belum menjalankan training VAE 10D.

## 5. Prediksi Kualitas Visualisasi Latent pada 10 Dimensi

Visualisasi langsung justru menjadi lebih sulit ketika latent space memakai sepuluh dimensi. Pada hasil sekarang, saya cukup melihat sumbu z1 dan z2 untuk membaca lokasi setiap gambar test. Seluruh koordinat mean posterior masuk ke satu scatter plot. Kemudahan ini akan hilang pada model 10D karena layar dua dimensi tidak dapat memperlihatkan seluruh koordinat latent secara langsung. Menurut saya, tambahan ruang untuk menyimpan informasi membawa biaya berupa berkurangnya kemudahan membaca representasi.

Saya tidak akan sekadar mengambil dua koordinat pertama lalu menganggapnya mewakili semua informasi. Misalnya, encoder 10D bisa menyimpan perbedaan bentuk digit 4 dan 9 pada koordinat lain. Apabila saya hanya menggambar z1 dan z2, kedua kelompok mungkin tetap tampak bertumpuk meskipun decoder sudah lebih mampu membedakannya. Saya akan keliru menilai model apabila menyamakan overlap pada potongan dua dimensi dengan overlap pada seluruh ruang latent.

Pilihan awal saya adalah memakai PCA untuk memproyeksikan mean posterior 10D menjadi dua komponen. Dengan cara itu, saya masih bisa memberi warna berdasarkan label dan membandingkan susunan kelompok. Namun, dua sumbu hasil PCA tidak sama dengan z1 dan z2 milik encoder. Saya harus menamai sumbunya sebagai komponen utama dan memeriksa seberapa banyak variasi data yang proyeksi tersebut pertahankan. Informasi yang tidak masuk ke dua komponen itu tetap bisa berguna bagi decoder.

Pengalaman membaca plot 2D sekarang membuat saya ingin lebih hati-hati terhadap gambar yang tampak rapi. Kelompok digit 1 terlihat cukup terpisah, tetapi sebagian digit lain tetap bercampur dan mengalami kesalahan reconstruction. Sebaliknya, pada model 10D nanti, cluster yang tampak bertumpuk setelah proyeksi belum tentu berarti model gagal menyimpan perbedaan bentuk. Saya akan melihat reconstruction dari titik yang berdekatan sebelum menafsirkan posisi mereka terlalu jauh.

Jadi, prediksi saya adalah kualitas representasinya berpeluang membaik, sedangkan kualitas visualisasi dalam arti kemudahan interpretasi langsung akan berkurang. Jika tujuan saya menjelaskan cara kerja VAE saat presentasi, plot dua dimensi tetap lebih mudah saya terangkan. Untuk model sepuluh dimensi, saya perlu menjelaskan proses proyeksi dan batasnya agar pembaca tidak menganggap gambar dua dimensi tersebut sebagai gambaran lengkap ruang latent.

## 6. Prediksi Kemampuan Sampling pada 10 Dimensi

Saya berharap sampling pada latent space sepuluh dimensi menghasilkan variasi tulisan yang lebih kaya. Pada grid model 2D sekarang, saya melihat banyak bentuk menyerupai angka 1 di bagian atas dan angka 0 di bagian kanan. Bentuk berubah secara bertahap ketika koordinat bergeser, tetapi beberapa lokasi menghasilkan digit campuran yang sulit saya baca. Hasil tersebut menunjukkan bahwa decoder sudah bisa menghasilkan pola tulisan tanpa gambar input, walaupun pilihan bentuknya masih terbatas oleh representasi dua dimensi.

Dengan sepuluh koordinat, saya berharap model lebih mampu menyimpan variasi dalam satu jenis digit. Contohnya, model mungkin menghasilkan beberapa bentuk angka 4 dengan perbedaan bukaan bagian atas, tanpa terlalu mudah mengarah ke bentuk angka 9. Harapan ini berhubungan dengan kegagalan reconstruction yang sudah saya lihat. Jika decoder menerima representasi yang lebih mampu membedakan ciri tulisan, koordinat baru berpeluang menghasilkan gambar yang lebih beragam sekaligus tetap mudah dikenali.

Cara sampling dasarnya tetap bisa memakai vektor dari distribusi normal standar. Perbedaannya, setiap vektor sekarang berisi sepuluh angka. Namun, memperbanyak koordinat tidak menjamin semua sampel akan lebih bagus. Kualitas gambar tetap bergantung pada kecocokan wilayah prior yang saya ambil dengan wilayah latent yang decoder pelajari selama training. Saya akan membedakan dugaan tentang bertambahnya kapasitas model dari bukti bahwa sampel baru benar-benar lebih baik.

Bagian yang paling berubah bagi saya justru cara menjelajahi hasil. Grid 10 × 10 sekarang mewakili variasi kedua koordinat latent. Pada model 10D, grid dengan dua sumbu hanya memperlihatkan satu irisan jika saya menahan delapan koordinat lain pada nilai tetap. Saya tidak bisa menganggap irisan tersebut mewakili seluruh kemampuan generasi model.

Untuk membandingkan sampling, saya akan membuat kumpulan 100 sampel acak dari prior masing-masing model dan memeriksa keterbacaan serta variasi bentuknya. Saya juga ingin mencoba perpindahan bertahap antara dua vektor latent lengkap, agar saya dapat melihat apakah transisi mempertahankan bentuk tulisan yang masuk akal. Jumlah gambar tetap sama supaya perbandingannya tidak berat sebelah.

Saya tidak berharap sepuluh dimensi otomatis menghasilkan sepuluh kelas digit dengan jumlah seimbang. Model ini tidak menerima label sebagai syarat generasi. Ukuran latent dan jumlah jenis digit merupakan dua hal berbeda, walaupun keduanya kebetulan sama-sama bernilai sepuluh pada hipotesis tersebut.

## 7. Kritik Model VAE Dibandingkan Autoencoder Biasa

Kelebihan VAE yang paling terasa bagi saya adalah kemampuannya memakai latent space untuk menghasilkan gambar baru dengan prosedur yang jelas. Saya melihatnya langsung pada grid 10 × 10: decoder menerima pasangan koordinat tanpa gambar input, lalu menghasilkan berbagai bentuk tulisan tangan. Beberapa bentuk memang kurang jelas, tetapi perubahan antarposisi masih memperlihatkan hubungan visual. Hasil itu membuat saya memahami manfaat prior normal standar lebih baik daripada jika saya hanya membaca rumus loss.

Pada autoencoder biasa tanpa regularisasi distribusi latent, saya tidak otomatis mempunyai alasan untuk menganggap vektor acak dari normal standar akan berada di wilayah yang decoder kenal. Model tersebut belajar memulihkan input dari kode encoder, tetapi tujuan itu sendiri tidak menjamin susunan kode sesuai dengan prior tertentu. Karena project ini belum melatih autoencoder pembanding, saya menempatkan bagian tersebut sebagai penilaian terhadap rancangan model, bukan hasil pengujian langsung.

Kelemahan VAE yang saya lihat adalah kesetiaannya terhadap detail input. Reconstruction digit 4 berubah menyerupai 9 dan digit 5 menyerupai 8. Bagi kebutuhan yang mengutamakan pemulihan gambar, perubahan identitas seperti itu cukup mengganggu. Saya tidak akan menilai hasilnya memadai hanya karena output masih terlihat seperti sebuah angka. Model perlu mempertahankan angka yang sama dengan input.

Saya menduga autoencoder biasa dengan kapasitas sebanding dapat menghasilkan reconstruction yang lebih setia karena objective-nya dapat berfokus pada reconstruction tanpa penalti KL. Tetapi saya belum punya hasil pembanding untuk menyatakan bahwa model tersebut pasti mengalahkan VAE ini. Membandingkan VAE 2D dengan autoencoder berdimensi jauh lebih besar juga tidak akan menjawab pertanyaan secara adil. Saya perlu menjaga kapasitas dan konfigurasi training sedekat mungkin.

Ada pula kesulitan dalam menilai keberhasilan training VAE. Total loss menggabungkan dua tujuan yang dapat bergerak berbeda. Pada eksperimen saya, reconstruction loss terus turun, sedangkan KL naik perlahan setelah epoch kedua. Jika saya hanya memeriksa salah satu komponen, penilaian saya bisa meleset. Pemeriksaan gambar tetap saya perlukan, terlebih karena rata-rata loss dapat menyembunyikan beberapa reconstruction yang gagal.

Pilihan model saya akhirnya bergantung pada tugasnya. Untuk percobaan generasi tulisan dan pembacaan perpindahan bentuk melalui latent space, saya memilih VAE. Jika tujuan utamanya menyalin kembali input seakurat mungkin, saya akan menguji autoencoder biasa sebagai pembanding serius. Hasil VAE ini cukup untuk menunjukkan kemampuan generatifnya, tetapi belum cukup untuk menyatakan bahwa regularisasi latent selalu sepadan dengan detail gambar yang hilang.

