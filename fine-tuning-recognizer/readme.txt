install tensorflow 2.3.0 dan keras 1.0.7

pip install tensorflow==2.3.0
pip install keras==1.0.7

jalankan :

python3 fine_tuning_recognizer.py

dataset borndigital akan didownload dan terbentuk direktory /borndigital/train dan /borndigital/test
selanjutnya akan run training character recognizer dengan menggunakan data set /borndigital

Bagaimana jika kita akan melatih dgn huruf huruf baru? misalnya dengan huruf huruf plat nomer Indonesia?

misalnya direktori borndigital saya copy jadi /wordset, maka :
1. Tambahkan gambar gambar dalam /wordset/train dan /wordset/test
2. Bikin list dari /wordset/train dan /wordset/test misalnya :

train_labels = [('./wordset/train/word_1.png', None, 'flying'), ('./wordset/train/word_2.png', None, 'today'), ('./wordset/train/word_3.png', None, 'means'), ('./wordset/train/word_4.png', None, 'vueling'), ('./wordset/train/word_5.png', None, 'get'), ('./wordset/train/word_6.png', None, 'away,'), ('./wordset/train.
...
('./wordset/train/platAB.png', None, 'AB1234MN'), ('./wordset/train/platjakarta.png', None, 'B34468ZXC'), ('./wordset/train/platbandung.png', None, 'D160DA,')]


test_labels =[('./wordset/test/word_1.png', None, 'bada'), ('./wordset/test/word_2.png', None, 'developer'), ('./wordset/test/word_3.png', None, 'day'), ('./wordset/test/word_4.png', None, 'hhonors'), ('./wordset/test/word_5.png', None, 'hilton'), ('./wordset/test/word_6.png', None, 'worldwide'), ('./wordset/test/word_7.png', ...
('./wordset/train/platsmg.png', None, 'H997TG'), ('./wordset/train/plattegal.png', None, 'G10ITB')]

contohnya :

python3 fine_tuning_recognizer_custom.py
