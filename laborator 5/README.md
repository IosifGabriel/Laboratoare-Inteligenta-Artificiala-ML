Funcția normalize_data(train_data, test_data, type=None) primește ca
parametri datele de antrenare, respectiv de testare și tipul de normalizare ({None,
‘standard’, ‘min_max’, ‘l1’, ‘l2’}) și întoarce aceste date normalizate. 

Clasa BagOfWords  al cărui constructor se inițializează vocabularul (un
dicționar gol). În cadrul ei este implementata metoda build_vocabulary(self, data) care
primește ca parametru o listă de mesaje(listă de liste de strings) și construiește
vocabularul pe baza acesteia. 

Cheile dicționarului sunt reprezentate de cuvintele din
eseuri, iar valorile de id-urile unice atribuite cuvintelor. 

Metoda get_features(self, data)  primește ca parametru o listă de
mesaje de dimensiune 𝑛𝑢𝑚_𝑠𝑎𝑚𝑝𝑙𝑒𝑠(listă de liste de strings) și returnează o matrice
de dimensiune (𝑛𝑢𝑚_𝑠𝑎𝑚𝑝𝑙𝑒𝑠 x 𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑎𝑟𝑦_𝑙𝑒𝑛𝑔𝑡ℎ) definită astfel:
𝒇𝒆𝒂𝒕𝒖𝒓𝒆𝒔(𝒔𝒂𝒎𝒑𝒍𝒆_𝒊𝒅𝒙,𝒘𝒐𝒓𝒅_𝒊𝒅𝒙) = 𝒏𝒖𝒎𝒂𝒓𝒖𝒍 𝒅𝒆 𝒂𝒑𝒂𝒓𝒊𝒕𝒊𝒊 𝒂𝒍
 𝒄𝒖𝒗𝒂𝒏𝒕𝒖𝒍𝒖𝒊 𝒄𝒖 𝒊𝒅− 𝒖𝒍 𝒘𝒐𝒓𝒅_𝒊𝒅𝒙 𝒊𝒏 𝒅𝒐𝒄𝒖𝒎𝒆𝒏𝒕𝒖𝒍 𝒔𝒂𝒎𝒑𝒍𝒆_𝒊𝒅
 
Se antreneaza un SVM cu kernel linear care să clasifice mesaje în mesaje spam/nonspam. Parametrul C are valoarea 1.
Se calculeaza acuratetea F1
Se afiseaza primele 10 cuvinte SPAM SI NON-SPAM.