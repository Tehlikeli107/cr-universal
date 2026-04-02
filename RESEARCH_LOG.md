# Architecture Research Log

## Denenen Paradigmalar ve Ogrenilenler

### 1. Attention (Transformer) — BASELINE
- QK^T dot product + softmax + V weighted sum
- Gucu: HER turdeki iliskiyi ogrenebiliyor (universalO
- Zayifligi: O(N^2), lineer similarity varsayimi
- OGRENILEN: Attention = pairwise linear routing. Basit ama etkili.

### 2. NEXUS (Relation Encoding) — +3.8% mixed rule
- Komsu tokenlar arasi diff/prod hesapla
- Heterogeneous window sizes
- OGRENILEN: EXPLICIT relation encoding attention'a bilgi EKLIYOR.
  Ama lokal window = sinirli menzil.

### 3. BlackBox (State-based) — KAYBETTI
- state = gelu(W @ state + token)
- OGRENILEN: Tek state = bilgi UNUTUYOR. Uzun menzil IMKANSIZ.
  increment %100 (tek adim) ama copy %3 (cok adim).
  STATE = SIKISTIRMA. N token'i 1 vektore = cok kayip.

### 4. LearnedPair (Ogrenilmis Baglanti) — esit (param ayarinda)
- f(xi, xj) = MLP(cat(xi, xj)) yerine dot product
- OGRENILEN: Daha expressive AMA daha fazla param gerekiyor.
  Dot product UCUZ ve VERIMLI. MLP PAHALI.
  Per-param verimlilik: attention > learned pair.

### 5. Adaptive Order (Pair+Triple) — +4.6% copy, +3.7% hard
- Ikili + uclu baglanti, per-position secim
- OGRENILEN: Triple connection BELIRLI patternlarda (copy, 3-token)
  avantajli. Ama genel icin pair yeterli. Zorluk arttikca
  higher-order DAHA FAYDALI.

### 6. Space Bender (Uzay Bukme) — ESIT (param ayarinda)
- Warp matrix ile uzayi buk, lokal islem yap
- OGRENILEN: O(N) ama ayni param'da transformer'la ESIT.
  Uzay bukme = nonlineer embedding degisimi. Guclu ama
  attention'in SELECTIVITY'sini saglamiyor.

### 7. Wave (FFT) — +36% AYNI PARAMDA!
- FFT -> frekans uzayinda islem -> iFFT
- OGRENILEN: Copy = faz kaymasi, Sum = konvolusyon.
  FFT bunlari DOGAL yapior. Attention OGRENMEK ZORUNDA.
  **EN BUYUK BULGU: Task'in DOGAL uzayinda islem yapmak CRITICAL.**
  FFT frekans uzayinda, attention embedding uzayinda.
  Eger task frekans tabanliysa FFT EZICI ustun.

### 8. Compress (Sikistirma) — +40% ama param fazla
- N token -> K token -> isle -> geri ac
- OGRENILEN: Sikistirma ZORLA global bilgi yakaliyor.
  K=8, N=19: 2.4x sikistirma. Iyi calisiyor ama param fazla.

### 9. Gravity (Fiziksel Dinamik) — KAYBETTI
- Token = parcacik, cekim/itme ile hareket
- OGRENILEN: Fiziksel dinamik BENZERLIK buluyor (copy %90)
  ama ARITMETIK YAPAMIYOR (sum %18). Cekim = benzerlik,
  toplama != cekim. **Fiziksel metafor SINIRLI.**

## DERIN PATTERN'LAR

### Pattern 1: Task'in dogal uzayi
FFT'nin +%36 kazanmasi = TASK'IN DOGAL UZAYINDA ISLEM YAPMAK.
Copy = shift = frekans. Sum = konvolusyon = frekans. FFT = frekans uzayi.
Attention = embedding uzayi (task'in dogal uzayi DEGIL).
**SORU: Her task'in "dogal uzayi" ne? Bunu OTOMATIK bulmak = en iyi mimari.**

### Pattern 2: Selectivity vs Efficiency
Attention: SELECTIVE (hangi token'a bakacagini SECER) ama PAHALI (O(N^2)).
FFT: EFFICIENT (O(N log N)) ama NON-SELECTIVE (TUM frekanslari etkiler).
Space Bend: EFFICIENT (O(N)) ama NON-SELECTIVE.
**SORU: Selective + Efficient mumkun mu? Mamba bunu yapiyor (selective state space).**

### Pattern 3: Aritmetik vs Benzerlik
Attention, Gravity, Space Bend = BENZERLIK tabanlı (dot product, cekim, yakinlik).
FFT, Compress = DONUSUM tabanli (frekans donusumu, sikistirma).
Benzerlik = copy, pattern matching icin iyi.
Donusum = aritmetik, reasoning icin iyi.
**SORU: Ikisini birlestirmek = her iki turu de yapabilmek?**

### Pattern 4: Param verimliligi
Attention cok param-verimli (dot product = 0 ek param, sadece QKV projeksiyonlari).
Learned functions (MLP-based) param-verimsiz (her fonksiyon icin yeni weight'ler).
**SORU: Param-verimli ama expressive mekanizma var mi?**

### 10. LearnedBasis — %85.5 (Transformer'dan iyi, FFT'den kotu)
- Ogrenilmis T matrisi. Causal (alt ucgen).
- OGRENILEN: FFT'nin sin/cos'u bu task icin ZATEN optimal.
  Ogrenilmis baz yaklasıyor ama tam yakalayamiyor.
  FARKLI task'ta ogrenilmis baz FFT'yi YENEBILIR.

### 11. FreqAttention — KOTU (%47.1)
- FFT -> attention in freq domain -> iFFT
- OGRENILEN: CALISMADI. Frekans uzayinda pozisyon bilgisi YOK.
  Selective + frequency = UYUMSUZ.

## DENENMEMIS FIKIRLER
1. FFT + Attention hybrid (frekans uzayinda selective attention)
2. Learned basis functions (FFT'nin cos/sin'i yerine ogrenilmis baz)
3. Multi-resolution (ayni anda farkli olceklerde islem)
4. Token interaction graph (token'lar arasi baglanti GRAFI ogren, sonra GNN)
5. Information bottleneck (her layer'da bilgiyi KASITLI sikistir)
6. Recursive computation (sabit derinlik yerine YAKINSAMAYA KADAR islem)
