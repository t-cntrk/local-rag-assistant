# local-rag-assistant

Tamamen **yerel (offline)** çalışan, belge tabanlı bir soru-cevap asistanı.
Örnek senaryo: "KahveUsta 300" adlı kurgusal bir akıllı kahve makinesinin
kullanım kılavuzu, garanti belgesi ve sorun giderme dokümanları üzerinden
Türkçe destek veren bir asistan.

Mimari **RAG** (Retrieval-Augmented Generation): model, kendi ezberinden değil,
belgelerden getirilen ilgili parçalardan cevap üretir. Hiçbir veri internete
çıkmaz; hem embedding hem dil modeli [Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/)
ile bilgisayarda çalışır.

Asistanın **iki ayrı reddetme davranışı** vardır:

| Durum | Cevap |
|---|---|
| Ürünle ilgili ama belgelerde olmayan bilgi (ör. "Fiyatı ne kadar?") | *Bu bilgi elimdeki belgelerde bulunmuyor.* |
| Ürünle tamamen alakasız istek (ör. "Bana masal anlat.") | *Ben yalnızca KahveUsta 300 hakkında yardımcı olabilirim.* |

Her iki durumda da kaynak gösterilmez.

---

## Tanıtım videosu

Projenin çalışırken anlatıldığı kayıt: [`docs/proje-tanitimi.mp4`](docs/proje-tanitimi.mp4)

---

## Mimari

İki ayrı akış var.

**1. Ingestion (bir kez, `src/ingest.py`)**

```
data/documents/*.md|*.txt
      → başlık bazlı parçalama (chunk)
      → her chunk için embedding (qwen3-embedding-0.6b, 1024 boyut)
      → SQLite'a yazma (knowledge.db, embedding JSON olarak)
```

**2. Sorgu (her soruda, `main.py`)**

```
kullanıcı sorusu
      → retrieval sorgusu = ürün bağlamı öneki + instruction prefix + soru
      → sorgu embedding'i
      → tüm chunk'larla kosinüs benzerliği (numpy)
      → en iyi 3 chunk
      → bağlam + ASIL soru → yerel LLM (qwen3-4b)
      → cevap + kaynak dosyalar
```

### Sorgu tarafı: instruction prefix + sorgu genişletme

İki katman var, ikisi de yalnızca **retrieval embedding'ini** etkiler; chat
modeline giden ve kullanıcıya gösterilen soru olduğu gibi kalır.

**a) Asimetrik retrieval (instruction prefix).** Qwen3-Embedding'in resmî model
kartı, *sorgu* tarafında bir görev talimatı kullanılmasını önerir, *belgelerde*
ise talimat istemez. Görev tanımı model kartının önerisi gereği İngilizcedir;
sorunun kendisi Türkçe kalır. Belgeler düz metin olarak embed'lenir.

**b) Sorgu genişletme (ürün bağlamı öneki).** Soru, embed edilmeden önce sabit
bir ürün bağlamıyla genişletilir:

```
Instruct: {görev tanımı}
Query:KahveUsta 300 akıllı kahve makinesi hakkında soru: {soru}
```

Neden: kısa ve eksik kurulmuş sorgular ("renkler neler") tek başına embed
edildiğinde ayırt edici bir vektör üretmiyordu — tüm chunk skorları 0,38–0,44
arasında sıkışıyor, sıralama gürültüye dönüyor ve doğru parça ilk üçe
giremiyordu. Ürün bağlamı eklemek aynı parçayı 15/40'tan 1/40'a çıkardı; düzgün
kurulmuş sorgular ise 1. sırada kalmaya devam etti (bkz. `diag_query_expansion.py`).

### Uydurmaya karşı korumalar

Cevap üretimi `temperature=0.2` ile yapılır. Güvence bir skor eşiğine değil,
**sistem prompt'undaki kurallara** dayanır:

1. **Kapsam kuralı.** Asistan yalnızca KahveUsta 300 hakkında yardımcı olur.
   Masal/şiir yazma, genel kültür, kod, sohbet gibi istekler kapsam dışıdır ve
   sabit kapsam reddiyle karşılanır.
2. **Belgede yoksa reddet.** İstek ürünle ilgiliyse ama bağlamda karşılığı
   yoksa, model sabit "belgelerde bulunmuyor" cümlesini verir.
3. **Özellik-yokluğu kuralı.** Bir özelliğin var olup olmadığı sorulduğunda
   ("su geçirmez mi?", "X özelliği var mı?") ve bağlamda o özellikten hiç söz
   edilmiyorsa model yorum yapmaz, tahmin yürütmez — doğrudan reddeder. Bir
   özelliğin bağlamda geçmemesi, o özelliğin olmadığı anlamına gelmez. Bu kural
   uydurmanın yoğunlaştığı tek noktayı hedefler: model, bilgi yokluğunu bir
   özellik iddiasına çevirme eğilimindeydi.
4. **Anlamsal reddetme tanıma.** `rag.is_refusal()`, reddi birebir cümle
   eşleşmesiyle değil anlamca tanır; model reddi kendi kelimeleriyle kurduğunda
   da ("fiyat bilgisi belgelerde yer almıyor") yakalar. Reddedilen cevaplarda
   `answered=False` olur ve kaynak gösterilmez.

**Ölçüm.** Cevaplanamaz sorularda yapılan tekrarlı denemelerde uydurma
gözlenmedi: hedefli prompt kuralından sonra 15/15 (`diag_hallucination.py`),
sorgu genişletme doğrulamasında 30/30 (A ve B kolları,
`diag_expansion_hallucination.py`). Bununla birlikte bu bir LLM sistemidir ve
güvence prompt kurallarına dayandığı için **%100 garanti verilemez**; ölçülmemiş
soru biçimlerinde farklı davranabilir.

### Klasör yapısı

```
local-rag-assistant/
├── data/documents/          # kaynak belgeler (7 adet, .md ve .txt)
├── src/
│   ├── config.py            # model alias'ları, yollar ve TOP_K gibi ayarların tek merkezi
│   ├── db.py                # SQLite katmanı: chunks tablosunu oluşturur, chunk yazar/okur
│   ├── foundry.py           # Foundry Local ortak yardımcısı: modeli yükler, OpenAI uyumlu istemciyi ve çözümlenmiş model id'sini verir, embedding üretir
│   ├── ingest.py            # belgeleri parçalar, embedding'lerini üretir ve veritabanını sıfırdan doldurur
│   ├── retrieve.py          # sorguyu genişletip embed'ler ve kosinüs benzerliğiyle en iyi K chunk'ı skorlarıyla döndürür
│   └── rag.py               # bağlamı kurar, sistem prompt'uyla modele sorar; cevabı, reddetme durumunu ve kaynakları üretir
├── scripts/
│   ├── hello_model.py       # minimal "model çalışıyor mu" testi
│   ├── test_retrieval.py    # sadece retrieval katmanını 5 örnek soruyla test eder
│   ├── test_rag.py          # uçtan uca RAG'i 5 örnek soruyla test eder
│   └── diag_*.py            # geçmiş kararları destekleyen ölçüm scriptleri (aşağıya bakınız)
├── main.py                  # CLI: modelleri hazırlar ve soru-cevap döngüsünü çalıştırır
├── knowledge.db             # ingestion çıktısı (sürüm kontrolüne girmez)
└── requirements.txt
```

### Belgeler hakkında

`data/documents/` altındaki yedi belge kurgusaldır. Bazı bilgiler **kasıtlı
olarak yoktur** — fiyat, su geçirmezlik ve elektrik tüketimi (watt/kWh); bunlar
"bilmiyorum" davranışını sınamak için ayrılmıştır.

Renk bilgisi başlangıçta "Genel Bilgiler" bölümünün içine gömülüydü ve renk
sorularında retrieval bunu getiremiyordu; `01_teknik_ozellikler.md` içine ayrı
bir **"Renk Seçenekleri"** bölümü eklendi. Bu tek başına yetmedi — asıl çözüm
sorgu genişletme oldu (bkz. yukarısı ve `diag_colors.py`).

---

## Kullanılan modeller

| Rol | Model | Neden |
|---|---|---|
| Sohbet / cevap üretimi | `qwen3-4b` | Türkçesi iyi, bağlama sadık kalıyor ve CPU'da çalışabilecek kadar küçük |
| Embedding | `qwen3-embedding-0.6b` | 1024 boyutlu, çok dilli, hızlı; Türkçe sorularda doğru parçaları getiriyor |

Başlangıçta chat modeli `qwen3-1.7b` idi; test setinde 5 sorudan yalnızca 1'ini
doğru cevapladığı (bağlamda açıkça yazan garanti süresini görememesi, sayı
uydurması ve kendini tekrar eden bozuk çıktılar üretmesi) için `qwen3-4b`'ye
geçildi — aynı testte 5/5.

Model alias'ları `src/config.py` içinde tek satırda değiştirilebilir.

---

## Kurulum (Windows)

Gereksinimler: Windows, **Python 3.11+** (bu proje 3.14.4 ile test edildi).

```powershell
# 1) Foundry Local CLI
winget install Microsoft.FoundryLocal

# 2) Sanal ortam
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Bağımlılıklar
pip install -r requirements.txt
```

Modeller ayrıca indirilmez; ilk çalıştırmada Foundry Local otomatik indirir
(`qwen3-embedding-0.6b` ~495 MB, `qwen3-4b` ~2,5 GB). Bu yalnızca bir kez olur.

---

## Çalıştırma

**1. Veritabanını doldur** (belgeler değiştiğinde tekrarla):

```powershell
.\.venv\Scripts\python.exe src\ingest.py
```

Beklenen çıktı: 7 belge → **40 chunk**, embedding boyutu 1024.
Bu komut `knowledge.db`'yi her seferinde sıfırdan oluşturur.

**2. Asistanı başlat:**

```powershell
.\.venv\Scripts\python.exe main.py
```

Örnek oturum (üç davranışı da gösterir):

```
======================================================================
  KahveUsta 300 — Yerel Destek Asistanı
======================================================================
Hazır. Sorunuzu yazın. (Çıkmak için 'çıkış' veya boş satır)

Soru> renkler neler

KahveUsta 300, paslanmaz çelik ve mat siyah renk seçenekleriyle sunulur.
Renkler arasında işlev farkı yoktur.

Kaynaklar: 01_teknik_ozellikler.md

Soru> Fiyatı ne kadar?

Bu bilgi elimdeki belgelerde bulunmuyor.

Soru> Bana masal anlat.

Ben yalnızca KahveUsta 300 hakkında yardımcı olabilirim.

Soru>
```

Çıkmak için `çıkış`, `exit` veya boş satır.

---

## Testler

Rutin doğrulama için:

```powershell
.\.venv\Scripts\python.exe scripts\test_retrieval.py   # sadece retrieval
.\.venv\Scripts\python.exe scripts\test_rag.py         # uçtan uca (5 soru)
```

### Ölçüm (diag) scriptleri

Bunlar rutin test değil; projedeki bazı kararların **neden** böyle alındığını
gösteren tek seferlik ölçümlerdir. Yavaştırlar (bazıları 30–75 dakika), ama bir
ayarı değiştirmeden önce tekrar çalıştırmak için referans olarak duruyorlar.

| Script | Ne ölçtü | Sonuç |
|---|---|---|
| `diag_colors.py` | "renkler neler" sorusu neden cevaplanamıyordu; renk chunk'ının sıralamadaki yeri | Arıza retrieval'daydı: hedef chunk 12/40'ta kalıyordu. Belgeye ayrı "Renk Seçenekleri" bölümü eklemek yalnızca düzgün kurulmuş sorguyu düzeltti |
| `diag_hallucination.py` | "Bilmiyorum" güvencesinin kararlılığı; üretim ayarının etkisi | `temperature=0.0` uydurmayı kesiyor ama cevaplanabilir soruları yanlış reddediyor → **0,2 korundu**. Hedefli prompt kuralı + anlamsal `is_answered` sonrası cevaplanamaz sorularda 15/15 |
| `diag_query_expansion.py` | Retrieval sorgusuna ürün bağlamı eklemenin etkisi (yalnızca skorlar) | Bozuk sorgular düzeldi ("renkler neler" 15/40 → 1/40), çalışan sorgular 1. sırada kaldı. Yan etki: tüm skorlar şişiyor, ilk üçün 2.-3. sıraları zayıflıyor |
| `diag_expansion_hallucination.py` | Genişletme uydurma güvencesini bozuyor mu (uçtan uca) | Bozmuyor: cevaplanamaz sorular A 15/15, B 15/15. Cevaplanabilir A 14/20 → B 19/20. Bu ölçümden sonra genişletme kalıcı hâle getirildi |

**Not:** `diag_query_expansion.py` ve `diag_expansion_hallucination.py` içindeki
"A: mevcut" modu, genişletme kalıcı hâle gelmeden önceki davranışı temsil eder.
Genişletme artık `retrieve.embed_query` içinde olduğu için bu scriptlerin "A"
kolu **artık üretimdeki davranış değildir**; tarihsel karşılaştırma olarak okuyun.

---

## Bilinen sınırlar

- **Hız.** Modeller CPU'da çalıştığı ve `qwen3-4b` düşünme (thinking) modunu
  kullandığı için soru başına **~40–90 saniye** sürer.
  `rag.answer_query(soru, think=False)` düşünme adımını kapatır ve hızlandırır,
  ama **önerilmez**: ölçümde bu mod "su geçirmez mi?" sorusunda 5 denemenin
  5'inde de uydurdu.
- **Güvence prompt'a dayanır, skor eşiği yoktur.** Alakasız sorularda bile en
  yakın 3 chunk modele verilir; reddetme kararını model verir. Ayrı bir
  sınıflandırıcı yoktur — kapsam kararı da modelindir.
- **Ham benzerlik skorları bir "güven" ölçüsü değildir.** Sorgu genişletme tüm
  skorları globalde yükseltti (her chunk ürün adını içerdiği için), bu yüzden
  cevaplanabilir ve cevaplanamaz sorular arasındaki fark daraldı. Skorlar
  sıralama içindir; mutlak değerlerine bakarak "bu cevap güvenilir" sonucu
  çıkarılmamalıdır. Farklı sürümlerde ölçülen skorlar da birbiriyle
  kıyaslanamaz.
- **Model boyutu ve Türkçe kusurlar.** 4B parametreli bir modelin Türkçesi akıcı
  ama kusursuz değil; nadiren kelime içinde harf düşürebiliyor (ör. "yöntemdir"
  yerine "yöntedir"). Bu bir token kesilmesi **değil**, modelin kendi üretim
  hatasıdır — ölçümlerde `finish_reason` her zaman `stop` döndü. Daha büyük
  modeller (`qwen3-8b`, `qwen2.5-7b`) katalogda var ama CPU'da belirgin şekilde
  yavaşlar.
- **Nadir fazladan reddetme.** Bazı sorularda (ölçümde "Garanti düşme sonucu
  hasarı kapsıyor mu?") model doğru cevabı verip sonuna gereksiz yere reddetme
  cümlesini ekleyebiliyor. Cevap doğru kalır, ancak anlamsal reddetme tanıma bu
  metni red sayar ve kaynak gösterilmez. 5 denemede 1 kez gözlendi.
- **Kaynak gösterimi yaklaşıktır.** Listelenen dosyalar, modele bağlam olarak
  verilen parçaların dosyalarıdır; modelin cümle cümle hangisini kullandığı
  izlenmez. Model reddettiğinde (her iki reddetme durumunda da) kaynak hiç
  gösterilmez.
- **Ingestion artımlı değildir.** `ingest.py` her çalıştığında tabloyu silip
  yeniden kurar.

---

## Olası geliştirmeler

- Streamlit veya basit bir web arayüzü (CLI yerine).
- Skor eşiğini **ikincil** güvenlik ağı olarak eklemek (prompt güvencesinin
  yanına, onun yerine değil) — genişletme sonrası skorların yeniden
  kalibrasyonu gerekir.
- Modelin fiilen kullandığı kaynağı işaretlemek (ör. modelden `[1]`, `[2]`
  biçiminde atıf istemek ve cevabı buna göre etiketlemek).
- Cevabı token token akıtmak (streaming) — uzun bekleme süresini gizler.
- Artımlı ingestion: yalnızca değişen belgeleri yeniden embed'lemek.
- `top_k`'yi yeniden değerlendirmek: sorgu genişletme 1. sırayı keskinleştirdi
  ama ilk üçün kuyruğunu zayıflattı.
