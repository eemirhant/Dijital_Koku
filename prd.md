DijitalKoku (Olfactory UI)
Sürüm: 2.1 (Pencere Seçici ve Görsel Takip Odaklı)

1. Proje Tanımı
DijitalKoku; seçilen bir uygulama penceresini veya ekran bölgesini analiz ederek, 7 temel esansın mikro-karışımlarıyla sahneye özgü atmosferik kokular sentezleyen bir donanım/yazılım ekosistemidir.

2. Yazılım Mimarisi ve Kontrol Paneli (Yeni!)
2.1. Dinamik Pencere Seçici (Window Selector)
Kontrol Ekranı: Kullanıcı, Dashboard üzerinden aktif çalışan pencerelerin listesini (Örn: Chrome, VLC Player, GTA V) görebilecek ve analiz edilecek hedefi seçecektir.

Görsel Geri Bildirim (Bounding Box): Analiz başladığında, seçilen pencere veya ekran bölgesi kullanıcıya renkli bir çerçeve içinde gösterilecek (OpenCV Overlay). Bu, sistemin hangi alanı "gördüğünün" anlık doğrulamasıdır.

2.2. Analiz ve İzleme Katmanı
Tahmin İzleyici (Scent Predictor): Kontrol ekranında, "Şu an algılanan koku profili" başlığı altında, sistemin o sahne için belirlediği baskın koku ve karışım oranları metinsel ve grafiksel olarak verilecektir (Örn: "Tahmin: %80 Yangın, %20 Metalik").

3. Gelişmiş Karar Motoru
(Önceki versiyondaki Hibrit Karar, Override Logic ve Smoothing kuralları aynen geçerlidir.)

4. Donanım ve Haberleşme
(Arduino Uno + PCA9685 yapısı ve Seri Protokol detayları korunmuştur.)

5. Geliştirme Fazları (Onaylı İlerleme)
🔵 Faz 1: Kontrol Paneli, Seçici ve Karar Motoru (Mevcut Odak)
Dashboard: Pencere seçme listesi, seçili alanın çerçeveli önizlemesi ve koku tahmin barları.

Onay Şartı: Pencere değiştirildiğinde sistemin yeni pencereyi çerçeve içine alması ve koku tahminini 100ms içinde güncellemesi.

Antigravity İçin Güncellenmiş "Başlangıç" Promptu
Bu yeni PRD'yi yükledikten sonra şu komutu vererek projeyi başlatabilirsin:

"Projeye yeni baştan başlıyoruz. Ekteki V2.1 PRD dökümanını ana rehber olarak al.

İLK GÖREV (Faz 1): > 1.  pygetwindow veya benzeri bir kütüphane ile çalışan pencereleri listeleyen bir Streamlit seçici yap.
2.  Seçilen pencereyi dxcam ile yakala ve kullanıcının hangi alanı analiz ettiğini görmesi için o pencereyi bir çerçeve (bounding box) içine alarak Dashboard'da göster.
3.  YOLOv8 ve K-Means kullanarak seçili alanı analiz et.
4.  Karar motorun yanan araba/vanilya gibi hataları engelleyecek 'Override' ve 'Conflict Resolver' mantığına sahip olsun.
5.  Kontrol ekranında 'Sistem Tahmini' olarak hangi kokuların üretileceğini (Örn: %70 Deniz, %30 Limon) anlık yazdır.

Ben bu görsel seçiciden ve tahmin tutarlılığından memnun kalana kadar donanım aşamasına geçmeyeceğiz. Başlayalım."

Bu Eklemeler Neyi Çözecek?
Odaklanma: Tüm ekranı analiz edip gereksiz ikonları/masaüstü resmini işleme yükünden kurtulacağız. Sadece seçtiğin videoyu veya oyunu analiz edecek.

Hata Tespiti: Eğer vanilya tahmini yapıyorsa, çerçeve içindeki görüntüye bakıp "Ha, sistem şu an siyah asfaltı görüyor, o yüzden yanılıyor" diyebileceğiz.

Profesyonellik: Jürinin önünde "Şu pencereyi seçiyorum" deyip kutu içine alman, projenin yazılımsal olgunluğunu %100 artırır.