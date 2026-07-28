@echo off
cd /d "%~dp0"
title DijitalKoku V3 -- Baslatiliyor...

echo ============================================================
echo   DijitalKoku V3  Semantik-Olasiliksal Koku Motoru
echo   Katman B: VLM (moondream/Ollama)
echo   Katman C: Bayes Fuzyon    Katman D: Kalman Filtresi
echo ============================================================
echo.

:: [1/6] Onceki DijitalKoku surecleri temizle
echo [1/6] Onceki DijitalKoku surecleri kapatiliyor...
wmic process where "commandline like '%%dashboard.py%%'"     delete >nul 2>&1
wmic process where "commandline like '%%streamlit%%'"        delete >nul 2>&1
wmic process where "commandline like '%%video_worker.py%%'"  delete >nul 2>&1
wmic process where "commandline like '%%window_worker.py%%'" delete >nul 2>&1
wmic process where "commandline like '%%pdm_engine.py%%'"    delete >nul 2>&1
wmic process where "commandline like '%%analyzer.py%%'"      delete >nul 2>&1
taskkill /f /im streamlit.exe >nul 2>&1
for /f "tokens=5" %%p in ('netstat -aon 2^>nul ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p /T >nul 2>&1
)
if exist ".dk_pid" del /f /q ".dk_pid"
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" >nul 2>&1
echo      Surec temizligi tamam.
echo.

:: [2/6] Python varligini dogrula (kritik - eksikse hemen hata ver)
echo [2/6] Python kontrol ediliyor...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [HATA] Python bulunamadi!
    echo         python.org adresinden Python 3.10+ kurulmali.
    echo         Kurulumdan sonra cmd'yi kapatip tekrar deneyin.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo      %%v bulundu. OK
echo.

:: [3/6] Sanal ortam / V3 bagimliliklari hizli kontrol
echo [3/6] Python ortami ve V3 bagimliliklari kontrol ediliyor...
if exist ".venv\Scripts\activate.bat" (
    echo      Sanal ortam [.venv] aktif ediliyor...
    call .venv\Scripts\activate.bat
)
python -c "import requests, streamlit, cv2, ultralytics" >nul 2>&1
if errorlevel 1 (
    echo      Eksik paketler tespit edildi, yukleniyor...
    echo      [Bu islem ilk kurulumda birkac dakika surebilir, lutfen bekleyin]
    python -m pip install requests>=2.31.0
    if errorlevel 1 (
        echo      [UYARI] requests yuklenemedi. Devam ediliyor...
    ) else (
        echo      Eksik paketler yuklendi. OK
    )
) else (
    echo      Tum bagimliliklar hazir. OK
)
echo.

:: [4/6] Ollama VLM servisi kontrol / baslatma
echo [4/6] Ollama VLM servisi kontrol ediliyor...
where ollama >nul 2>&1
if errorlevel 1 (
    echo      [UYARI] ollama.exe PATH'te bulunamadi.
    echo              VLM katmani devre disi - sistem yine de calisir.
    echo              Ollama kurmak icin: https://ollama.com/download
    goto :ollama_done
)
tasklist /fi "imagename eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
if errorlevel 1 (
    echo      Ollama calismiyor, arka planda baslatiliyor...
    start /b "" ollama serve >nul 2>&1
    timeout /t 5 /nobreak >nul
    echo      Ollama baslatildi. OK
) else (
    echo      Ollama zaten calisiyor. OK
)

:: [5/6] moondream model kontrolu
echo.
echo [5/6] moondream modeli kontrol ediliyor...
ollama show moondream >nul 2>&1
if errorlevel 1 (
    echo      moondream lokalde bulunamadi, indiriliyor...
    echo      [Ag hizina gore 1-5 dakika surebilir, lutfen bekleyin]
    ollama pull moondream
    if errorlevel 1 (
        echo      [UYARI] moondream indirilemedi. VLM devre disi kalacak.
    ) else (
        echo      moondream basariyla indirildi. OK
    )
) else (
    echo      moondream zaten yuklu. OK
)

:ollama_done
echo.

:: [6/6] Streamlit arayuzu
echo [6/6] DijitalKoku V3 arayuzu baslatiliyor...
echo.
echo      NOT: CLIP modeli ilk yuklemede 30-60 sn surebilir.
echo      VLM arka planda her N karede bir calisir - FPS etkilenmez.
echo      Adres: http://localhost:8501
echo.
echo ============================================================
echo.

python -m streamlit run dashboard.py ^
    --server.port 8501 ^
    --server.headless false ^
    --server.runOnSave false ^
    --server.fileWatcherType none ^
    --browser.gatherUsageStats false

echo.
echo ============================================================
echo  [HATA] Streamlit beklenmedik sekilde kapandi!
echo         Yukardaki hata mesajina bak.
echo ============================================================
echo.
pause
