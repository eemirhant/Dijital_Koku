@echo off
cd /d "%~dp0"
title DijitalKoku V3 -- Kapatiliyor...

echo ============================================================
echo   DijitalKoku V3  Sistem Kapatma
echo ============================================================
echo.

:: [1/3] DijitalKoku Python/Streamlit surecleri
echo [1/3] DijitalKoku surecleri kapatiliyor...
wmic process where "commandline like '%%dashboard.py%%'"     delete >nul 2>&1
wmic process where "commandline like '%%streamlit%%'"        delete >nul 2>&1
wmic process where "commandline like '%%video_worker.py%%'"  delete >nul 2>&1
wmic process where "commandline like '%%window_worker.py%%'" delete >nul 2>&1
wmic process where "commandline like '%%pdm_engine.py%%'"    delete >nul 2>&1
wmic process where "commandline like '%%analyzer.py%%'"      delete >nul 2>&1
wmic process where "commandline like '%%context_analyzer%%'" delete >nul 2>&1
wmic process where "commandline like '%%presentation.py%%'"  delete >nul 2>&1
taskkill /f /im streamlit.exe >nul 2>&1
echo      Surec temizligi tamam.

:: [2/3] Port 8501 / 8502 temizle
echo.
echo [2/3] Portlar temizleniyor (8501 dashboard / 8502 sunum modu)...
for /f "tokens=5" %%p in ('netstat -aon 2^>nul ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p /T >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -aon 2^>nul ^| findstr ":8502" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p /T >nul 2>&1
)
echo      Port temizligi tamam.

:: [3/3] Ollama VLM Servisi (opsiyonel)
echo.
echo [3/3] Ollama VLM servisi kapatilsin mi?
echo       E = Kapat  /  H = Calissin kalmaya devam etsin
echo       (8 saniye icinde cevap verilmezse H secilir)
choice /c EH /n /t 8 /d H /m "      Seciminiz [E/H]: "
if errorlevel 2 (
    echo      Ollama calismaya devam edecek.
) else (
    echo      Ollama kapatiliyor...
    taskkill /f /im ollama.exe >nul 2>&1
    if errorlevel 1 (
        echo      Ollama zaten kapali veya bulunamadi.
    ) else (
        echo      Ollama kapatildi. OK
    )
)

:: Temizlik
if exist ".dk_pid" del /f /q ".dk_pid"

echo.
echo ============================================================
echo   DijitalKoku V3 basariyla kapatildi.
echo ============================================================
timeout /t 3 /nobreak >nul
exit /b 0
