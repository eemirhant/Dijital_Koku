@echo off
cd /d "%~dp0"

echo ============================================================
echo  DijitalKoku V3 — Ilk Kurulum
echo ============================================================
echo.

:: Python kontrolu
python --version >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadi. python.org'dan Python 3.10+ kur.
    pause
    exit /b 1
)
echo [+] Python bulundu:
python --version

:: pip guncelle
echo.
echo [+] pip guncelleniyor...
python -m pip install --upgrade pip

:: Temel gereksinimler (torch haric)
echo.
echo [+] Temel paketler kuruluyor...
python -m pip install ^
    streamlit>=1.37.0 ^
    opencv-python>=4.9.0 ^
    numpy>=1.24.0 ^
    Pillow>=10.0.0 ^
    ultralytics>=8.1.0 ^
    scikit-learn>=1.3.0 ^
    pygetwindow>=0.0.9 ^
    mss>=9.0.0 ^
    pandas>=2.0.0 ^
    plotly>=5.18.0 ^
    pyserial>=3.5 ^
    pywin32>=306 ^
    transformers>=4.40.0 ^
    accelerate>=0.27.0 ^
    yt-dlp>=2024.1.0 ^
    requests>=2.31.0

:: CUDA kontrol ve torch kurulumu
echo.
echo [+] NVIDIA GPU kontrol ediliyor...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [!] NVIDIA GPU bulunamadi veya surucu eksik — CPU modunda torch kuruluyor.
    python -m pip install torch>=2.1.0
) else (
    echo [+] NVIDIA GPU tespit edildi — CUDA 12.1 destekli torch kuruluyor.
    echo     Bu islem buyuk dosya indirirken birkac dakika surebilir...
    python -m pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
)

:: Kurulum dogrulama
echo.
echo [+] Kurulum dogrulaniyor...
python -c "import torch; print('  torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import streamlit; print('  streamlit:', streamlit.__version__)"
python -c "import transformers; print('  transformers:', transformers.__version__)"
python -c "import ultralytics; print('  ultralytics: OK')"
python -c "import cv2; print('  opencv:', cv2.__version__)"
python -c "import accelerate; print('  accelerate: OK')"

echo.
echo ============================================================
echo  Kurulum tamamlandi! Basmak icin: start.bat
echo.
echo  NOT (V3): VLM katmani icin Ollama kurulmali ve
echo  moondream modeli cekilmeli. start.bat bunu otomatik
echo  yapar. Manuel kurmak icin:
echo    1. https://ollama.com/download adresinden Ollama kur
echo    2. ollama pull moondream
echo ============================================================
pause
