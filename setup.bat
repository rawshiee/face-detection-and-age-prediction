:: -------------------------------
:: Age-Gender Detection Setup Script
:: -------------------------------

:: 1. Create and activate virtual environment
python -m venv venv

:: 2. Activate the environment
call venv\Scripts\activate

:: 3. Upgrade pip (always a good idea)
python -m pip install --upgrade pip

:: 4. Install core dependencies
pip install numpy==1.26.4
pip install opencv-python==4.10.0.84
pip install opencv-contrib-python==4.10.0.84
pip install imutils==0.5.4
pip install tqdm==4.66.5
pip install onnxruntime==1.19.2

:: 5. Optional (useful for debugging, camera testing, etc.)
pip install matplotlib
pip install pillow

:: 6. Confirm installation
python -c "import cv2, numpy; print('OpenCV:', cv2.__version__, '| NumPy:', numpy.__version__)"
