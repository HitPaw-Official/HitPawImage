1. Environment setup  
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://mirrors.aliyun.com/pypi/simple  
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple  

2. Run the matting model
python matting_with_depthAnything_sodAything.py