
sudo chmod -R 777 /opt/conda/envs/ptca
conda env list
conda init
source ~/.bashrc
conda activate ptca

pip install --upgrade pip setuptools
cd lerobot
pip install -e .
cd ..
pip install -r requirements.txt
pip list 
sudo apt-get update && sudo apt-get install -y libglib2.0-0 libglib2.0-dev
sudo apt-get install -y libgtk2.0-dev 
pip install --upgrade torch torchvision torchaudio
sudo apt-get install -y fonts-dejavu 
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda install -y ffmpeg=7.0.1 -c conda-forge
sudo apt install -y libavutil-dev 
pip install datasets==4.0.0
pip install torchcodec
sudo apt install -y libvulkan1
sudo apt install -y xvfb
sudo apt install -y screen