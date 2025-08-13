# cp -r /mnt/dihu/ffmpeg/ ./
# cd ffmpeg
# ./configure --prefix=$HOME/ffmpeg --enable-shared --disable-static
# make -j$(nproc)   # Use all CPU cores for faster compilation
# make install

# git clone https://git.ffmpeg.org/ffmpeg.git
# cd ffmpeg
# ./configure --prefix=$HOME/ffmpeg --enable-shared --disable-static
# make -j$(nproc)   # Use all CPU cores for faster compilation
# make install

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
apt-get install -y libgtk2.0-dev 
pip install --upgrade torch torchvision torchaudio
sudo apt-get install fonts-dejavu
conda install -y ffmpeg=7.0.1 -c conda-forge
sudo apt install -y libavutil-dev 
pip install datasets==4.0.0
# export PATH="$HOME/ffmpeg/bin:$PATH"
