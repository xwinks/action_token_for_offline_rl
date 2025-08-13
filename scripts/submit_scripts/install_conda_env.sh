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

echo $PATH
conda env list
conda create -n offline_rl_for_vla python=3.10 -y
conda env list
conda init
# conda activate offline_rl_for_vla
source ~/.bashrc
conda activate offline_rl_for_vla
echo try activate offline_rl_for_vla
conda activate /home/aiscuser/.conda/envs/offline_rl_for_vla


export PATH="/home/aiscuser/.conda/envs/offline_rl_for_vla/bin:$PATH"
export CONDA_PREFIX="/home/aiscuser/.conda/envs/offline_rl_for_vla"

echo $PATH

pip install --upgrade pip setuptools
cd lerobot
pip install -e .
cd ..
pip install -r requirements.txt
pip list 

sudo apt-get update && sudo apt-get install -y libglib2.0-0 libglib2.0-dev
# export PATH="$HOME/ffmpeg/bin:$PATH"
