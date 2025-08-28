git clone https://github.com/TencentARC/Moto.git

git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
pip install numpy==1.24.4
cd SimplerEnv/ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda install -y ffmpeg=7.0.1 -c conda-forge
sudo apt install -y libavutil-dev 
sudo apt install -y libvulkan1
sudo apt install -y xvfb
pip install setuptools==58.2.0
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1
pip install git+https://github.com/nathanrooy/simulated-annealing
cd ..
cd Moto
pip install -r requirements.txt
pip install mediapy
