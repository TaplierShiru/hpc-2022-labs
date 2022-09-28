dnf makecache --refresh
dnf -y install python38-devel
pip3 install numpy
pip3 install Cython
pip3 install pandas
pip3 install tqdm
pip3 install matplotlib
pip3 install seaborn
pip3 install jupyter
cd code && make build
jupyter-notebook --ip=0.0.0.0 --port=50000 --allow-root --no-browser