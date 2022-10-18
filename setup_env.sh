cd &&\
git clone https://github.com/KaiyangZhou/deep-person-reid.git &&\
# create environment
cd deep-person-reid/ &&\
conda create --name reid python=3.7 -y &&\
conda activate reid &&\

# install dependencies
# make sure `which python` and `which pip` point to the correct path
pip install -r requirements.txt  &&\

# install torch and torchvision (select the proper cuda version to suit your machine)
conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch &&\

# install torchreid (don't need to re-build it if you modify the source code)
python setup.py install &&\
cd &&\
rm -rf deep-person-reid/
