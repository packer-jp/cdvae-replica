# cdvae

Describe your project here.

```
pyenv local 3.9.18
python -m venv venv
source venv/bin/activate
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip intstall torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric
pip install -r requirements.txt
```