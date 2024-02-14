# CA-PatchCore on MVTec LOCO AD

**This repository is implementation of Cross-attention PatchCore (CA-PatchCore) on MVTec LOCO AD.**

## Installation
Donwload MVTec LOCO AD to **./dataset**.
(https://www.mvtec.com/company/research/datasets/mvtec-loco)

### environment
~~~
python==3.10.12
torch==2.0.1
torchvision==0.15.2
~~~

Install packages with:
~~~
pip install -r requirements.txt
~~~

Install PyDenceCRF with:
~~~
pip install cython
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
~~~
(Reference:https://github.com/lucasb-eyer/pydensecrf)

## Usage
Run with:
~~~
python main.py 
~~~

**Click [here](https://github.com/IshidaKengo/CA-PatchCore) for implementation in Co-occurrence Anomaly Detection Dataset (CAD-SD).**
