# HyperBCS: Biscale Convolutional Self-Attention Network for Hyperspectral Coastal Wetlands Classification
<b> Official code repository of HyperBCS by [Junshen Luo](https://github.com/JeasunLok), Zhi He, Haomei Lin, Heqian Wu. </b>
***
![](images/3D-HyperBCS.jpg)

***
## How to use it?
### 1. Installation
```
git clone https://github.com/JeasunLok/HyperBCS.git && cd HyperBCS
conda create -n HyperBCS python=3.7
conda activate HyperBCS
pip install -r requirements.txt
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html 
```

### 2. Download our datasets

Download our datasets then place them in `data` folder

Baiduyun: https://pan.baidu.com/s/1hyye2fVxoUaOJ6YR_RUSJg 
(access code: js66)

Google Drive: https://drive.google.com/drive/folders/1jjg6Jlyb92pVrUzbdr5fHSMzYQnr2U47

### 3. Quick start to use our SOTA model HyberBCS

<b> Dataset MongCai </b>
```
python main_argparse.py -mt HyperBCS -hbcsm 3D -e 100 -lr 5e-3 -bs 32 -d MongCai
```
<b> Dataset CamPha </b>
```
python main_argparse.py -mt HyperBCS -hbcsm 3D -e 100 -lr 5e-3 -bs 32 -d CamPha
```

### 4. More detailed information
```
python main_argparse.py -h
```
***