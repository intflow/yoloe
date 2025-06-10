
## run docker container
```
bash docker/run_docker.sh
```

## Dependency
프로젝트 내의 파이썬 패키지들(`ultralytics, CLIP, ml-mobileclip, lvis-api` 등)을 editable 형태로 설치하여 python 라이브러리 형태로 import 해서 쓸 수 있게 하기 위함.
```
pip install -r requirements.txt
```

## Download image-text embedding model
```
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
```

## Tracker Install
```
apt-get update && apt-get install -y build-essential python3-dev && \
pip install torchreid==0.2.5 gdown==4.7.1 cython==3.0.2 yacs==0.1.8 lap && \
export PYTHONPATH=/works/yoloe:$PYTHONPATH
```