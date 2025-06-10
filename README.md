
## run docker container
```
bash docker/run_docker.sh
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