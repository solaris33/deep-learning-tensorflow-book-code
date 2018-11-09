# Inception V3 Retraining을 이용한 나만의 분류기 만들기
Inception V3 retraining을 위해서는 아래 과정을 단계별로 수행해야 합니다.

## 나만의 트레이닝 데이터 준비 및 파인튜닝하기
루트 폴더(e.g. cat_photos)를 생성하고, 그 안에 레이블(e.g. charteux, persian, ragdoll)을 폴더명으로 지정하고, 폴더별로 레이블에 해당되는 이미지 데이터를 최소 30장 이상씩 준비합니다. 그 뒤, 루트 폴더를 원하는 경로(e.g. ~/cat_photos, C:\cat_photos)에 이동시킨뒤 아래 명령어로 retrain 스크립트를 실행합니다. 

[Mac/Linux]
```
python inceptionv3_retrain.py --image_dir=~/cat_photos
```

[Windows]
```
python inceptionv3_retrain.py --image_dir=C:\cat_photos
```

## 파인튜닝된 모델을 이용해서 추론하기
파인튜닝이 끝나면 아래 명령어로 테스트 이미지에 대한 추론을 진행합니다. 추론할 이미지에 대한 디폴트 경로는 /tmp/test_charteux.jpg로 지정되어 있습니다.

```
python inceptionv3_inference.py
```

## Reference
[https://www.tensorflow.org/hub/tutorials/image_retraining](https://www.tensorflow.org/hub/tutorials/image_retraining)
