# nerf-pytorch
- 기본적인 reference 인듯

# Dataloader
- nerf-pytorch와는 `./dataloader/load_llff.py` 디렉토리 안에 있냐 없냐 차이.
- DoF-NeRF load_llff.py 320 line에 저자에 흔적이 남아있음.

# nerf_utils.py
- nerf-pytorch run_nerf.py(학습 코드)의 config 이전까지, 즉 학습 부분 전까지는 nerf_utils.py와 거의 동일
- N_image라는 파라미터가 DoF-NeRF에는 존재.

# bokeh_utils.py
- bokeh_renderer 내에 있는 scatter.py 참조
- scatter.py는 cupy로 작성되어있음

# options.py
- nerf-pytorch run_nerf.py config 부분과 비슷
- default 값이 생기고, bokeh 관련 config 생기고, indent가 좀 다름.


# To do
- 논문 참고하면서 새로운 train.py 같은 파일 만들어서 run_nerf.py의 train 함수와 같은 것 구현.

# condigs
- Deblur-NeRF 스타일 참고

# load llff
- LensNeRF dataset load를 위해 수정