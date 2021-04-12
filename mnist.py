import sys, os
import numpy as np
sys.path.append(os.pardir)   # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from get_data import load_mnist

# 이 형식으로 반환하겠다 뭐 그런뜻
# load_mnist를 통해 피클 파일 생성해줌
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)  # 인수로 normalize, flatten, one_hot_label 설정

# 각 데이터 형상 출력
print(x_train.shape)
