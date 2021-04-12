import sys, os
import numpy as np
sys.path.append(os.pardir)   # 부모 디렉터리의 파일을 가져올 수 있도록 설정


# sum of square for error, 오차제곱합
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# cross entropy error, 교차 엔트로피
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
