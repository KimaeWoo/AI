#인공지능 퍼셉트론 프로그램

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [1,0], [0,1], [1,1]]) # 입력 데이터
Y = np.array([-1,1,1,1]) # 레이블

# 데이터 시각화
plt.scatter(X[0][0], X[0][1], c='red') # x,y좌표에 빨간색 점 그리
plt.scatter(X[1][0], X[1][1], c='blue')
plt.scatter(X[2][0], X[2][1], c='blue')
plt.scatter(X[3][0], X[3][1], c='blue')
plt.show()

# 초기 가중치 (bias, w1, w2)
w = np.array([1., 1., 1.]) # [bias, w1, w2]

# 퍼셉트론 예측을 위한 forward 함수
def forward(x):
    # np.dot : 두 배열 간의 점곱(내적, dot product)을 계산
    # 선형 모델의 핵심 계산인 **가중치(weight)**와 **특징(feature)**의 곱을 효과적으로 계산하기 위해
    # 내적은 머신러닝에서 선형 회귀(linear regression), 퍼셉트론(perceptron), 그리고 신경망의 기본 계산 방식
    return np.dot(x, w[1:]) + w[0]

# 예측 함수
def predict(X):
    # 조건에 따라 배열을 생성하는 함수
    # 퍼셉트론의 예측값을 계산
    # np.where(조건, 참일 때 값, 거짓일 때 값)
    return np.where(forward(X) > 0, 1, -1)
    
print('predict (before traning)', w)

for epoch in range(50):
    # 퍼셉트론 학습 알고리즘의 핵심 부분
    # 주어진 입력 데이터 𝑋와 해당 레이블 𝑌에 대해 **가중치(weight)**를 업데이트하는 과정
    # 이 과정은 오차를 기반으로 가중치를 점진적으로 조정하여 모델의 예측 성능을 향상시킵니다.
    for x_val, y_val in zip(X, Y):
        update = 0.01 * (y_val - predict(x_val)) # 오차 계산
        w[1:] += update * x_val # 가중치 업데이트
        w[0] += update # bias 업데이트

print('predict (after traning)', w)