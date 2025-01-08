# Perceptron

생성 일시: 2025년 1월 8일 오전 11:45

```python
# PERCEPTION 라이브러리 이용

# scikit-learn dsm 파이썬으로 구현한 머신러닝 확장 라이브러리로 다양한 API제공
from sklearn.linear_model import Perceptron  

# 샘플과 레이블 
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 0, 0, 1]

# 퍼셉트론 생성. tol은 종료 조건, random_state는 난수의 시드
clf = Perceptron(tol = 1e-3, random_state = 0)

# 학습을 수행
clf.fit(X,y) # 데이터와 라벨로 학습

# 테스트를 수행. 학습된 모델을 사용하여 테스트 데이터를 분류한 결과를 반환
print(clf.predict(X))

# PERCEPTION 라이브러리 이용 X
# 뉴론의 출력 계산 함수
def calculate(input):
    global weights            # 전체 네트워크에서 가중치를 공유
    global bias               # 전체 네트워크에서 바이어스를 공유
    activation = bias         # 바이어스
    for i in range(2):        # 입력 신호 총합 계산 
        activation += weights[i] * input[i]
    if activation >= 0.0:  # 스텝 활성화 함수
        return 1.0
    else:
        return 0.0

# 학습 알고리즘
def train_weights(X, y, l_rate, n_epoch):
    global weights
    global bias
    for epoch in range(n_epoch):     # 에포크 반복
        sum_error = 0.0
        for row, target in zip(X, y):      # 데이터셋을 반복
            actual = calculate(row)        # 실제 출력 계산
            error = target - actual        # 실제 출력계산
            bias = bias + l_rate * error   
            sum_error += error ** 2        # 오류의 제곱 계산
            for i in range(2):             # 가중치 변경
                weights[i] = weights[i] + l_rate * error * row[i]
            print(weights, bias)
        print('에포크 번호 = %d, 학습률 = %.3f, 오류 = %.3f' % (epoch, l_rate, sum_error))
    return weights

# AND 연산 학습 데이터셋, 샘플과 레이블
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 0, 0, 1]

# 가중치와 바이어스 초기값
weights = [0.0, 0.0]
bias = 0.0
l_rate = 0.1      # 학습률
n_epoch = 5       # 에포크 횟수
weights = train_weights(X, y, l_rate, n_epoch)
print(weights, bias)
print(X,clf.predict(X))
```

```
[0 0 0 1]
[0.0, 0.0] -0.1
[0.0, 0.0] -0.1
[0.0, 0.0] -0.1
[0.1, 0.1] 0.0
에포크 번호 = 0, 학습률 = 0.100, 오류 = 2.000
[0.1, 0.1] -0.1
[0.1, 0.0] -0.2
[0.1, 0.0] -0.2
[0.2, 0.1] -0.1
에포크 번호 = 1, 학습률 = 0.100, 오류 = 3.000
[0.2, 0.1] -0.1
[0.2, 0.0] -0.2
[0.1, 0.0] -0.30000000000000004
[0.2, 0.1] -0.20000000000000004
에포크 번호 = 2, 학습률 = 0.100, 오류 = 3.000
[0.2, 0.1] -0.20000000000000004
[0.2, 0.1] -0.20000000000000004
[0.2, 0.1] -0.20000000000000004
[0.2, 0.1] -0.20000000000000004
에포크 번호 = 3, 학습률 = 0.100, 오류 = 0.000
[0.2, 0.1] -0.20000000000000004
[0.2, 0.1] -0.20000000000000004
[0.2, 0.1] -0.20000000000000004
[0.2, 0.1] -0.20000000000000004
에포크 번호 = 4, 학습률 = 0.100, 오류 = 0.000
[0.2, 0.1] -0.20000000000000004
[[0, 0], [0, 1], [1, 0], [1, 1]] [0 0 0 1]
```
