import matplotlib.pyplot as plt

# 초기값 및 설정
x = 10
learning_rate = 0.01
precision = 0.00001
max_iterations = 100

# 손실 함수와 그래디언트 정의
loss_func = lambda x: (x-3) ** 2 + 10
gradient = lambda x: 2 * x - 6

# 손실 함수 값을 저장할 리스트
loss_values = []

# 경사하강법
for i in range(max_iterations):
    x = x - learning_rate * gradient(x)  # x 업데이트
    current_loss = loss_func(x)         # 현재 손실 함수 값 계산
    loss_values.append(current_loss)    # 리스트에 추가
    print(f"Iteration {i+1}, x = {x:.5f}, Loss = {current_loss:.5f}")

print('최소값 x = ', x)

# 손실 함수 그래프 그리기
plt.plot(range(1, max_iterations + 1), loss_values, linestyle='-', color='r')
plt.title('Loss Function Value Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss Function Value')
plt.grid()
# x축과 y축의 비율을 동일하게 설정
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
