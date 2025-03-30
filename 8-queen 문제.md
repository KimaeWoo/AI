## 1. 서론

8-Queen 문제는 8개의 퀸을 체스판 위에 배치하되, 서로 공격하지 않도록 배치하는 조합 최적화 문제이다. 이 문제를 해결하기 위해 유전자 알고리즘(Genetic Algorithm, GA)을 적용할 수 있으며, 염색체를 특정 형식으로 정의하고, 교차연산(Crossover) 및 돌연변이(Mutation) 연산을 통해 최적해를 탐색할 수 있다.

## 2. 유전자 알고리즘 개요

유전자 알고리즘은 자연선택과 유전적 변이를 모방한 탐색 알고리즘으로, 다음과 같은 절차를 따른다.

1. 초기 개체군 생성
2. 적합도 평가
3. 부모 선택 (룰렛 휠 알고리즘 적용)
4. 교차연산 (Crossover)
5. 돌연변이 (Mutation)
6. 새로운 개체군 형성
7. 종료 조건 만족 시 알고리즘 종료

## 3. 염색체 표현

각 퀸은 서로 다른 열에 위치해야 하므로, 염색체는 8개의 정수로 구성되며, 각 정수는 해당 열에서의 행 번호를 의미한다. 각 열에는 반드시 한 개의 퀸만 존재해야 하므로, 염색체는 1부터 8까지의 숫자를 포함하는 순열이어야 한다. 예를 들어 [4, 2, 7, 3, 6, 8, 5, 1]은 첫 번째 열에 4행, 두 번째 열에 2행, 세 번째 열에 7행에 퀸을 배치한 상태를 나타낸다.

## 4. 적합도 함수

적합도는 공격받지 않는 퀸 쌍의 개수를 사용하여 평가한다. 총 8개의 퀸이 존재하므로 최대한의 적합도 값은 28이 된다. 적합도 함수는 다음과 같이 정의된다:

여기서 h는 서로 공격하는 퀸 쌍의 개수이다.

## 5. 부모 선택 (룰렛 휠 선택)

룰렛 휠 선택 방법은 적합도가 높은 염색체가 선택될 확률을 증가시키는 방법이다. 각 개체의 적합도를 기반으로 전체 적합도 합에서 해당 개체가 차지하는 비율에 따라 부모를 선택한다.

## 6. 교차연산 (Crossover)

두 개의 부모를 선택한 후, 1부터 8 사이의 난수를 생성하여 교차점을 결정한다. 교차 연산 후에도 중복이 없는 순열을 유지해야 하므로, 일부 값은 재조정된다.

## 7. 돌연변이 (Mutation)

돌연변이는 개체의 일부를 변경하여 탐색 공간을 확장하는 역할을 한다. 확률적으로 한 개체의 두 위치를 교환하는 방식으로 변이를 수행하여 순열 특성을 유지한다.

## 8. 문제 해결 과정

1. **초기 개체군 생성**: 1부터 8까지의 숫자를 랜덤하게 섞은 배열을 여러 개 생성한다.
2. **적합도 평가**: 각 개체의 적합도를 계산하여 공격받지 않는 퀸 쌍의 개수를 측정한다.
3. **부모 선택**: 룰렛 휠 선택 방식으로 적합도가 높은 개체를 우선적으로 선택한다.
4. **교차 연산**: 두 개의 부모를 선택하고 특정 지점에서 유전자를 교환하되, 중복된 값이 없도록 조정한다.
5. **돌연변이 연산**: 일정 확률로 두 값을 교환하여 순열을 유지한다.
6. **새로운 개체군 형성**: 교차 연산과 돌연변이로 생성된 자손을 포함하여 새로운 개체군을 형성하고 다음 세대로 전달한다.
7. **종료 조건 확인**: 적합도가 28인 개체가 생성되거나, 최대 세대 수에 도달하면 알고리즘을 종료한다.

## 9. 문제 해결을 위한 파이썬 코드

```python
import random
import matplotlib.pyplot as plt
import numpy as np

# 적합도 함수 정의
def fitness(chromosome):
    h = 0
    for i in range(len(chromosome)):
        for j in range(i + 1, len(chromosome)):
            if abs(chromosome[i] - chromosome[j]) == abs(i - j):
                h += 1
    return 28 - h

# 부모 선택 (룰렛 휠)
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [f / total_fitness for f in fitness_scores]
    return random.choices(population, probabilities, k=2)

# 교차 연산
def crossover(parent1, parent2):
    point = random.randint(1, 7)
    child = parent1[:point] + [x for x in parent2 if x not in parent1[:point]]
    return child

# 돌연변이 연산
def mutate(chromosome, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(8), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

# 유전자 알고리즘 실행
def genetic_algorithm(pop_size=100, generations=1000):
    population = [random.sample(range(1, 9), 8) for _ in range(pop_size)]
    for generation in range(generations):
        fitness_scores = [fitness(chromosome) for chromosome in population]
        if max(fitness_scores) == 28:
            return population[fitness_scores.index(28)]
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
    return max(population, key=fitness)

# 체스판 시각화
def draw_chessboard(solution):
    board = np.zeros((8, 8))
    for col, row in enumerate(solution):
        board[row - 1, col] = 1
    fig, ax = plt.subplots()
    ax.matshow(board, cmap=plt.cm.binary)
    for i in range(8):
        for j in range(8):
            if board[i, j] == 1:
                ax.text(j, i, 'Q', ha='center', va='center', fontsize=20, color='red')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# 실행
solution = genetic_algorithm()
print("Solution:", solution)
print("Fitness:", fitness(solution))
draw_chessboard(solution)
```

## 실행 결과

Solution: [1, 7, 4, 6, 8, 2, 5, 3]

Fitness: 28

![image](https://github.com/user-attachments/assets/8125029e-c27f-4aa6-b1fb-050aeb74f0fa)
