# 8 puzzle

생성 일시: 2025년 3월 23일 오후 9:08

### 1. **퍼즐의 개요**

- **퍼즐 종류**: 2x5 형태의 슬라이딩 퍼즐 (주로 "타일 퍼즐" 또는 "슬라이딩 타일 퍼즐"이라고 불림).
- **목표**: 주어진 퍼즐의 시작 상태에서 빈 칸(0)을 이용해 타일들을 옮겨, 최종 목표 상태인 `[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]` 형태로 만들기.
- **퍼즐 크기**: 2행 5열의 10개 타일이 있으며, 타일 번호는 1에서 9까지 있고 0은 빈 칸을 나타냄.

### 2. **퍼즐의 규칙**

- 타일은 빈 칸(0)과 인접한 곳으로만 이동할 수 있음. 즉, 빈 칸이 있는 곳에 인접한 타일을 상, 하, 좌, 우로 움직일 수 있음.
- 목표는 타일들을 이동시켜 목표 상태인 `[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]`을 얻는 것.

### 3. **문제 해결을 위한 알고리즘**

- **문제의 본질**: 주어진 퍼즐에서 타일을 빈 칸을 이용해 이동시키는 문제로, 상태 공간 탐색 문제에 해당함.
- **탐색 방법**: 이 문제는 상태 공간을 탐색하고, 최단 경로로 목표 상태를 찾아야 하므로, 상태 탐색 알고리즘을 사용해야 함.

### 4. **상태 공간 탐색 알고리즘**

퍼즐을 풀기 위한 두 가지 주요 알고리즘을 사용:

- **BFS (Breadth-First Search)**: 너비 우선 탐색 알고리즘. 모든 가능한 상태를 하나씩 탐색하여 최단 경로를 찾아냄.
- *A 알고리즘*: 휴리스틱을 사용한 탐색 알고리즘. BFS보다 더 효율적으로 목표에 도달할 수 있음. 맨해튼 거리(Manhattan Distance)를 휴리스틱으로 사용하여, 목표 상태에 가까운 상태를 우선적으로 탐색함.

### 5. **구체적인 해결 방법**

1. **초기 상태와 목표 상태 정의**
    - 목표 상태는 `[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]`임.
    - 초기 상태는 0부터 9까지의 숫자들을 랜덤하게 섞은 값으로 시작.
2. **해결 가능한 상태인지 확인**
    - 퍼즐은 **해결 가능성**을 체크해야 함. 해결 가능성을 체크하기 위해서는 "역전수(inversion)"라는 개념을 사용.
    - **역전수**: 리스트에서 작은 숫자가 큰 숫자 뒤에 오는 경우를 셈. 역전수가 짝수일 경우만 퍼즐이 해결 가능.
3. **BFS 알고리즘**
    - BFS는 시작 상태에서부터 목표 상태로 가는 모든 경로를 탐색하여, 목표 상태에 도달하는 **최단 경로**를 찾는다.
    - 큐(Queue)를 사용하여 **너비 우선 탐색**을 구현하며, 각 상태를 방문하고 큐에 추가하여 진행한다.
    - 목표 상태에 도달하면, 큐에서 꺼낸 **깊이**(즉, 이동 횟수)를 반환한다.
4. *A 알고리즘*
    - A* 알고리즘은 **휴리스틱**을 사용하여 목표에 더 가까운 상태를 우선적으로 탐색한다.
    - *맨해튼 거리(Manhattan Distance)**를 휴리스틱 함수로 사용하여, 각 상태에서 목표 상태까지의 예상 거리를 계산하고, 이를 기반으로 탐색 순서를 결정한다.
    - *우선순위 큐(Priority Queue)**를 사용하여 상태를 관리하며, `f(n) = g(n) + h(n)` 방식으로 경로 비용을 계산하여 최단 경로를 탐색한다.

### 6. **시간 복잡도 및 성능**

- **BFS**: 각 상태를 탐색하는데 시간이 걸리고, 탐색된 상태들을 큐에 추가하는 방식으로 동작하므로 상태 공간이 매우 크면 시간 소요가 커짐.
- **A**: A* 알고리즘은 휴리스틱을 사용해 보다 효율적으로 탐색하지만, 여전히 상태 공간이 클 경우 시간 소요가 많을 수 있음. 하지만 휴리스틱을 잘 설계하면 BFS보다 빠를 수 있음.

### 7. **코드 설명**

- **BFS**: `queue`에 상태를 넣고, 현재 상태에서 가능한 모든 방향으로 타일을 이동시키며 탐색. 각 상태를 `visited` 집합에 추가하여 중복 방문을 방지.
- **A**: `priority queue`에 `f(n)` 값이 작은 상태부터 탐색하며, 목표 상태에 가까운 상태를 우선적으로 탐색.
- `state_to_string`: 상태를 문자열로 변환하여 집합에 저장하고, 방문한 상태를 추적.
- `manhattan_distance`: 각 타일에서 목표 위치까지의 거리를 계산하여 A* 알고리즘에서 사용할 휴리스틱 값을 계산.

### 8. **실행 시간 비교**

- **BFS**와 **A**알고리즘의 실행 시간 비교:
    - `time.time()`을 사용하여 각 알고리즘의 실행 시간을 측정.
    - 예시 출력에서는 각 알고리즘이 퍼즐을 풀 때 걸린 시간을 초 단위로 출력하며, 두 알고리즘의 성능 차이를 비교 가능.

### 9. **결론**

- 이 퍼즐 문제는 상태 공간 탐색을 통해 해결할 수 있으며, BFS는 모든 경로를 탐색하므로 최단 경로를 보장하지만, 탐색 속도가 느릴 수 있습니다. A* 알고리즘은 휴리스틱을 이용하여 더 효율적으로 탐색할 수 있으므로, 퍼즐 크기가 커질수록 A* 알고리즘이 더 유리할 수 있습니다.

```python
import random
import time
import heapq
from collections import deque

# 목표 상태
goal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

# 가능한 방향 (상, 하, 좌, 우)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (row, col) 변화

# 퍼즐의 크기
rows, cols = 2, 5

# 퍼즐 상태를 문자열로 변환 (이용할 때마다 리스트로 변환)
def state_to_string(state):
    return ''.join(map(str, state))

# 맨해튼 거리 계산 함수
def manhattan_distance(state):
    distance = 0
    for i in range(len(state)):
        if state[i] != 0:  # 빈 칸 제외
            target_row, target_col = (state[i] - 1) // cols, (state[i] - 1) % cols
            current_row, current_col = i // cols, i % cols
            distance += abs(target_row - current_row) + abs(target_col - current_col)
    return distance

# 퍼즐을 풀기 위한 BFS 함수
def bfs(start):
    queue = deque([(start, start.index(0), 0)])  # 시작 상태, 빈 칸의 위치, 이동 횟수
    visited = set()
    visited.add(state_to_string(start))
    
    while queue:
        state, zero_pos, depth = queue.popleft()
        
        # 목표 상태에 도달하면 깊이를 반환
        if state == goal:
            return depth
        
        # 빈 칸의 행, 열 구하기
        row, col = zero_pos // cols, zero_pos % cols
        
        # 가능한 4방향으로 이동
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                new_zero_pos = new_row * cols + new_col
                new_state = state[:]
                # 빈 칸과 교환
                new_state[zero_pos], new_state[new_zero_pos] = new_state[new_zero_pos], new_state[zero_pos]
                
                # 새로운 상태가 방문한 적 없는 상태라면 큐에 넣기
                new_state_str = state_to_string(new_state)
                if new_state_str not in visited:
                    visited.add(new_state_str)
                    queue.append((new_state, new_zero_pos, depth + 1))
    
    return -1  # 해결할 수 없는 경우

# 퍼즐을 풀기 위한 A* 함수
def a_star(start):
    # 우선순위 큐, 큐에는 (f(n), 상태, 빈 칸의 위치, g(n))
    queue = []
    heapq.heappush(queue, (manhattan_distance(start), start, start.index(0), 0))
    visited = set()
    visited.add(state_to_string(start))
    
    while queue:
        f, state, zero_pos, g = heapq.heappop(queue)

        # 목표 상태에 도달하면 깊이를 반환
        if state == goal:
            return g
        
        # 빈 칸의 행, 열 구하기
        row, col = zero_pos // cols, zero_pos % cols
        
        # 가능한 4방향으로 이동
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                new_zero_pos = new_row * cols + new_col
                new_state = state[:]
                # 빈 칸과 교환
                new_state[zero_pos], new_state[new_zero_pos] = new_state[new_zero_pos], new_state[zero_pos]
                
                # 새로운 상태가 방문한 적 없는 상태라면 큐에 넣기
                new_state_str = state_to_string(new_state)
                if new_state_str not in visited:
                    visited.add(new_state_str)
                    f_new = g + 1 + manhattan_distance(new_state)  # f(n) = g(n) + h(n)
                    heapq.heappush(queue, (f_new, new_state, new_zero_pos, g + 1))
    
    return -1  # 해결할 수 없는 경우

# 초기 상태 설정: 0부터 9까지의 숫자
initial_state = list(range(10))

# 상태를 무작위로 섞음
random.shuffle(initial_state)

# 섞인 상태가 해결 가능한 상태인지 확인하는 함수 (풀이가 가능한 상태로 섞였는지 확인)
def is_solvable(state):
    inv_count = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i] > state[j] and state[i] != 0 and state[j] != 0:
                inv_count += 1
    return inv_count % 2 == 0  # 역전수가 짝수일 경우만 해결 가능

# 랜덤하게 섞은 상태가 해결 가능한 상태인지 확인
while not is_solvable(initial_state):
    random.shuffle(initial_state)

print(f"랜덤 초기 상태: {initial_state}")

# BFS 실행 시간 측정
start_time = time.time()
bfs_moves = bfs(initial_state)
bfs_time = time.time() - start_time
print(f"[BFS] 퍼즐을 푸는데 필요한 최소 이동 횟수: {bfs_moves}")
print(f"[BFS] 실행 시간: {bfs_time:.6f} 초")

# A* 실행 시간 측정
start_time = time.time()
a_star_moves = a_star(initial_state)
a_star_time = time.time() - start_time
print(f"[A*] 퍼즐을 푸는데 필요한 최소 이동 횟수: {a_star_moves}")
print(f"[A*] 실행 시간: {a_star_time:.6f} 초")

```