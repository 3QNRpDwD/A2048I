
import numpy as np

# 휴리스틱 가중치
WEIGHT_MONOTONICITY = 1.0
WEIGHT_SMOOTHNESS = 0.1
WEIGHT_EMPTY_TILES = 2.7
WEIGHT_MAX_VALUE = 1.0

def evaluate(board):
    """주어진 보드 상태를 평가하여 점수를 반환합니다."""
    mono_score = monotonicity_score(board)
    smooth_score = smoothness_score(board)
    empty_tiles = len(np.argwhere(board == 0))
    max_value = np.max(board)

    return (
        mono_score * WEIGHT_MONOTONICITY +
        smooth_score * WEIGHT_SMOOTHNESS +
        np.log(empty_tiles) * WEIGHT_EMPTY_TILES if empty_tiles > 0 else 0 +
        max_value * WEIGHT_MAX_VALUE
    )

def monotonicity_score(board):
    """단조성 점수: 타일이 한 방향으로 정렬될수록 높은 점수를 받습니다."""
    scores = [0, 0, 0, 0]
    # 상/하, 좌/우 방향으로 단조성 체크
    for i in range(4):
        # 좌 -> 우
        current = 0
        next = current + 1
        while next < 4:
            while next < 4 and board[i, next] == 0:
                next += 1
            if next >= 4: next -= 1
            current_val = np.log2(board[i, current]) if board[i, current] > 0 else 0
            next_val = np.log2(board[i, next]) if board[i, next] > 0 else 0
            if current_val > next_val:
                scores[0] += next_val - current_val
            elif next_val > current_val:
                scores[1] += current_val - next_val
            current = next
            next += 1
        # 상 -> 하
        current = 0
        next = current + 1
        while next < 4:
            while next < 4 and board[next, i] == 0:
                next += 1
            if next >= 4: next -= 1
            current_val = np.log2(board[current, i]) if board[current, i] > 0 else 0
            next_val = np.log2(board[next, i]) if board[next, i] > 0 else 0
            if current_val > next_val:
                scores[2] += next_val - current_val
            elif next_val > current_val:
                scores[3] += current_val - next_val
            current = next
            next += 1
            
    return max(scores[0], scores[1]) + max(scores[2], scores[3])

def smoothness_score(board):
    """매끄러움 점수: 인접한 타일의 값 차이가 적을수록 높은 점수를 받습니다."""
    score = 0
    for i in range(4):
        for j in range(4):
            if board[i, j] != 0:
                val = np.log2(board[i, j])
                # 오른쪽 인접 타일
                if j + 1 < 4 and board[i, j+1] != 0:
                    neighbor_val = np.log2(board[i, j+1])
                    score -= abs(val - neighbor_val)
                # 아래쪽 인접 타일
                if i + 1 < 4 and board[i+1, j] != 0:
                    neighbor_val = np.log2(board[i+1, j])
                    score -= abs(val - neighbor_val)
    return score
