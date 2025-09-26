
import numpy as np

# 표준 Expectimax AI를 위한 기본 가중치
DEFAULT_WEIGHTS = {
    "monotonicity": 1.0,
    "smoothness": 0.1,
    "empty_tiles": 2.7,
    "max_value": 1.0,
}

def evaluate(board, weights):
    """가중치 딕셔너리와 보드 상태를 받아, 평가 점수를 반환합니다."""
    
    mono_score = monotonicity_score(board)
    smooth_score = smoothness_score(board)
    empty_tiles = len(np.argwhere(board == 0))
    max_value = np.max(board)

    return (
        mono_score * weights.get("monotonicity", 0) +
        smooth_score * weights.get("smoothness", 0) +
        (np.log(empty_tiles) * weights.get("empty_tiles", 0) if empty_tiles > 0 else 0) +
        max_value * weights.get("max_value", 0)
    )

def monotonicity_score(board):
    """단조성 점수: 타일이 한 방향으로 정렬될수록 높은 점수를 받습니다."""
    scores = [0, 0, 0, 0]
    # 상/하, 좌/우 방향으로 단조성 체크
    for i in range(4):
        # 좌 -> 우
        current = 0
        next_ = current + 1
        while next_ < 4:
            while next_ < 4 and board[i, next_] == 0:
                next_ += 1
            if next_ >= 4: next_ -= 1
            current_val = np.log2(board[i, current]) if board[i, current] > 0 else 0
            next_val = np.log2(board[i, next_]) if board[i, next_] > 0 else 0
            if current_val > next_val:
                scores[0] += next_val - current_val
            elif next_val > current_val:
                scores[1] += current_val - next_val
            current = next_
            next_ += 1
        # 상 -> 하
        current = 0
        next_ = current + 1
        while next_ < 4:
            while next_ < 4 and board[next_, i] == 0:
                next_ += 1
            if next_ >= 4: next_ -= 1
            current_val = np.log2(board[current, i]) if board[current, i] > 0 else 0
            next_val = np.log2(board[next_, i]) if board[next_, i] > 0 else 0
            if current_val > next_val:
                scores[2] += next_val - current_val
            elif next_val > current_val:
                scores[3] += current_val - next_val
            current = next_
            next_ += 1
            
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
