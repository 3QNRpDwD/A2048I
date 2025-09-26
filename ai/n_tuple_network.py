import numpy as np
import os
import random

# N-Tuple 네트워크에서 사용할 튜플들 정의
# 각 튜플은 보드 상의 4개 위치 (r, c)의 조합입니다.
TUPLES = [
    # 가로 튜플
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    [(2, 0), (2, 1), (2, 2), (2, 3)],
    [(3, 0), (3, 1), (3, 2), (3, 3)],
    # 세로 튜플
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 1), (1, 1), (2, 1), (3, 1)],
    [(0, 2), (1, 2), (2, 2), (3, 2)],
    [(0, 3), (1, 3), (2, 3), (3, 3)],
]

class NTupleNetwork:
    def __init__(self, num_tuples=len(TUPLES)):
        self.num_tuples = num_tuples
        # 각 타일 값은 2^0 ~ 2^15 (0 ~ 15) 범위로 매핑될 수 있다고 가정합니다.
        # 4개 타일의 조합이므로 16^4 크기의 조회 테이블(LUT)이 필요합니다.
        self.lut_size = 16 ** 4
        self.luts = [np.random.uniform(-0.1, 0.1, self.lut_size) for _ in range(self.num_tuples)]

    def get_tuple_value(self, board, tuple_indices):
        """보드와 튜플 인덱스로부터 LUT 인덱스를 계산합니다."""
        val = 0
        for r, c in tuple_indices:
            tile_val = board[r, c]
            power = int(np.log2(tile_val)) if tile_val > 0 else 0
            val = val * 16 + power
        return val

    def evaluate(self, board):
        """N-Tuple 네트워크를 사용해 보드 상태를 평가합니다."""
        total_score = 0
        for i, t in enumerate(TUPLES):
            lut_index = self.get_tuple_value(board, t)
            total_score += self.luts[i][lut_index]
        return total_score

    def update(self, board, error, learning_rate):
        """오차와 학습률을 기반으로 네트워크의 가중치(LUT)를 업데이트합니다."""
        for i, t in enumerate(TUPLES):
            lut_index = self.get_tuple_value(board, t)
            self.luts[i][lut_index] += learning_rate * error

    def save_weights(self, path):
        """학습된 가중치(LUT)를 파일에 저장합니다."""
        try:
            np.save(path, self.luts)
            print(f"가중치를 '{path}'에 저장했습니다.")
        except Exception as e:
            print(f"가중치 저장 중 오류 발생: {e}")

    def load_weights(self, path):
        """파일에서 가중치(LUT)를 불러옵니다."""
        if os.path.exists(path):
            try:
                self.luts = np.load(path, allow_pickle=True)
                print(f"'{path}'에서 가중치를 불러왔습니다.")
            except Exception as e:
                print(f"가중치 로딩 중 오류 발생: {e}")
        else:
            print(f"가중치 파일 '{path}'를 찾을 수 없습니다. 무작위 초기 가중치를 사용합니다.")
