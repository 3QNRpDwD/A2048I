import json
import os
from .expectimax_ai import ExpectimaxAI

# 유전 알고리즘으로 생성된 가중치를 저장하는 파일 경로
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "evolved_weights.json")

def load_evolved_weights():
    """
    JSON 파일에서 진화된 가중치를 로드합니다.
    파일이 없으면 하드코딩된 기본값을 반환합니다.
    """
    if os.path.exists(WEIGHTS_FILE):
        print(f"'{WEIGHTS_FILE}'에서 진화된 가중치를 로드합니다.")
        with open(WEIGHTS_FILE, "r") as f:
            return json.load(f)
    else:
        print("진화된 가중치 파일을 찾을 수 없습니다. 기본 하드코딩 가중치를 사용합니다.")
        # 파일이 없을 경우의 기본값 (기존의 EVOLVED_WEIGHTS)
        return {
            "monotonicity": 4.5,
            "smoothness": 2.0,
            "empty_tiles": 2.5,
            "max_value": 0.5,
        }

# 프로그램 시작 시 가중치를 로드합니다.
EVOLVED_WEIGHTS = load_evolved_weights()

class EvoExpectimaxAI(ExpectimaxAI):
    """
    '진화된' 휴리스틱 가중치 세트를 사용하는 Expectimax AI입니다.
    가중치는 train_ga.py를 통해 생성된 `evolved_weights.json`에서 로드되거나,
    파일이 없을 경우 하드코딩된 값을 사용합니다.
    """
    def __init__(self, depth=3):
        super().__init__(depth)
        # 기본 가중치를 진화된 가중치 세트로 오버라이드합니다.
        self.heuristic_weights = EVOLVED_WEIGHTS
