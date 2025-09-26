import numpy as np
from .base_ai import BaseAI
from .policy_model import PolicyModel

class ReinforceAI(BaseAI):
    """REINFORCE 알고리즘을 사용하는 정책 기반 AI입니다."""
    def __init__(self, learning_rate=0.01):
        self.model = PolicyModel()
        self.learning_rate = learning_rate
        
        # 한 에피소드(게임 한 판) 동안의 기록을 저장합니다.
        self.episode_history = []
        # get_move에서 결정한 행동과 상태를 임시 저장합니다.
        self.last_state_action = None
        self.update_count = 0

    def get_move(self, game):
        """정책 모델의 확률 분포에 따라 행동을 선택합니다."""
        board = game.board
        
        # 1. 정책 모델로부터 행동 확률을 얻습니다.
        action_probs = self.model.get_action_probs(board)
        
        # 2. 확률에 따라 행동을 샘플링합니다. (탐험)
        # np.random.choice는 주어진 확률에 따라 인덱스를 선택합니다.
        move = np.random.choice(len(action_probs), p=action_probs)
        
        # 3. 나중에 학습에 사용하기 위해 상태와 행동을 저장합니다.
        self.last_state_action = {"board": np.copy(board), "action": move}
        
        # 분석 데이터는 각 행동의 확률으로 채웁니다.
        analysis_data = {i: prob for i, prob in enumerate(action_probs)}
        
        return move, analysis_data, None

    def post_move_update(self, reward, game_over):
        """
        수가 실행된 후 호출되어 보상을 기록하고, 게임이 끝나면 학습을 실행합니다.
        Args:
            reward (float): 방금 수행한 행동으로 얻은 보상 (점수).
            game_over (bool): 게임이 종료되었는지 여부.
        """
        if self.last_state_action is None:
            return

        # 1. 임시 저장된 상태/행동과 방금 받은 보상을 묶어 에피소드 기록에 추가합니다.
        board = self.last_state_action["board"]
        action = self.last_state_action["action"]
        self.episode_history.append((board, action, reward))
        
        self.last_state_action = None # 임시 데이터 클리어

        # 2. 게임이 종료되면, 전체 에피소드 기록을 사용해 모델을 업데이트합니다.
        if game_over:
            self.model.update(self.episode_history, self.learning_rate)
            self.episode_history = [] # 다음 에피소드를 위해 기록 초기화
            self.update_count += 1
            print(f"[ReinforceAI] 학습 완료 (업데이트 횟수: {self.update_count})")
