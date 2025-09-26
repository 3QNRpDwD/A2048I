from .base_ai import BaseAI
from .n_tuple_network import NTupleNetwork
import numpy as np

class NTupleAI(BaseAI):
    def __init__(self, weights_path="n_tuple_weights.npy", learning_rate=0.0025, save_interval=1000):
        """온라인 학습이 가능한 N-Tuple 네트워크 AI를 초기화합니다."""
        self.weights_path = weights_path
        self.learning_rate = learning_rate
        self.save_interval = save_interval
        self.update_count = 0
        
        self.network = NTupleNetwork()
        self.network.load_weights(self.weights_path)
        
        # 학습에 필요한 이전 상태 정보를 저장합니다.
        self.last_state_for_learning = None

    def get_move(self, game):
        """가능한 각 수의 결과를 평가하여 최상의 수를 찾고, 현재 상태를 학습용으로 저장합니다."""
        current_board_for_eval = np.copy(game.board)
        current_value = self.network.evaluate(current_board_for_eval)
        
        best_move, best_score = -1, -np.inf
        analysis_data = {}

        for move in range(4):
            temp_game = game.clone()
            board_changed, _ = temp_game.move(move)

            if not board_changed:
                analysis_data[move] = -np.inf
                continue

            score = self.network.evaluate(temp_game.board)
            analysis_data[move] = score

            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move == -1:
            self.last_state_for_learning = None # 움직일 수 없으면 학습하지 않음
            return 0, {}, None

        # 다음 업데이트를 위해 현재 상태와 평가 가치를 저장
        self.last_state_for_learning = {
            'board': current_board_for_eval,
            'value': current_value,
            'next_value_pred': best_score # V(s')에 대한 예측
        }

        return best_move, analysis_data, None

    def perform_update(self, reward, next_board_real):
        """한 수가 실행된 후, 실제 보상과 다음 상태를 받아 네트워크를 업데이트합니다."""
        if self.last_state_for_learning is None:
            return

        # TD 오차 계산: error = r + V(s') - V(s)
        # 여기서 V(s')는 get_move에서 예측한 다음 상태의 가치(next_value_pred)를 사용합니다.
        # 실제 다음 상태(next_board_real)를 다시 평가하지 않아 계산 비용을 줄입니다.
        td_error = reward + self.last_state_for_learning['next_value_pred'] - self.last_state_for_learning['value']
        
        # 저장된 이전 상태(s)의 보드에 대해 오차를 전파하여 학습
        self.network.update(self.last_state_for_learning['board'], td_error, self.learning_rate)

        self.update_count += 1
        if self.update_count % self.save_interval == 0:
            self.save()

    def save(self):
        """현재 네트워크 가중치를 파일에 저장합니다."""
        self.network.save_weights(self.weights_path)