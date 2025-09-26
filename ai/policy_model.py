import numpy as np
from .heuristic import monotonicity_score, smoothness_score

class PolicyModel:
    """REINFORCE 알고리즘을 위한 간단한 선형 정책 모델입니다."""
    def __init__(self, num_features=4, num_actions=4):
        """
        Args:
            num_features (int): 상태를 설명하는 특징의 수.
            num_actions (int): 가능한 행동의 수 (상,하,좌,우 = 4).
        """
        self.num_features = num_features
        self.num_actions = num_actions
        # 가중치를 무작위로 초기화합니다. (특징 수 x 행동 수)
        self.weights = np.random.randn(num_features, num_actions) * 0.1

    def _extract_features(self, board):
        """게임 보드로부터 특징 벡터를 추출합니다."""
        mono_score = monotonicity_score(board)
        smooth_score = smoothness_score(board)
        empty_tiles = np.log(np.count_nonzero(board == 0) + 1)
        max_value = np.log2(np.max(board) + 1)
        
        # 특징 벡터를 정규화하거나 스케일링하면 학습 안정성에 도움이 될 수 있습니다.
        features = np.array([mono_score, smooth_score, empty_tiles, max_value])
        return features

    def get_action_probs(self, board):
        """특징과 가중치를 기반으로 각 행동의 확률을 계산합니다."""
        features = self._extract_features(board)
        # 선형 결합: z = W * x
        logits = np.dot(features, self.weights)
        
        # Softmax 함수를 통해 확률 분포를 얻습니다.
        # 안정적인 계산을 위해 logit의 최대값을 빼줍니다.
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def update(self, episode_history, learning_rate):
        """REINFORCE 알고리즘을 사용해 에피소드 기록으로 가중치를 업데이트합니다."""
        # 에피소드 기록: [(state, action, reward), ...]
        
        # 1. 각 시점(time step)의 리턴(return)을 계산합니다.
        # 리턴 = 해당 시점부터 에피소드 끝까지 받은 보상의 합
        returns = []
        cumulative_reward = 0
        for _, _, reward in reversed(episode_history):
            cumulative_reward += reward
            returns.insert(0, cumulative_reward)
        
        # 리턴을 표준화하여 학습 안정성 향상
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # 2. 정책 경사(Policy Gradient)를 계산하고 가중치를 업데이트합니다.
        grad_weights = np.zeros_like(self.weights)
        
        for i, (board, action, _) in enumerate(episode_history):
            G_t = returns[i]
            features = self._extract_features(board)
            action_probs = self.get_action_probs(board)

            for a in range(self.num_actions):
                # ∂log(π(a|s)) / ∂W 계산
                # 여기서는 softmax를 사용한 선형 모델이므로, 그래디언트는 다음과 같이 계산됩니다.
                # grad = features * (I(a=chosen_action) - P(a|s))
                indicator = 1 if a == action else 0
                grad_log_pi = np.outer(features, (indicator - action_probs[a]))
                
                # 최종 그래디언트에 리턴 G_t를 곱하여 누적
                grad_weights += grad_log_pi * G_t

        # 학습률을 적용하여 가중치 업데이트 (경사 상승법)
        self.weights += learning_rate * (grad_weights / len(episode_history))
