import numpy as np
from .base_ai import BaseAI
from .policy_model import PolicyModel

class OptimizedReinforceAI(BaseAI):
    """배치 학습과 성능 최적화를 포함한 개선된 REINFORCE AI"""
    
    def __init__(self, learning_rate=0.01, batch_size=5, update_frequency=10):
        self.model = PolicyModel()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        # 배치 학습을 위한 저장소
        self.episode_history = []
        self.game_history = []
        self.update_count = 0
        self.game_count = 0
        
        # 성능 최적화를 위한 설정
        self.last_state_action = None
        self.recent_rewards = []
        self.avg_game_length = 100
        
    def get_move(self, game):
        """정책 모델의 확률 분포에 따라 행동을 선택합니다."""
        board = game.board
        
        # 1. 정책 모델로부터 행동 확률을 얻습니다.
        action_probs = self.model.get_action_probs(board)
        
        # 2. 확률에 따라 행동을 샘플링합니다.
        move = np.random.choice(len(action_probs), p=action_probs)
        
        # 3. 나중에 학습에 사용하기 위해 상태와 행동을 저장합니다.
        self.last_state_action = {"board": np.copy(board), "action": move}
        
        # 분석 데이터는 각 행동의 확률로 채웁니다.
        analysis_data = {i: prob for i, prob in enumerate(action_probs)}
        
        return move, analysis_data, None
    
    def post_move_update(self, reward, game_over):
        """
        배치 학습을 위한 보상 누적 및 주기적 업데이트
        """
        if self.last_state_action is None:
            return
        
        # 1. 보상 누적
        board = self.last_state_action["board"]
        action = self.last_state_action["action"]
        self.episode_history.append((board, action, reward))
        self.recent_rewards.append(reward)
        
        # 최근 보상 기록 제한
        if len(self.recent_rewards) > 1000:
            self.recent_rewards.pop(0)
        
        self.last_state_action = None
        
        # 2. 게임이 종료되면 배치에 추가
        if game_over:
            self.game_count += 1
            self.game_history.append(self.episode_history)
            self.episode_history = []
            
            # 평균 게임 길이 업데이트
            self.avg_game_length = 0.9 * self.avg_game_length + 0.1 * len(self.recent_rewards)
            self.recent_rewards = []
            
            # 3. 배치 크기 도달 시 학습
            if len(self.game_history) >= self.batch_size:
                self.batch_update()
                self.game_history = []
            
            # 4. 주기적 학습 (게임 수 기반)
            elif self.game_count % self.update_frequency == 0:
                if self.game_history:
                    self.batch_update()
                    self.game_history = []
    
    def batch_update(self):
        """
        배치 단위로 모델을 업데이트합니다.
        """
        if not self.game_history:
            return
        
        self.update_count += 1
        total_episodes = len(self.game_history)
        
        # 배치 크기에 따라 학습률 조정
        adaptive_lr = self.learning_rate * (self.batch_size / max(total_episodes, 1))
        
        # 모든 에피소드에 대해 업데이트
        all_updates = []
        for episode in self.game_history:
            if episode:  # 빈 에피소드 제외
                # 이 에피소드의 총 보상 계산
                total_reward = sum(reward for _, _, reward in episode)
                
                # 정규화된 보상 사용
                normalized_reward = total_reward / max(len(episode), 1)
                
                # 각 스텝에 대한 업데이트 준비
                for board, action, reward in episode:
                    all_updates.append((board, action, normalized_reward))
        
        # 배치 업데이트 수행
        if all_updates:
            self.model.batch_update(all_updates, adaptive_lr)
            
            # 통계 출력
            avg_reward = np.mean([sum(r for _, _, r in ep) for ep in self.game_history if ep])
            avg_length = np.mean([len(ep) for ep in self.game_history if ep])
            
            if self.update_count % 5 == 0:  # 5번마다 한 번 출력
                print(f"[OptimizedReinforceAI] 배치 학습 완료 (배치: {total_episodes}게임, "
                      f"평균 보상: {avg_reward:.1f}, 평균 길이: {avg_length:.1f}, "
                      f"lr: {adaptive_lr:.4f})")

# --- 성능 최적화된 정책 모델 ---
class OptimizedPolicyModel(PolicyModel):
    """성능을 최적화한 정책 모델"""
    
    def __init__(self):
        super().__init__()
        # 가중치 캐싱을 위한 저장소
        self.weight_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
    
    def get_action_probs(self, board):
        """캐싱을 활용한 행동 확률 계산"""
        # 보드의 해시 생성 (빠른 비교를 위해)
        board_hash = hash(board.tobytes())
        
        # 캐시 확인
        if board_hash in self.weight_cache:
            self.cache_hit_count += 1
            return self.weight_cache[board_hash]
        
        self.cache_miss_count += 1
        
        # 원래 계산 수행
        probs = super().get_action_probs(board)
        
        # 캐시에 저장 (최근 1000개만)
        if len(self.weight_cache) > 1000:
            # LRU: 가장 오래된 항목 제거
            oldest_key = next(iter(self.weight_cache))
            del self.weight_cache[oldest_key]
        
        self.weight_cache[board_hash] = probs
        return probs
    
    def batch_update(self, updates, learning_rate):
        """배치 단위로 가중치 업데이트"""
        if not updates:
            return
        
        # 배치 크기에 따라 업데이트 빈도 조정
        batch_size = len(updates)
        adjusted_lr = learning_rate / np.sqrt(batch_size)
        
        # 그라데이션 누적
        total_gradient = np.zeros_like(self.weights)
        
        for board, action, reward in updates:
            # 순전파
            probs = self.get_action_probs(board)
            
            # 그라데이션 계산 (벡터화)
            gradient = -probs
            gradient[action] += 1
            
            # 보상 가중치 적용
            gradient *= reward * adjusted_lr
            
            # 누적
            total_gradient += gradient
        
        # 평균 그라데이트 적용
        total_gradient /= batch_size
        
        # 가중치 업데이트
        self.weights -= total_gradient
        
        # 정규화
        weight_norm = np.linalg.norm(self.weights)
        if weight_norm > 10.0:  # 가중치 클리핑
            self.weights = self.weights / weight_norm * 10.0
        
        # 캐시 초기화 (가중치가 바뀌었으므로)
        self.weight_cache.clear()
    
    def get_cache_stats(self):
        """캐시 성능 통계 반환"""
        total = self.cache_hit_count + self.cache_miss_count
        if total == 0:
            return {"hit_rate": 0.0, "size": 0}
        
        return {
            "hit_rate": self.cache_hit_count / total,
            "size": len(self.weight_cache),
            "hits": self.cache_hit_count,
            "misses": self.cache_miss_count
        }