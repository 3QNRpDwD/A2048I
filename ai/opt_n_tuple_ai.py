from .base_ai import BaseAI
from .n_tuple_network import NTupleNetwork
import numpy as np

class OptimizedNTupleAI(BaseAI):
    def __init__(self, weights_path="n_tuple_weights.npy", learning_rate=0.0025, 
                 save_interval=1000, update_frequency=5):
        """성능 최적화된 N-Tuple 네트워크 AI"""
        self.weights_path = weights_path
        self.base_learning_rate = learning_rate
        self.save_interval = save_interval
        self.update_frequency = update_frequency
        
        self.network = NTupleNetwork()
        self.network.load_weights(self.weights_path)
        
        # 성능 최적화를 위한 설정
        self.last_state_for_learning = None
        self.update_count = 0
        self.game_count = 0
        
        # 업데이트 누적을 위한 버퍼
        self.update_buffer = []
        self.buffer_size = update_frequency
        
        # 캐싱 시스템
        self.evaluation_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 학습률 조절을 위한 통계
        self.recent_td_errors = []
        self.adaptive_lr_multiplier = 1.0

    def get_update_count(self):
        return self.update_count
        
    def get_move(self, game):
        """캐싱과 성능 최적화를 포함한 수 선택"""
        current_board_for_eval = np.copy(game.board)
        
        # 캐시 확인
        board_hash = hash(current_board_for_eval.tobytes())
        if board_hash in self.evaluation_cache:
            self.cache_hit_count += 1
            current_value = self.evaluation_cache[board_hash]
        else:
            self.cache_miss_count += 1
            current_value = self.network.evaluate(current_board_for_eval)
            
            # 캐시에 저장 (크기 제한)
            if len(self.evaluation_cache) > 500:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.evaluation_cache))
                del self.evaluation_cache[oldest_key]
            
            self.evaluation_cache[board_hash] = current_value
        
        best_move, best_score = -1, -np.inf
        analysis_data = {}
        
        # 가능한 모든 수에 대해 평가
        for move in range(4):
            temp_game = game.clone()
            board_changed, _ = temp_game.move(move)
            
            if not board_changed:
                analysis_data[move] = -np.inf
                continue
            
            # 다음 상태 평가 (캐시 활용)
            next_board_hash = hash(temp_game.board.tobytes())
            if next_board_hash in self.evaluation_cache:
                self.cache_hit_count += 1
                score = self.evaluation_cache[next_board_hash]
            else:
                self.cache_miss_count += 1
                score = self.network.evaluate(temp_game.board)
                
                # 캐시에 저장
                if len(self.evaluation_cache) < 500:
                    self.evaluation_cache[next_board_hash] = score
            
            analysis_data[move] = score
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move == -1:
            self.last_state_for_learning = None
            return 0, {}, None
        
        # 학습을 위한 상태 저장
        self.last_state_for_learning = {
            'board': current_board_for_eval,
            'value': current_value,
            'next_value_pred': best_score,
            'board_hash': board_hash
        }
        
        return best_move, analysis_data, None
    
    def perform_update(self, reward, next_board_real):
        """
        누적된 업데이트를 주기적으로 처리
        """
        if self.last_state_for_learning is None:
            return
        
        # TD 오차 계산
        # 실제 다음 상태 평가 (가끔씩만 수행하여 비용 절감)
        if self.game_count % 10 == 0:  # 10번마다 한 번 실제 평가
            next_value_real = self.network.evaluate(next_board_real)
        else:
            next_value_real = self.last_state_for_learning['next_value_pred']
        
        td_error = reward + next_value_real - self.last_state_for_learning['value']
        
        # 최근 TD 오차 저장 (학습률 조절용)
        self.recent_td_errors.append(abs(td_error))
        if len(self.recent_td_errors) > 100:
            self.recent_td_errors.pop(0)
        
        # 업데이트 버퍼에 추가
        self.update_buffer.append({
            'board': self.last_state_for_learning['board'],
            'td_error': td_error,
            'board_hash': self.last_state_for_learning['board_hash']
        })
        
        # 버퍼가 가득 차면 배치 업데이트 수행
        if len(self.update_buffer) >= self.buffer_size:
            self.batch_update()
        
        self.last_state_for_learning = None
    
    def batch_update(self):
        """
        누적된 업데이트를 한 번에 처리
        """
        if not self.update_buffer:
            return
        
        self.update_count += 1
        self.game_count += 1
        
        # 적응형 학습률 계산
        if self.recent_td_errors:
            avg_td_error = np.mean(self.recent_td_errors)
            # TD 오차가 크면 학습률을 줄임
            if avg_td_error > 1.0:
                self.adaptive_lr_multiplier *= 0.9
            elif avg_td_error < 0.1:
                self.adaptive_lr_multiplier *= 1.1
            
            # 학습률 멀티플라이어 제한
            self.adaptive_lr_multiplier = max(0.1, min(2.0, self.adaptive_lr_multiplier))
        
        adaptive_lr = self.base_learning_rate * self.adaptive_lr_multiplier
        
        # 배치 업데이트 수행
        total_error = 0.0
        for update in self.update_buffer:
            self.network.update(update['board'], update['td_error'], adaptive_lr)
            total_error += abs(update['td_error'])
            
            # 캐시에서 제거 (가중치가 바뀌었으므로)
            if update['board_hash'] in self.evaluation_cache:
                del self.evaluation_cache[update['board_hash']]
        
        avg_error = total_error / len(self.update_buffer)
        
        # 버퍼 초기화
        self.update_buffer = []
        
        # 주기적으로 저장 및 통계 출력
        if self.update_count % self.save_interval == 0:
            self.save()
            cache_stats = self.get_cache_stats()
            print(f"[OptimizedNTupleAI] 배치 업데이트 완료 (업데이트: {self.update_count}, "
                  f"게임: {self.game_count}, 평균 오차: {avg_error:.4f}, "
                  f"lr: {adaptive_lr:.4f}, 캐시 적중률: {cache_stats['hit_rate']:.2%})")
    
    def save(self):
        """현재 네트워크 가중치를 파일에 저장합니다."""
        self.network.save_weights(self.weights_path)
    
    def get_cache_stats(self):
        """캐시 성능 통계 반환"""
        total = self.cache_hit_count + self.cache_miss_count
        if total == 0:
            return {"hit_rate": 0.0, "size": 0}
        
        return {
            "hit_rate": self.cache_hit_count / total,
            "size": len(self.evaluation_cache),
            "hits": self.cache_hit_count,
            "misses": self.cache_miss_count
        }

# --- 성능 최적화된 N-Tuple 네트워크 ---
class OptimizedNTupleNetwork(NTupleNetwork):
    """성능을 최적화한 N-Tuple 네트워크"""
    
    def __init__(self):
        super().__init__()
        # 가중치 조회 최적화를 위한 캐시
        self.weight_lookup_cache = {}
        self.lookup_cache_hits = 0
        self.lookup_cache_misses = 0
    
    def get_weight_index(self, tuple_values):
        """가중치 인덱스 조회 캐싱"""
        # tuple_values를 해시 가능한 형태로 변환
        key = tuple(tuple_values)
        
        if key in self.weight_lookup_cache:
            self.lookup_cache_hits += 1
            return self.weight_lookup_cache[key]
        
        self.lookup_cache_misses += 1
        index = super().get_weight_index(tuple_values)
        
        # 캐시에 저장 (크기 제한)
        if len(self.weight_lookup_cache) > 10000:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.weight_lookup_cache))
            del self.weight_lookup_cache[oldest_key]
        
        self.weight_lookup_cache[key] = index
        return index
    
    def update(self, board, td_error, learning_rate):
        """최적화된 가중치 업데이트"""
        # TD 오차가 너무 작으면 업데이트 건
        if abs(td_error) < 0.001:
            return
        
        # 기본 업데이트 수행
        super().update(board, td_error, learning_rate)
        
        # 가중치가 변경되었으므로 관련 캐시 무효화
        self.weight_lookup_cache.clear()
    
    def get_performance_stats(self):
        """성능 통계 반환"""
        total_lookups = self.lookup_cache_hits + self.lookup_cache_misses
        lookup_hit_rate = self.lookup_cache_hits / max(total_lookups, 1)
        
        return {
            'lookup_cache_hit_rate': lookup_hit_rate,
            'lookup_cache_size': len(self.weight_lookup_cache),
            'lookup_cache_hits': self.lookup_cache_hits,
            'lookup_cache_misses': self.lookup_cache_misses
        }