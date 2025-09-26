
from .base_ai import BaseAI
import random

class RandomAI(BaseAI):
    """무작위로 움직이는 간단한 AI입니다."""
    def get_move(self, game):
        # 게임 로직이 4방향을 모두 시도하므로, 아무거나 반환해도 됨
        # 실제로는 가능한 움직임 중 하나를 선택해야 하지만,
        # 현재 구조에서는 메인 루프에서 처리 가능
        return random.randint(0, 3), {}, None
