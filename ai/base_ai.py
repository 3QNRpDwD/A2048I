
class BaseAI:
    """모든 AI 클래스가 상속받을 기본 클래스입니다."""
    def get_move(self, game):
        """
        주어진 게임 상태(game)를 기반으로 최적의 움직임을 결정합니다.
        
        Args:
            game (Game2048): 현재 게임 상태를 담고 있는 인스턴스.
            
        Returns:
            int: 0:상, 1:하, 2:좌, 3:우 중 하나의 움직임.
            dict: (구현 예정) 분석 데이터.
            object: (구현 예정) 탐색 트리.
        """
        raise NotImplementedError("이 메소드는 서브클래스에서 반드시 구현되어야 합니다.")
