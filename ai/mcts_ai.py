
from .base_ai import BaseAI
import numpy as np
import random
import math

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # Game2048 인스턴스
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(range(4)) # 0,1,2,3

    def select_child(self, exploration_param=1.414):
        """UCT(Upper Confidence Bound for Trees)를 사용하여 최적의 자식 노드를 선택합니다."""
        best_child = max(self.children, 
                         key=lambda c: (c.wins / c.visits) + exploration_param * math.sqrt(math.log(self.visits) / c.visits))
        return best_child

    def expand(self):
        """아직 시도하지 않은 움직임으로 자식 노드를 확장합니다."""
        move = self.untried_moves.pop()
        new_state = self.state.clone()
        board_changed = new_state.move(move)
        
        # 움직임이 유효할 때만 자식 노드 생성
        if board_changed:
            child_node = MCTSNode(new_state, parent=self, move=move)
            self.children.append(child_node)
            return child_node
        return None

    def update(self, result):
        """시뮬레이션 결과를 역전파합니다."""
        self.visits += 1
        self.wins += result

class MCTSAI(BaseAI):
    def __init__(self, iterations=100):
        self.iterations = iterations

    def get_move(self, game):
        root = MCTSNode(state=game)

        for _ in range(self.iterations):
            node = root
            # 1. 선택 (Selection)
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()

            # 2. 확장 (Expansion)
            if node.untried_moves != []:
                new_node = node.expand()
                if new_node:
                    node = new_node

            # 3. 시뮬레이션 (Simulation)
            sim_state = node.state.clone()
            while not sim_state.game_over:
                sim_state.move(random.randint(0, 3))
            
            # 4. 역전파 (Backpropagation)
            while node is not None:
                node.update(sim_state.score)
                node = node.parent

        # 가장 많이 방문된 노드의 움직임을 선택
        best_child = max(root.children, key=lambda c: c.visits)
        best_move = best_child.move

        analysis_data = {c.move: c.visits for c in root.children}
        search_tree = self._convert_to_dict_tree(root)

        return best_move, analysis_data, search_tree

    def _convert_to_dict_tree(self, node):
        """MCTSNode를 렌더러가 사용할 수 있는 딕셔너리 트리로 변환합니다."""
        if node.move is not None:
            move_name = ['Up', 'Down', 'Left', 'Right'][node.move]
            name = f"{move_name} (V: {node.visits}, W: {node.wins/node.visits:.0f})"
        else:
            name = f"Root (V: {node.visits})"

        tree = {
            'name': name,
            'board': node.state.board,
            'children': [self._convert_to_dict_tree(c) for c in node.children]
        }
        return tree
