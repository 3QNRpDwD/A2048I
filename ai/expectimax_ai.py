from .base_ai import BaseAI
from .heuristic import evaluate, DEFAULT_WEIGHTS
import numpy as np

class ExpectimaxAI(BaseAI):
    def __init__(self, depth=3):
        self.depth = depth
        self.heuristic_func = evaluate
        self.heuristic_weights = DEFAULT_WEIGHTS

    def get_move(self, game):
        best_move, best_score = -1, -np.inf
        analysis_data = {}
        search_tree = {'name': 'Root', 'children': []}

        for move in range(4): # 0:상, 1:하, 2:좌, 3:우
            temp_game = game.clone()
            board_changed = temp_game.move(move)

            if not board_changed:
                analysis_data[move] = -np.inf
                continue

            score, child_node = self.expectimax(temp_game, self.depth - 1, is_max_turn=False)
            analysis_data[move] = score
            child_node['name'] = f"{['Up','Down','Left','Right'][move]}: {score:.1f}"
            search_tree['children'].append(child_node)

            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move == -1:
            return 0, {}, None

        return best_move, analysis_data, search_tree

    def expectimax(self, game, depth, is_max_turn):
        if depth == 0 or game.game_over:
            score = self.heuristic_func(game.board, self.heuristic_weights)
            return score, {'name': f'Leaf: {score:.1f}', 'board': game.board, 'children': []}

        if is_max_turn: # AI의 턴 (Max 노드)
            max_score, best_child = -np.inf, None
            node = {'name': 'Max', 'board': game.board, 'children': []}
            for move in range(4):
                temp_game = game.clone()
                if temp_game.move(move):
                    score, child_node = self.expectimax(temp_game, depth - 1, is_max_turn=False)
                    child_node['name'] = f"{['Up','Down','Left','Right'][move]}: {score:.1f}"
                    node['children'].append(child_node)
                    if score > max_score:
                        max_score = score
                        best_child = child_node
            return max_score, node
        else: # 컴퓨터의 턴 (Chance 노드)
            total_score = 0
            empty_tiles = game.get_empty_tiles()
            node = {'name': 'Chance', 'board': game.board, 'children': []}
            if not empty_tiles:
                return self.heuristic_func(game.board, self.heuristic_weights), node

            num_empty = len(empty_tiles)
            # 모든 빈칸에 대해 2와 4가 나올 경우를 모두 계산하면 너무 복잡해지므로,
            # 대표적인 몇 개만 샘플링하거나, 하나의 빈칸에 대해서만 계산하여 근사.
            # 여기서는 간단하게 하나의 빈칸에 대해서만 계산.
            pos = empty_tiles[0]
            
            # 2가 나올 경우
            game_with_2 = game.clone()
            game_with_2.board[pos] = 2
            score_2, child_node_2 = self.expectimax(game_with_2, depth - 1, is_max_turn=True)
            child_node_2['name'] = f"Tile 2: {score_2:.1f}"
            node['children'].append(child_node_2)
            total_score += 0.9 * score_2

            # 4가 나올 경우
            game_with_4 = game.clone()
            game_with_4.board[pos] = 4
            score_4, child_node_4 = self.expectimax(game_with_4, depth - 1, is_max_turn=True)
            child_node_4['name'] = f"Tile 4: {score_4:.1f}"
            node['children'].append(child_node_4)
            total_score += 0.1 * score_4
            
            return total_score, node