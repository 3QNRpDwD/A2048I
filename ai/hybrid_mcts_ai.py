from .mcts_ai import MCTSAI, MCTSNode
import numpy as np

class HybridMCTSAI(MCTSAI):
    """
    MCTS AI that uses a simple heuristic for the simulation phase
    instead of random rollouts.
    """
    def __init__(self, iterations):
        super().__init__(iterations)

    def _simulate(self, node: MCTSNode) -> int:
        """
        Run a simulation from the given node to the end of the game.
        Instead of random moves, use a simple heuristic to choose moves.
        """
        game = node.state.clone()

        # Simulate until the game is over
        while not game.game_over:
            # Use a simple heuristic: choose the move that gives the highest immediate score.
            # This is equivalent to a depth-1 expectimax/greedy search.
            possible_moves = game.get_possible_moves()
            if not possible_moves:
                break

            best_move = -1
            best_score = -1

            # Evaluate each possible move
            for move in possible_moves:
                g_clone = game.clone()
                board_changed = g_clone.move(move)
                if board_changed:
                    if g_clone.score > best_score:
                        best_score = g_clone.score
                        best_move = move

            if best_move != -1:
                game.move(best_move)
            else:
                # No move improves the score or is valid, game is likely over
                break
        
        return game.score
