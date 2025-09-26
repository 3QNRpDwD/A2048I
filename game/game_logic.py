
import numpy as np
import random

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        """게임을 초기 상태로 리셋합니다."""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.game_over = False
        self.win = False
        self.add_new_tile(count=2)

    def clone(self):
        """AI 시뮬레이션을 위한 현재 게임 상태의 깊은 복사본을 만듭니다."""
        new_game = Game2048(self.size)
        new_game.board = np.copy(self.board)
        new_game.score = self.score
        new_game.game_over = self.game_over
        new_game.win = self.win
        return new_game

    def get_empty_tiles(self):
        """비어있는 타일의 위치를 (행, 열) 튜플 리스트로 반환합니다."""
        return list(zip(*np.where(self.board == 0)))

    def add_new_tile(self, count=1):
        """비어있는 위치에 새로운 타일(2 또는 4)을 추가합니다."""
        empty_tiles = self.get_empty_tiles()
        if not empty_tiles:
            return

        for _ in range(count):
            if not empty_tiles:
                break
            pos = random.choice(empty_tiles)
            empty_tiles.remove(pos)
            value = 4 if random.random() < 0.1 else 2
            self.board[pos] = value

    def move(self, direction):
        """
        주어진 방향으로 보드를 움직이고, 변화가 있었다면 새 타일을 추가합니다.
        0:상, 1:하, 2:좌, 3:우
        """
        if self.game_over:
            return False

        original_board = np.copy(self.board)
        
        rotated_board = np.rot90(self.board, k=direction)
        new_board, move_score = self._move_left(rotated_board)
        self.board = np.rot90(new_board, k=-direction)
        
        self.score += move_score
        
        if not self.win and 2048 in self.board:
            self.win = True
            
        board_changed = not np.array_equal(original_board, self.board)
        
        if board_changed:
            self.add_new_tile()
        
        if not self.get_available_moves():
            self.game_over = True
        
        return board_changed

    def _move_left(self, board):
        """왼쪽으로 타일을 밀고 합치는 로직"""
        new_board = np.zeros_like(board)
        score = 0
        for i in range(self.size):
            row = board[i][board[i] != 0]
            new_row = []
            j = 0
            while j < len(row):
                if j + 1 < len(row) and row[j] == row[j+1]:
                    new_value = row[j] * 2
                    new_row.append(new_value)
                    score += new_value
                    j += 2
                else:
                    new_row.append(row[j])
                    j += 1
            new_board[i, :len(new_row)] = new_row
        return new_board, score

    def get_available_moves(self):
        """현재 보드에서 가능한 움직임이 있는지 확인합니다."""
        if self.get_empty_tiles():
            return True
        
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return True
        
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i, j] == self.board[i+1, j]:
                    return True
        
        return False
