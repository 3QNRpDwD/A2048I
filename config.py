
import pygame

# --- 화면 및 UI 설정 ---
SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 850
PANEL_GAP = 20
BACKGROUND_COLOR = (250, 248, 239)
FPS = 60

# --- 게임 보드 설정 ---
BOARD_SIZE = 4
TILE_SIZE = 120
TILE_PADDING = 15
BOARD_X_OFFSET = 50
BOARD_Y_OFFSET = 150

# --- 폰트 설정 ---
pygame.font.init()
SCORE_FONT = pygame.font.Font(None, 48)
TILE_FONT_SIZE = 55
TILE_FONT = pygame.font.Font(None, TILE_FONT_SIZE)
UI_FONT = pygame.font.Font(None, 36)
UI_FONT_SMALL = pygame.font.Font(None, 28)

# --- 타일 색상 ---
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
    8192: (60, 58, 50),
}

TEXT_COLORS = {
    2: (119, 110, 101),
    4: (119, 110, 101),
    8: (249, 246, 242),
    16: (249, 246, 242),
    32: (249, 246, 242),
    64: (249, 246, 242),
    128: (249, 246, 242),
    256: (249, 246, 242),
    512: (249, 246, 242),
    1024: (249, 246, 242),
    2048: (249, 246, 242),
    4096: (249, 246, 242),
    8192: (249, 246, 242),
}

# --- AI 설정 ---
AI_MOVE_DELAY = 50  # ms, AI가 움직이는 간격
ANIMATION_DURATION = 200 # ms, 타일 애니메이션 지속 시간
EXPECTIMAX_DEPTH = 3 # Expectimax 탐색 깊이
MCTS_ITERATIONS = 100 # MCTS 시뮬레이션 횟수
