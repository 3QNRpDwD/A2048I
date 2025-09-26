import pygame
import config
import numpy as np

class Tile:
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos  # board 위치 (r, c)
        self.pixel_pos = self._get_pixel_pos(pos)
        self.is_new = False
        self.is_merged = False
        self.scale = 1.0

    def _get_pixel_pos(self, pos):
        """보드 좌표 (r, c)를 픽셀 좌표로 변환"""
        r, c = pos
        x = config.TILE_PADDING + c * (config.TILE_SIZE + config.TILE_PADDING)
        y = config.TILE_PADDING + r * (config.TILE_SIZE + config.TILE_PADDING)
        return [x, y]

    def start_move(self, origin_pos, target_pos, duration, merged=False):
        """애니메이션 제거 버전: 즉시 목적지로 위치 설정.
        함수 시그니처는 변경하지 않았습니다.
        """
        # 즉시 목표 위치로 배치
        self.pixel_pos = list(self._get_pixel_pos(target_pos))
        self.is_new = (origin_pos is None)
        self.is_merged = merged
        self.scale = 1.0

    def update(self, dt):
        # 애니메이션이 삭제되어 더 이상 시간 기반 업데이트가 필요하지 않습니다.
        return

    def draw(self, surface):
        size = config.TILE_SIZE * self.scale
        offset = (config.TILE_SIZE - size) / 2
        x, y = self.pixel_pos
        rect = pygame.Rect(x + offset, y + offset, size, size)
        pygame.draw.rect(surface, config.TILE_COLORS.get(self.value, (0,0,0)), rect, border_radius=3)

        if self.value != 0:
            text_color = config.TEXT_COLORS.get(self.value, (255,255,255))
            font_size = int(config.TILE_FONT_SIZE)
            font = pygame.font.Font(None, max(20, font_size))
            text_surface = font.render(str(self.value), True, text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            surface.blit(text_surface, text_rect)

class BoardRenderer:
    def __init__(self):
        self.tiles = []
        self.is_animating = False

    def set_board(self, board):
        self.tiles = [Tile(val, (r,c)) for (r,c), val in np.ndenumerate(board) if val != 0]
        for tile in self.tiles:
            tile.is_new = False
            tile.scale = 1.0

    def start_animation(self, old_board, new_board, move_direction, duration_seconds):
            """
            애니메이션 제거 상태에서의 동기화 안정화 버전.
            - results (sources -> merged results)를 new_board 상의 실제 non-zero 타겟 칸들에만 매핑합니다.
            - 함수 시그니처 변경 없음.
            """
            self.is_animating = False
            self.tiles = []
            tracker = np.copy(old_board)
            n = config.BOARD_SIZE

            for idx in range(n):
                line_positions = self._line_positions(idx, move_direction)

                # movement order 기준으로 소스(값 != 0) 수집
                sources = [((r, c), tracker[r, c]) for (r, c) in line_positions if tracker[r, c] != 0]

                # sources -> results (2048 규칙대로)
                results = []
                i = 0
                while i < len(sources):
                    (p1, v1) = sources[i]
                    if i + 1 < len(sources) and sources[i + 1][1] == v1:
                        p2, v2 = sources[i + 1]
                        # 병합: origin은 더 멀리 있는 쪽(p2)로 지정 (원래 UX 유지)
                        results.append((v1 * 2, True, [p1, p2], p2))
                        i += 2
                    else:
                        results.append((v1, False, [p1], p1))
                        i += 1

                # new_board 상에서 이 라인의 실제 타겟(값 != 0) 위치들만 추출
                target_positions = [pos for pos in line_positions if new_board[pos] != 0]

                # results를 target_positions에 맞춰 차례로 할당
                # 만약 결과 수가 타겟 수보다 작으면(예상치 않음) 남는 target은 new tile로 처리
                for j, target_pos in enumerate(target_positions):
                    if j < len(results):
                        val_res, merged, srcs, origin_pos = results[j]
                        # 화면에 표시할 값은 new_board의 값을 우선 사용
                        nb_val = new_board[target_pos]
                        tile_val = nb_val if nb_val != 0 else val_res
                        tile = Tile(tile_val, target_pos)
                        tile.is_new = (len(srcs) == 0) or origin_pos is None
                        tile.is_merged = merged
                        # 애니메이션 없음 -> 즉시 위치 설정
                        tile.pixel_pos = tile._get_pixel_pos(target_pos)
                        self.tiles.append(tile)

                        # 사용한 소스는 tracker에서 제거
                        for (rs, cs) in srcs:
                            tracker[rs, cs] = 0
                    else:
                        # results가 부족한 경우: new_board에 있는 값을 그대로 새 타일로 생성
                        nb_val = new_board[target_pos]
                        tile = Tile(nb_val, target_pos)
                        tile.is_new = True
                        tile.is_merged = False
                        tile.pixel_pos = tile._get_pixel_pos(target_pos)
                        self.tiles.append(tile)

                # 만약 results가 타겟보다 많은 (보통은 일어나지 않음) 경우,
                # 남는 results 항목은 안전하게 무시하거나 로그 처리(디버깅용)할 수 있음.
                # 여기서는 tracker를 정리
                # 이미 제거한 소스들 외에 남은 소스는 제거해 두자
                for (_val, _merged, srcs, _origin) in results[len(target_positions):]:
                    for (rs, cs) in srcs:
                        tracker[rs, cs] = 0

    def _line_positions(self, idx, direction):
        n = config.BOARD_SIZE
        if direction == 0:  # Up: top->bottom
            return [(r, idx) for r in range(n)]
        elif direction == 1:  # Down: bottom->top
            return [(r, idx) for r in reversed(range(n))]
        elif direction == 2:  # Left: left->right
            return [(idx, c) for c in range(n)]
        elif direction == 3:  # Right: right->left
            return [(idx, c) for c in reversed(range(n))]
        else:
            return [(r, c) for r in range(n) for c in range(n)]

    def update(self, dt):
        # 더 이상 애니메이션을 업데이트하지 않으므로 아무 작업도 하지 않음
        return

    def draw(self, surface, x_offset, y_offset):
        board_width = config.BOARD_SIZE * config.TILE_SIZE + (config.BOARD_SIZE + 1) * config.TILE_PADDING
        board_height = board_width
        board_surface = pygame.Surface((board_width, board_height))
        board_surface.fill((187, 173, 160))

        for r in range(config.BOARD_SIZE):
            for c in range(config.BOARD_SIZE):
                tile_x = config.TILE_PADDING + c * (config.TILE_SIZE + config.TILE_PADDING)
                tile_y = config.TILE_PADDING + r * (config.TILE_SIZE + config.TILE_PADDING)
                pygame.draw.rect(board_surface, config.TILE_COLORS[0], 
                               (tile_x, tile_y, config.TILE_SIZE, config.TILE_SIZE), 
                               border_radius=3)

        for tile in sorted(self.tiles, key=lambda t: 1 if t.is_merged else 0):
            tile.draw(board_surface)

        surface.blit(board_surface, (x_offset, y_offset))

class GameRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.overlay_font = pygame.font.Font(None, 80)
        self.tree_font = pygame.font.Font(None, 22)
        self.preview_board_surface = pygame.Surface((200, 200))

    def draw_main_ui(self, games, analyses, turn_counts, board_renderers, thinking_ai=None):
        self.screen.fill(config.BACKGROUND_COLOR)
        ai_names = list(games.keys())
        num_ais = len(ai_names)
        
        panel_width = config.BOARD_SIZE * config.TILE_SIZE + (config.BOARD_SIZE + 1) * config.TILE_PADDING
        total_width = num_ais * panel_width + (num_ais - 1) * config.PANEL_GAP
        start_x = (self.screen.get_width() - total_width) // 2

        for i, ai_name in enumerate(ai_names):
            game = games[ai_name]
            analysis = analyses[ai_name]
            board_renderer = board_renderers[ai_name]
            
            x = start_x + i * (panel_width + config.PANEL_GAP)
            y = 100 # Constant y position

            title = config.UI_FONT.render(f"{i+1}. {ai_name}", True, (0,0,0))
            self.screen.blit(title, (x, y))
            score_text = config.SCORE_FONT.render(f"Score: {game.score}", True, (0,0,0))
            self.screen.blit(score_text, (x, y + 40))
            turn_text = config.UI_FONT.render(f"Turn: {turn_counts[ai_name]}", True, (0,0,0))
            self.screen.blit(turn_text, (x + 250, y + 45))

            board_y = y + 90
            board_renderer.draw(self.screen, x, board_y)
            
            if ai_name == thinking_ai:
                self.draw_thinking_overlay(self.screen, x, board_y)
            elif game.game_over:
                self.draw_game_status_overlay(self.screen, "GAME OVER", x, board_y)
            elif game.win:
                 self.draw_game_status_overlay(self.screen, "VICTORY!", x, board_y)
            
            self.draw_ai_analysis_panel(self.screen, ai_name, analysis, x, y + 90 + 550)

    def draw_thinking_overlay(self, surface, x_offset, y_offset):
        board_width = config.BOARD_SIZE * config.TILE_SIZE + (config.BOARD_SIZE + 1) * config.TILE_PADDING
        overlay = pygame.Surface((board_width, board_width), pygame.SRCALPHA)
        overlay.fill((100, 100, 100, 128))
        text_surface = self.overlay_font.render("Thinking...", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(board_width / 2, board_width / 2))
        overlay.blit(text_surface, text_rect)
        surface.blit(overlay, (x_offset, y_offset))

    def draw_game_status_overlay(self, surface, text, x_offset, y_offset):
        board_width = config.BOARD_SIZE * config.TILE_SIZE + (config.BOARD_SIZE + 1) * config.TILE_PADDING
        overlay = pygame.Surface((board_width, board_width), pygame.SRCALPHA)
        if text == "VICTORY!":
            overlay.fill((237, 194, 46, 128))
            color = (255, 255, 255)
        else:
            overlay.fill((255, 255, 255, 128))
            color = (119, 110, 101)
        text_surface = self.overlay_font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(board_width / 2, board_width / 2))
        overlay.blit(text_surface, text_rect)
        surface.blit(overlay, (x_offset, y_offset))

    def draw_ai_analysis_panel(self, surface, ai_name, analysis_data, x_offset, y_offset):
        title_surface = config.UI_FONT.render(f"{ai_name} Analysis", True, (0,0,0))
        surface.blit(title_surface, (x_offset, y_offset))
        if not analysis_data: return
        moves = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        y_pos = y_offset + 40
        sorted_moves = sorted(analysis_data.items(), key=lambda item: item[1], reverse=True)
        for move, score in sorted_moves:
            move_name = moves.get(move, 'Unknown')
            score_text = f"{move_name}: {score:.2f}"
            text_surface = config.UI_FONT_SMALL.render(score_text, True, (80, 80, 80))
            surface.blit(text_surface, (x_offset, y_pos))
            y_pos += 30

    def draw_post_game_screen(self, trees, interaction_data):
        pass
