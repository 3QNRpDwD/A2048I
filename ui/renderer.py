import pygame
import config
import numpy as np
from ai.mcts_ai import MCTSAI
from ai.n_tuple_ai import NTupleAI
from ai.expectimax_ai import ExpectimaxAI

class Tile:
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos
        self.pixel_pos = self._get_pixel_pos(pos)

    def _get_pixel_pos(self, pos):
        r, c = pos
        x = config.TILE_PADDING + c * (config.TILE_SIZE + config.TILE_PADDING)
        y = config.TILE_PADDING + r * (config.TILE_SIZE + config.TILE_PADDING)
        return [x, y]

    def draw(self, surface):
        rect = pygame.Rect(self.pixel_pos[0], self.pixel_pos[1], config.TILE_SIZE, config.TILE_SIZE)
        pygame.draw.rect(surface, config.TILE_COLORS.get(self.value, (0,0,0)), rect, border_radius=3)
        if self.value != 0:
            text_color = config.TEXT_COLORS.get(self.value, (255,255,255))
            text_surface = config.TILE_FONT.render(str(self.value), True, text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            surface.blit(text_surface, text_rect)

class BoardRenderer:
    def __init__(self):
        self.tiles = []
        self.is_animating = False

    def set_board(self, board):
        self.tiles = [Tile(val, (r,c)) for (r,c), val in np.ndenumerate(board) if val != 0]

    def start_animation(self, old_board, new_board, move_direction, duration_seconds):
        self.is_animating = True
        self.set_board(new_board)
        self.is_animating = False

    def update(self, dt):
        pass

    def draw(self, surface, x_offset, y_offset):
        board_width = config.BOARD_SIZE * config.TILE_SIZE + (config.BOARD_SIZE + 1) * config.TILE_PADDING
        board_surface = pygame.Surface((board_width, board_width))
        board_surface.fill(config.GRID_COLOR)

        for r in range(config.BOARD_SIZE):
            for c in range(config.BOARD_SIZE):
                tile_x = config.TILE_PADDING + c * (config.TILE_SIZE + config.TILE_PADDING)
                tile_y = config.TILE_PADDING + r * (config.TILE_SIZE + config.TILE_PADDING)
                pygame.draw.rect(board_surface, config.TILE_COLORS[0], 
                               (tile_x, tile_y, config.TILE_SIZE, config.TILE_SIZE), 
                               border_radius=3)

        for tile in self.tiles:
            tile.draw(board_surface)

        surface.blit(board_surface, (x_offset, y_offset))

class GameRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.vis_font_small = pygame.font.Font(None, 24)
        self.vis_font_bold = pygame.font.Font(None, 28)
        self.vis_font_title = pygame.font.Font(None, 32)

    def draw_main_ui(self, games, analyses, turn_counts, board_renderers, retry_counts, highest_tiles, 
                     thinking_ai, game_phase, winner_ai_name, last_trees, ais, ga_status=None):
        self.screen.fill(config.BACKGROUND_COLOR)
        ai_names = list(games.keys())
        num_ais = len(ai_names)
        
        panel_width = config.BOARD_SIZE * config.TILE_SIZE + (config.BOARD_SIZE + 1) * config.TILE_PADDING
        # 실시간 분석 패널 높이(약 150px)를 포함하여 패널 전체 높이 재계산
        panel_height = config.PANEL_HEIGHT

        max_cols = 4
        num_rows = (num_ais + max_cols - 1) // max_cols

        total_grid_height = num_rows * panel_height + (num_rows - 1) * config.PANEL_V_GAP
        start_y = max(10, (self.screen.get_height() - total_grid_height) // 2)

        for i, ai_name in enumerate(ai_names):
            row = i // max_cols
            col = i % max_cols

            cols_in_this_row = min(max_cols, num_ais - row * max_cols)
            row_width = cols_in_this_row * panel_width + (cols_in_this_row - 1) * config.PANEL_GAP
            start_x_for_row = (self.screen.get_width() - row_width) // 2

            x = start_x_for_row + col * (panel_width + config.PANEL_GAP)
            y = start_y + row * (panel_height + config.PANEL_V_GAP)

            title_color = (200, 0, 0) if ai_name == winner_ai_name else (0,0,0)
            title = config.UI_FONT.render(f"{i+1}. {ai_name}", True, title_color)
            self.screen.blit(title, (x, y))

            game = games[ai_name]
            score_text = config.SCORE_FONT.render(f"Score: {game.score}", True, (0,0,0))
            self.screen.blit(score_text, (x, y + config.TEXT_V_GAP))
            turn_text = config.UI_FONT.render(f"Turn: {turn_counts[ai_name]}", True, (0,0,0))
            self.screen.blit(turn_text, (x + config.TEXT_GAP, y + config.TEXT_V_GAP + 5))
            retry_text = config.UI_FONT.render(f"Retries: {retry_counts[ai_name]}", True, (0,0,0))
            self.screen.blit(retry_text, (x, y + + config.TEXT_V_GAP + 40))
            highest_tile_text = config.UI_FONT.render(f"Highest: {highest_tiles[ai_name]}", True, (0,0,0))
            self.screen.blit(highest_tile_text, (x + config.TEXT_GAP, y + config.TEXT_V_GAP + 40))

            # --- 유전 알고리즘 상태 표시 ---
            if "Evo" in ai_name and ga_status:
                gen_text = config.UI_FONT.render(f"Gen: {ga_status.get('generation', 0)}", True, (0, 100, 0))
                self.screen.blit(gen_text, (x, y + config.TEXT_V_GAP + 65))
                
                fitness = ga_status.get('best_fitness', 0)
                fitness_text = config.UI_FONT.render(f"Fit: {fitness:,.0f}", True, (0, 100, 0))
                self.screen.blit(fitness_text, (x + config.TEXT_GAP, y + config.TEXT_V_GAP + 65))
            # --- N-Tuple 학습 상태 표시 ---
            elif "N-Tuple" in ai_name:
                ai_instance = ais.get(ai_name)
                if ai_instance:
                    update_count = ai_instance.update_count
                    update_text = config.UI_FONT.render(f"Updates: {update_count}", True, (0, 0, 150))
                    self.screen.blit(update_text, (x, y + config.TEXT_V_GAP + 65))
            # --------------------------

            board_y = y + 120
            analysis_y = board_y + panel_width + 20

            if game_phase == "VICTORY_LAP":
                self.draw_visualization_panel(self.screen, ai_name, ais.get(ai_name), analyses.get(ai_name), last_trees.get(ai_name), x, board_y)
            else:
                board_renderers[ai_name].draw(self.screen, x, board_y)
                if ai_name == thinking_ai:
                    self.draw_thinking_overlay(self.screen, x, board_y)
                elif game.game_over:
                    self.draw_game_status_overlay(self.screen, "GAME OVER", x, board_y)
                elif game.win:
                    self.draw_game_status_overlay(self.screen, "VICTORY!", x, board_y)
                
                self.draw_ai_analysis_panel(self.screen, ai_name, analyses.get(ai_name), x, analysis_y)

    def draw_ai_analysis_panel(self, surface, ai_name, analysis_data, x_offset, y_offset):
        title_surface = config.ANALYSIS_FONT.render(f"{ai_name} Analysis", True, (0,0,0))
        surface.blit(title_surface, (x_offset, y_offset))
        if not analysis_data: return

        moves = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        y_pos = y_offset + 35

        valid_analysis = {k: v for k, v in analysis_data.items() if not np.isinf(v)}
        if not valid_analysis: return

        sorted_moves = sorted(valid_analysis.items(), key=lambda item: item[1], reverse=True)
        for move, score in sorted_moves:
            move_name = moves.get(move, 'Unknown')
            score_text = f"{move_name}: {score:.2f}"
            text_surface = config.ANALYSIS_FONT_SMALL.render(score_text, True, (80, 80, 80))
            surface.blit(text_surface, (x_offset, y_pos))
            y_pos += 30

    def draw_visualization_panel(self, surface, ai_name, ai_instance, analysis, tree, x, y):
        panel_width = config.BOARD_SIZE * config.TILE_SIZE + (config.BOARD_SIZE + 1) * config.TILE_PADDING
        panel_rect = pygame.Rect(x, y, panel_width, panel_width)
        pygame.draw.rect(surface, (240, 240, 240), panel_rect, border_radius=5)
        pygame.draw.rect(surface, (200, 200, 200), panel_rect, 2, border_radius=5)

        title_text = f"'{ai_name}' 최종 결정 과정"
        title_surf = self.vis_font_title.render(title_text, True, (50, 50, 50))
        surface.blit(title_surf, (x + 15, y + 15))

        if isinstance(ai_instance, (MCTSAI)):
            self.draw_mcts_tree(surface, tree, x, y + 60)
        elif isinstance(ai_instance, (ExpectimaxAI, NTupleAI)):
            self.draw_score_analysis(surface, analysis, x, y + 60, "가치 평가")
        else:
            info_text = "시각화 정보 없음"
            info_surf = self.vis_font_small.render(info_text, True, (100, 100, 100))
            surface.blit(info_surf, (x + 20, y + 60))

    def draw_mcts_tree(self, surface, tree, x, y):
        if not tree or 'children' not in tree:
            return
        
        root_pos = (x + 200, y + 50)
        pygame.draw.circle(surface, (100, 100, 200), root_pos, 20)
        root_text = self.vis_font_small.render("Now", True, (255,255,255))
        surface.blit(root_text, root_text.get_rect(center=root_pos))

        children = sorted(tree['children'], key=lambda c: c.get('visits', 0), reverse=True)
        for i, child in enumerate(children[:4]):
            angle = (i - 1.5) * (np.pi / 4) + np.pi / 2
            child_x = root_pos[0] + 150 * np.cos(angle)
            child_y = root_pos[1] + 120 * np.sin(angle)
            child_pos = (child_x, child_y)

            pygame.draw.line(surface, (180, 180, 180), root_pos, child_pos, 2)
            pygame.draw.circle(surface, (200, 100, 100), child_pos, 30)

            move_name = child.get('name', 'N/A').split(':')[0]
            visits = child.get('visits', 0)
            wins = child.get('wins', 0)
            score = wins / visits if visits > 0 else 0

            move_surf = self.vis_font_bold.render(move_name, True, (255,255,255))
            score_surf = self.vis_font_small.render(f"{score:.2f} ({visits})", True, (255,255,255))
            surface.blit(move_surf, move_surf.get_rect(center=(child_pos[0], child_pos[1] - 12)))
            surface.blit(score_surf, score_surf.get_rect(center=(child_pos[0], child_pos[1] + 12)))

    def draw_score_analysis(self, surface, analysis, x, y, title):
        if not analysis:
            return
        
        moves = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        valid_analysis = {k: v for k, v in analysis.items() if not np.isinf(v)}
        if not valid_analysis:
            return

        sorted_moves = sorted(valid_analysis.items(), key=lambda item: item[1], reverse=True)
        
        max_score = max(valid_analysis.values())
        min_score = min(valid_analysis.values())

        y_pos = y + 40
        for move, score in sorted_moves:
            move_name = moves.get(move, 'Unknown')
            text_surf = self.vis_font_bold.render(f"{move_name}: {score:.2f}", True, (80, 80, 80))
            surface.blit(text_surf, (x + 20, y_pos))

            bar_x = x + 150
            bar_width = 220
            
            if max_score > min_score:
                normalized_score = (score - min_score) / (max_score - min_score)
            else:
                normalized_score = 1.0

            bar_length = max(1, int(normalized_score * bar_width))
            bar_color = (100, 150, 200) if score == max_score else (180, 180, 180)
            pygame.draw.rect(surface, bar_color, (bar_x, y_pos, bar_length, 25), border_radius=3)
            y_pos += 50

    def draw_thinking_overlay(self, surface, x_offset, y_offset):
        board_width = config.BOARD_SIZE * config.TILE_SIZE + (config.BOARD_SIZE + 1) * config.TILE_PADDING
        overlay = pygame.Surface((board_width, board_width), pygame.SRCALPHA)
        overlay.fill((100, 100, 100, 128))
        text_surface = config.UI_FONT.render("Thinking...", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(board_width / 2, board_width / 2))
        overlay.blit(text_surface, text_rect)
        surface.blit(overlay, (x_offset, y_offset))

    def draw_game_status_overlay(self, surface, text, x_offset, y_offset):
        board_width = config.BOARD_SIZE * config.TILE_SIZE + (config.BOARD_SIZE + 1) * config.TILE_PADDING
        overlay = pygame.Surface((board_width, board_width), pygame.SRCALPHA)
        color = (119, 110, 101)
        bg_color = (255, 255, 255, 180)
        if text == "VICTORY!":
            color = (255, 255, 255)
            bg_color = (237, 194, 46, 180)
        
        overlay.fill(bg_color)
        text_surface = config.SCORE_FONT.render(text, True, color)
        text_rect = text_surface.get_rect(center=(board_width / 2, board_width / 2))
        overlay.blit(text_surface, text_rect)
        surface.blit(overlay, (x_offset, y_offset))