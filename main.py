import pygame
import sys
import numpy as np
import config
from game.game_logic import Game2048
from ai.expectimax_ai import ExpectimaxAI
from ai.mcts_ai import MCTSAI
from ui.renderer import GameRenderer, BoardRenderer, Tile
import threading
import queue
import logging
import traceback

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.FATAL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("2048-AI-BATTLE")

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("2048 AI Battle - with Animation")
    clock = pygame.time.Clock()

    main_renderer = GameRenderer(screen)
    ais = {
        "Expectimax": ExpectimaxAI(depth=config.EXPECTIMAX_DEPTH),
        "MCTS": MCTSAI(iterations=config.MCTS_ITERATIONS)
    }

    # 각 AI마다 독립적인 게임, 보드 렌더러, 큐, 스레드, 상태 저장
    games = {name: Game2048() for name in ais.keys()}
    brs = {name: BoardRenderer() for name in ais.keys()}
    for name, g in games.items():
        brs[name].set_board(g.board)

    analyses = {name: {} for name in ais.keys()}
    turns = {name: 0 for name in ais.keys()}
    last_trees = {name: None for name in ais.keys()}

    # 스레드 안전한 큐와 스레드 참조
    ai_queues = {name: queue.Queue(maxsize=1) for name in ais.keys()}
    ai_threads = {name: None for name in ais.keys()}
    states = {name: "PLAYING" for name in ais.keys()}  # "PLAYING", "THINKING", "ANIMATING", "POST_GAME"

    interaction_data = {}

    logger.info("시작: AIs=%s", list(ais.keys()))

    while True:
        dt = clock.tick(config.FPS) / 1000.0
        mouse_pos = pygame.mouse.get_pos()
        interaction_data['mouse_pos'] = mouse_pos
        interaction_data['click'] = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                logger.info("종료 신호 수신 - 종료합니다.")
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP:
                interaction_data['click'] = event.pos

        # 각 AI별 독립 루프 처리
        for name, ai in ais.items():
            g = games[name]
            br = brs[name]
            s = states[name]
            q = ai_queues[name]
            t = ai_threads[name]

            if s == "PLAYING":
                if g.game_over:
                    logger.info("[%s] 게임 오버 - 리셋", name)
                    g.reset()
                    br.set_board(g.board)
                    turns[name] = 0
                    analyses[name] = {}
                    last_trees[name] = None
                    states[name] = "PLAYING"
                    continue

                if not g.win:
                    # 스레드가 현재 없거나 종료된 경우 새로운 think 스레드 시작
                    if t is None or not t.is_alive():
                        # 안전한 게임 복제
                        g_clone = g.clone()

                        def think_task(out_q: queue.Queue, ai_local, g_clone_local):
                            try:
                                # 큐를 깨끗하게 비움(드문 경우지만 안전하게)
                                try:
                                    while not out_q.empty():
                                        out_q.get_nowait()
                                except queue.Empty:
                                    pass

                                logger.debug("[%s] think_task 시작", name)
                                res = ai_local.get_move(g_clone_local)  # (move, analysis, tree)
                                # 결과 put (blocking 가능성 거의 없음, maxsize=1)
                                out_q.put(res)
                                logger.debug("[%s] think_task 결과 put 완료", name)
                            except Exception as e:
                                # 예외 발생 시 traceback을 큐에 넣고 로깅
                                tb = traceback.format_exc()
                                logger.exception("[%s] think_task 예외 발생: %s", name, e)
                                try:
                                    out_q.put(("__EXCEPTION__", tb))
                                except Exception:
                                    logger.error("[%s] 큐에 예외 정보도 넣을 수 없습니다.", name)

                        logger.debug("[%s] THINKING으로 전환 - 스레드 시작", name)
                        th = threading.Thread(target=think_task, args=(q, ai, g_clone), daemon=True)
                        ai_threads[name] = th
                        th.start()
                        states[name] = "THINKING"

                else:
                    # 이미 승리한 경우
                    states[name] = "POST_GAME"
                    if last_trees[name]:
                        last_trees[name]['expanded'] = True
                        logger.debug("[%s] 승리 상태, last_trees expanded 플래그 설정", name)

            elif s == "THINKING":
                # 스레드가 종료되었는지 확인
                t = ai_threads[name]
                if t is None:
                    logger.warning("[%s] THINKING 상태인데 스레드 참조가 없습니다. PLAYING으로 복귀", name)
                    states[name] = "PLAYING"
                    continue

                if not t.is_alive():
                    # 스레드가 이미 종료됨 — 결과 큐에서 꺼내기 시도
                    try:
                        res = q.get_nowait()
                    except queue.Empty:
                        logger.warning("[%s] 스레드 종료되었지만 큐에 결과 없음. PLAYING으로 복귀", name)
                        ai_threads[name] = None
                        states[name] = "PLAYING"
                        continue

                    ai_threads[name] = None  # 스레드 참조 정리

                    if isinstance(res, tuple) and res and res[0] == "__EXCEPTION__":
                        tb = res[1] if len(res) > 1 else "no traceback"
                        logger.error("[%s] AI 내부 예외 발생:\n%s", name, tb)
                        states[name] = "PLAYING"
                        continue

                    if res is None:
                        logger.error("[%s] AI가 None 결과를 반환했습니다. PLAYING으로 복귀", name)
                        states[name] = "PLAYING"
                        continue

                    # 정상적인 (move, analysis, tree) 결과 처리
                    try:
                        move, analysis, tree = res
                        logger.info("[%s] 스레드 완료 - move=%s", name, move)
                    except Exception as e:
                        logger.exception("[%s] 결과 파싱 실패: %s", name, e)
                        states[name] = "PLAYING"
                        continue

                    old_b = np.copy(g.board)
                    board_changed = g.move(move)

                    if board_changed:
                        analyses[name] = analysis
                        last_trees[name] = tree
                        logger.debug("[%s] 보드 변경 발생 - 애니메이션 시작", name)
                        br.start_animation(old_b, g.board, move, config.ANIMATION_DURATION / 1000.0)
                        turns[name] += 1
                        states[name] = "ANIMATING"
                        logger.info("[%s] ANIMATING으로 전환", name)
                    else:
                        logger.warning("[%s] AI가 유효하지 않은 move를 선택함: %s", name, move)
                        states[name] = "PLAYING"

            elif s == "ANIMATING":
                br.update(dt)
                if not br.is_animating:
                    br.set_board(g.board)
                    states[name] = "PLAYING"
                    logger.debug("[%s] 애니메이션 종료 - PLAYING으로 전환", name)

            elif s == "POST_GAME":
                # 포스트게임 처리가 필요하면 여기에 추가
                pass

        # 렌더링: 여러 AI가 THINKING이면 첫번째 하나만 전달 (draw_main_ui의 요구사항에 따라 조정 가능)
        thinking_list = [n for n, st in states.items() if st == "THINKING"]
        thinking_param = thinking_list[0] if thinking_list else None

        main_renderer.draw_main_ui(games, analyses, turns, brs, thinking_param)
        pygame.display.flip()

if __name__ == '__main__':
    main()
