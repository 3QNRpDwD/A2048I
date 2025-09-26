import pygame
import sys
import numpy as np
import config
import threading
import multiprocessing
import queue
import logging
import traceback
import random
import time

from game.game_logic import Game2048
from ai.expectimax_ai import ExpectimaxAI
from ai.mcts_ai import MCTSAI
from ai.hybrid_mcts_ai import HybridMCTSAI
from ai.evo_expectimax_ai import EvoExpectimaxAI
from ai.n_tuple_ai import NTupleAI
from ui.renderer import GameRenderer, BoardRenderer

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("2048-AI-BATTLE")

# --- 유전 알고리즘 온라인 학습 프로세스 ---
class GeneticAlgorithmProcess(multiprocessing.Process):
    def __init__(self, update_queue, initial_weights):
        super().__init__(daemon=True)
        self.name = "GA-Trainer-Process"
        self.update_queue = update_queue
        self.population_size = 20
        self.mutation_rate = 0.1
        self.mutation_strength = 0.2
        self.crossover_rate = 0.7
        self.fitness_games = 2
        self.weight_ranges = {
            "monotonicity": (0, 10),
            "smoothness": (0, 5),
            "empty_tiles": (0, 5),
            "max_value": (0, 1),
        }
        self.population = [self._create_individual() for _ in range(self.population_size)]
        self.population[0] = initial_weights
        self.generation = 0
        self.best_individual = initial_weights
        self.best_fitness = -1

    def _create_individual(self):
        return {key: random.uniform(low, high) for key, (low, high) in self.weight_ranges.items()}

    def _calculate_fitness(self, weights):
        ai = ExpectimaxAI(depth=2)
        ai.heuristic_weights = weights
        total_score, total_max_tile = 0, 0
        for _ in range(self.fitness_games):
            game = Game2048()
            while not game.game_over:
                move, _, _ = ai.get_move(game.clone())
                if move is None: break
                game.move(move)
            total_score += game.score
            total_max_tile += np.max(game.board)
        return (total_score / self.fitness_games) + total_max_tile

    def _selection(self, fitnesses):
        tournament_size = 3
        selected = []
        for _ in range(self.population_size):
            participants_indices = random.sample(range(self.population_size), tournament_size)
            winner_index = max(participants_indices, key=lambda i: fitnesses[i])
            selected.append(self.population[winner_index])
        return selected

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate: return parent1.copy(), parent2.copy()
        child1, child2 = {}, {}
        for key in parent1.keys():
            alpha = random.random()
            child1[key] = parent1[key] * alpha + parent2[key] * (1 - alpha)
            child2[key] = parent2[key] * alpha + parent1[key] * (1 - alpha)
        return child1, child2

    def _mutate(self, individual):
        for key in individual.keys():
            if random.random() < self.mutation_rate:
                change = random.uniform(-self.mutation_strength, self.mutation_strength)
                new_value = individual[key] + change
                low, high = self.weight_ranges[key]
                individual[key] = max(low, min(new_value, high))
        return individual

    def run(self):
        print(f"[{self.name}] 온라인 학습 프로세스를 시작합니다.")
        while True:
            self.generation += 1
            print(f"[{self.name}] 세대 {self.generation} 학습 시작...")
            fitnesses = [self._calculate_fitness(ind) for ind in self.population]
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_individual = self.population[max_fitness_idx].copy()
                print(f"[{self.name}] 새로운 최적 가중치 발견! 적합도: {self.best_fitness:.2f}")
                self.update_queue.put({
                    "generation": self.generation,
                    "best_fitness": self.best_fitness,
                    "best_weights": self.best_individual
                })
            parents = self._selection(fitnesses)
            next_population = [self.best_individual.copy()]
            while len(next_population) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = self._crossover(p1, p2)
                next_population.append(self._mutate(c1))
                if len(next_population) < self.population_size:
                    next_population.append(self._mutate(c2))
            self.population = next_population
            time.sleep(1)

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("2048 AI Battle - Online Learning (Multiprocess)")
    clock = pygame.time.Clock()
    main_renderer = GameRenderer(screen)
    
    ais = {
        "Expectimax": ExpectimaxAI(depth=config.EXPECTIMAX_DEPTH),
        "Evo-Expectimax": EvoExpectimaxAI(depth=config.EXPECTIMAX_DEPTH),
        "MCTS": MCTSAI(iterations=config.MCTS_ITERATIONS),
        "Hybrid MCTS": HybridMCTSAI(iterations=config.MCTS_ITERATIONS),
        "Online N-Tuple": NTupleAI(),
    }
    evo_ai_name = "Evo-Expectimax"
    evo_ai_lock = threading.Lock()

    games = {name: Game2048() for name in ais.keys()}
    brs = {name: BoardRenderer() for name in ais.keys()}
    for name, g in games.items():
        brs[name].set_board(g.board)

    analyses = {name: {} for name in ais.keys()}
    turns = {name: 0 for name in ais.keys()}
    last_trees = {name: None for name in ais.keys()}
    retry_counts = {name: 0 for name in ais.keys()}
    highest_tiles = {name: 0 for name in ais.keys()}

    ai_queues = {name: queue.Queue(maxsize=1) for name in ais.keys()}
    ai_threads = {name: None for name in ais.keys()}
    states = {name: "PLAYING" for name in ais.keys()}

    ga_update_queue = multiprocessing.Queue()
    ga_status = {"generation": 0, "best_fitness": 0}
    initial_weights = ais[evo_ai_name].heuristic_weights.copy()
    ga_process = GeneticAlgorithmProcess(ga_update_queue, initial_weights)
    ga_process.start()

    logger.info("시작: AIs=%s", list(ais.keys()))

    while True:
        dt = clock.tick(config.FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                logger.info("종료 신호 수신 - 종료합니다.")
                ga_process.terminate()
                ga_process.join()
                # NTupleAI 가중치 저장
                for ai in ais.values():
                    if isinstance(ai, NTupleAI):
                        ai.save()
                pygame.quit()
                sys.exit()

        try:
            update = ga_update_queue.get_nowait()
            ga_status = {"generation": update["generation"], "best_fitness": update["best_fitness"]}
            with evo_ai_lock:
                ais[evo_ai_name].heuristic_weights = update["best_weights"]
            logger.info(f"[Main] {evo_ai_name} 가중치가 Gen {update['generation']} 최적 가중치로 업데이트됨.")
        except queue.Empty:
            pass

        for name, ai in ais.items():
            g = games[name]
            s = states[name]
            q = ai_queues[name]
            t = ai_threads[name]

            if s == "PLAYING" and not g.win:
                if g.game_over:
                    logger.info(f"[{name}] 게임 오버 - 리셋")
                    if isinstance(ai, NTupleAI): ai.save() # 게임 오버 시 가중치 저장
                    retry_counts[name] += 1
                    g.reset()
                    brs[name].set_board(g.board)
                    turns[name] = 0
                    states[name] = "PLAYING"
                    continue

                if t is None or not t.is_alive():
                    g_clone = g.clone()
                    def think_task(out_q, ai_local, g_clone_local, name_local, lock_local):
                        try:
                            res = None
                            if name_local == evo_ai_name and lock_local:
                                with lock_local:
                                    res = ai_local.get_move(g_clone_local)
                            else:
                                res = ai_local.get_move(g_clone_local)
                            out_q.put(res)
                        except Exception as e:
                            tb = traceback.format_exc()
                            logger.exception(f"[{name_local}] think_task 예외: {e}")
                            out_q.put(("__EXCEPTION__", tb))

                    current_lock = evo_ai_lock if name == evo_ai_name else None
                    th = threading.Thread(target=think_task, args=(q, ai, g_clone, name, current_lock), daemon=True)
                    ai_threads[name] = th
                    th.start()
                    states[name] = "THINKING"
            
            elif s == "THINKING" and t and not t.is_alive():
                try:
                    res = q.get_nowait()
                except queue.Empty:
                    states[name] = "PLAYING"
                    continue
                
                ai_threads[name] = None
                if isinstance(res, tuple) and res[0] == "__EXCEPTION__":
                    logger.error(f"[{name}] AI 예외 발생: {res[1]}")
                    states[name] = "PLAYING"
                    continue
                
                if res is None: 
                    states[name] = "PLAYING"
                    continue

                move, analysis, tree = res
                old_b = np.copy(g.board)
                board_changed, move_score = g.move(move)

                if board_changed:
                    # N-Tuple AI 온라인 학습 로직
                    if isinstance(ai, NTupleAI):
                        ai.perform_update(move_score, g.board)

                    current_max_tile = int(np.max(g.board))
                    if current_max_tile > highest_tiles.get(name, 0):
                        highest_tiles[name] = current_max_tile
                    analyses[name] = analysis
                    last_trees[name] = tree
                    brs[name].start_animation(old_b, g.board, move, config.ANIMATION_DURATION / 1000.0)
                    turns[name] = turns.get(name, 0) + 1
                    states[name] = "ANIMATING"
                else:
                    states[name] = "PLAYING"

            elif s == "ANIMATING":
                brs[name].update(dt)
                if not brs[name].is_animating:
                    brs[name].set_board(g.board)
                    states[name] = "PLAYING"

        thinking_list = [n for n, st in states.items() if st == "THINKING"]
        thinking_param = thinking_list[0] if thinking_list else None
        
        main_renderer.draw_main_ui(games, analyses, turns, brs, retry_counts, highest_tiles, 
                                     thinking_param, "PLAYING", None, last_trees, ais, ga_status)
        pygame.display.flip()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
