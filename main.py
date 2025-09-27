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
from ai.reinforce_ai import ReinforceAI
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

# main.py (수정본: 프로세스 + 내부 스레드 병렬 실행)
import pygame
import sys
import numpy as np
import config
import multiprocessing
import queue
import logging
import traceback
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from game.game_logic import Game2048
from ai.expectimax_ai import ExpectimaxAI
from ai.mcts_ai import MCTSAI
from ai.hybrid_mcts_ai import HybridMCTSAI
from ai.evo_expectimax_ai import EvoExpectimaxAI
from ai.n_tuple_ai import NTupleAI
from ai.reinforce_ai import ReinforceAI
from ui.renderer import GameRenderer, BoardRenderer

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("2048-AI-BATTLE")

# --- 유전 알고리즘 온라인 학습 프로세스 ---
# (기존 코드 동일)
# ... [생략: 기존 GeneticAlgorithmProcess 정의 그대로 유지] ...


# --- 공통 워커 베이스 ---
def multi_ai_worker(task_q, result_q, control_q, ai_factories, updateable_ai_names=None):
    """
    워커: 여러 AI를 항상 병렬 실행.
    - task_q: 메인 프로세스에서 받은 게임 상태
    - result_q: move 결과 반환
    - control_q: 가중치 업데이트 등 제어 메시지
    - ai_factories: {name: factory()} AI 생성자
    - updateable_ai_names: 가중치 업데이트가 적용될 AI 이름 집합
    """
    try:
        local_ais = {name: factory() for name, factory in ai_factories.items()}
        lock = threading.Lock()
        executor = ThreadPoolExecutor(max_workers=len(local_ais))

        def run_ai(ai_name, task):
            game = Game2048()
            game.board = np.array(task["board"])
            game.score = task["score"]
            game.game_over = task["game_over"]

            start = time.time()
            try:
                ai_local = local_ais[ai_name]
                move_res = ai_local.get_move(game)
                move = move_res[0] if move_res else None
                think_time = time.time() - start
                result_q.put({
                    "task_id": task["task_id"],
                    "ai_name": ai_name,
                    "move": move,
                    "think_time": think_time,
                    "status": "OK"
                })
            except Exception as e:
                result_q.put({
                    "task_id": task["task_id"],
                    "ai_name": ai_name,
                    "move": None,
                    "think_time": time.time() - start,
                    "status": "EXCEPTION",
                    "trace": traceback.format_exc()
                })

        logger.info(f"[Worker] 시작 - AIs={list(local_ais.keys())}")

        while True:
            # 컨트롤 메시지 처리
            try:
                while True:
                    ctrl = control_q.get_nowait()
                    if ctrl.get("cmd") == "update_weights" and updateable_ai_names:
                        with lock:
                            for ai_name in updateable_ai_names:
                                if ai_name in local_ais:
                                    local_ais[ai_name].heuristic_weights = ctrl["weights"]
                                    logger.info(f"[Worker] {ai_name} 가중치 업데이트 적용")
            except queue.Empty:
                pass

            task = task_q.get()
            if task is None:
                break

            # 받은 task를 모든 AI에게 동시에 실행
            for ai_name in local_ais.keys():
                executor.submit(run_ai, ai_name, task)

    except Exception:
        logger.exception("[Worker] 치명적 예외로 종료합니다.")


# --- AI 그룹별 워커 래퍼 ---
def expectimax_worker(task_q, result_q, control_q):
    ai_factories = {
        "Expectimax": lambda: ExpectimaxAI(depth=config.EXPECTIMAX_DEPTH),
        "Evo-Expectimax": lambda: EvoExpectimaxAI(depth=config.EXPECTIMAX_DEPTH),
    }
    updateable = {"Evo-Expectimax"}
    multi_ai_worker(task_q, result_q, control_q, ai_factories, updateable_ai_names=updateable)


def mcts_worker(task_q, result_q, control_q):
    ai_factories = {
        "MCTS": lambda: MCTSAI(iterations=config.MCTS_ITERATIONS),
        "Hybrid MCTS": lambda: HybridMCTSAI(iterations=config.MCTS_ITERATIONS),
    }
    multi_ai_worker(task_q, result_q, control_q, ai_factories)


def rl_worker(task_q, result_q, control_q):
    ai_factories = {
        "REINFORCE": lambda: ReinforceAI(),
        "Online N-Tuple": lambda: NTupleAI(),
    }
    multi_ai_worker(task_q, result_q, control_q, ai_factories)


# --- 그룹 매핑 ---
def ai_group_of(name):
    if name in ("Expectimax", "Evo-Expectimax"):
        return "expectimax"
    if name in ("MCTS", "Hybrid MCTS"):
        return "mcts"
    return "rl"


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("2048 AI Battle - Multiprocess + Threaded Workers")
    clock = pygame.time.Clock()
    main_renderer = GameRenderer(screen)

    # 메인 프로세스에서 AI 인스턴스 유지 (학습용)
    ais = {
        "Expectimax": ExpectimaxAI(depth=config.EXPECTIMAX_DEPTH),
        "Evo-Expectimax": EvoExpectimaxAI(depth=config.EXPECTIMAX_DEPTH),
        "MCTS": MCTSAI(iterations=config.MCTS_ITERATIONS),
        "Hybrid MCTS": HybridMCTSAI(iterations=config.MCTS_ITERATIONS),
        "Online N-Tuple": NTupleAI(),
        "REINFORCE": ReinforceAI(),
    }
    evo_ai_name = "Evo-Expectimax"

    games = {name: Game2048() for name in ais.keys()}
    brs = {name: BoardRenderer() for name in ais.keys()}
    for name, g in games.items():
        brs[name].set_board(g.board)

    analyses = {name: {} for name in ais.keys()}
    turns = {name: 0 for name in ais.keys()}
    retry_counts = {name: 0 for name in ais.keys()}
    highest_tiles = {name: 0 for name in ais.keys()}
    states = {name: "PLAYING" for name in ais.keys()}
    thinking_since = {name: 0 for name in ais.keys()}

    # --- 워커 IPC ---
    exp_task_q, exp_res_q, exp_ctrl_q = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()
    mcts_task_q, mcts_res_q, mcts_ctrl_q = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()
    rl_task_q, rl_res_q, rl_ctrl_q = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()

    exp_proc = multiprocessing.Process(target=expectimax_worker, args=(exp_task_q, exp_res_q, exp_ctrl_q), daemon=True)
    mcts_proc = multiprocessing.Process(target=mcts_worker, args=(mcts_task_q, mcts_res_q, mcts_ctrl_q), daemon=True)
    rl_proc = multiprocessing.Process(target=rl_worker, args=(rl_task_q, rl_res_q, rl_ctrl_q), daemon=True)

    exp_proc.start(); mcts_proc.start(); rl_proc.start()

    # --- GA 프로세스 ---
    ga_update_queue = multiprocessing.Queue()
    ga_status = {"generation": 0, "best_fitness": 0}
    initial_weights = ais[evo_ai_name].heuristic_weights.copy()
    ga_process = GeneticAlgorithmProcess(ga_update_queue, initial_weights)
    ga_process.start()

    logger.info("시작: AIs=%s", list(ais.keys()))

    next_task_id = 1
    pending_tasks = {}

    try:
        while True:
            dt = clock.tick(config.FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    exp_task_q.put(None); mcts_task_q.put(None); rl_task_q.put(None)
                    ga_process.terminate()
                    exp_proc.terminate(); mcts_proc.terminate(); rl_proc.terminate()
                    for ai in ais.values():
                        if isinstance(ai, NTupleAI): ai.save()
                    pygame.quit(); sys.exit()

            # GA 업데이트 처리
            try:
                update = ga_update_queue.get_nowait()
                ga_status = {"generation": update["generation"], "best_fitness": update["best_fitness"]}
                ais[evo_ai_name].heuristic_weights = update["best_weights"]
                exp_ctrl_q.put({"cmd": "update_weights", "weights": update["best_weights"]})
            except queue.Empty:
                pass

            # AI 작업 제출
            for name, ai in ais.items():
                g = games[name]
                if states[name] == "PLAYING" and not g.win:
                    if g.game_over:
                        if isinstance(ai, NTupleAI): ai.save()
                        if isinstance(ai, ReinforceAI): ai.post_move_update(0, g.game_over)
                        retry_counts[name] += 1
                        g.reset(); brs[name].set_board(g.board)
                        turns[name] = 0; states[name] = "PLAYING"
                        continue
                    task_id = next_task_id; next_task_id += 1
                    task = {
                        "task_id": task_id,
                        "board": g.board.tolist(),
                        "score": int(g.score),
                        "game_over": bool(g.game_over)
                    }
                    group = ai_group_of(name)
                    if group == "expectimax":
                        exp_task_q.put(task)
                    elif group == "mcts":
                        mcts_task_q.put(task)
                    else:
                        rl_task_q.put(task)
                    pending_tasks[task_id] = name
                    states[name] = "THINKING"; thinking_since[name] = time.time()

            # 결과 처리 (expectimax + mcts + rl)
            for res_q in (exp_res_q, mcts_res_q, rl_res_q):
                try:
                    while True:
                        res = res_q.get_nowait()
                        ai_name = res["ai_name"]
                        task_id = res["task_id"]
                        expected_name = pending_tasks.pop(task_id, ai_name)
                        g = games[expected_name]; ai_main = ais[expected_name]
                        states[expected_name] = "PLAYING"
                        if res["status"] != "OK" or res["move"] is None:
                            continue
                        move = res["move"]
                        board_changed = g.move(move)
                        if board_changed:
                            move_score = g.score  # 점수 갱신
                            if isinstance(ai_main, NTupleAI):
                                ai_main.perform_update(move_score, g.board)
                            if isinstance(ai_main, ReinforceAI):
                                ai_main.post_move_update(move_score, g.game_over)
                            current_max_tile = int(np.max(g.board))
                            if current_max_tile > highest_tiles.get(expected_name, 0):
                                highest_tiles[expected_name] = current_max_tile
                            analyses[expected_name] = {"think_time": res["think_time"]}
                            brs[expected_name].start_animation(g.board)
                            turns[expected_name] += 1
                            states[expected_name] = "ANIMATING"
                        else:
                            if isinstance(ai_main, ReinforceAI):
                                ai_main.post_move_update(0, g.game_over)
                            states[expected_name] = "PLAYING"

                except queue.Empty:
                    pass

            # 애니메이션 업데이트
            for name, s in states.items():
                if s == "ANIMATING":
                    brs[name].update(dt)
                    if not brs[name].is_animating:
                        brs[name].set_board(games[name].board)
                        states[name] = "PLAYING"

            thinking_list = [n for n, st in states.items() if st == "THINKING"]
            thinking_param = thinking_list[0] if thinking_list else None

            main_renderer.draw_main_ui(
                games, analyses, turns, brs, retry_counts, highest_tiles,
                thinking_param, "PLAYING", None, None, ais, ga_status
            )
            pygame.display.flip()

    except KeyboardInterrupt:
        exp_task_q.put(None); mcts_task_q.put(None); rl_task_q.put(None)
        ga_process.terminate(); exp_proc.terminate(); mcts_proc.terminate(); rl_proc.terminate()
        pygame.quit(); sys.exit()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
