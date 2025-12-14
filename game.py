import random
import pygame
import sys
import math
import argparse
import time, os
import copy
from enum import Enum
from typing import Tuple, Optional, Dict, Any, Callable


class Action(Enum):
    """Action space for the game"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

State: type = Dict[str, Any]

class Tile:
    """Represents a game tile"""
    def __init__(self, position, value):
        self.x = position['x']
        self.y = position['y']
        self.value = value
        self.merged_from = None
        self.previous_position = None
    
    def save_position(self):
        """Save current position"""
        self.previous_position = {'x': self.x, 'y': self.y}
    
    def update_position(self, position):
        """Update position"""
        self.x = position['x']
        self.y = position['y']
    
    def serialize(self):
        """Serialize tile information"""
        return {
            'position': {'x': self.x, 'y': self.y},
            'value': self.value
        }


class Grid:
    """Game grid class"""
    def __init__(self, size, previous_state=None):
        self.size = size
        self.cells = self._empty() if previous_state is None else self._from_state(previous_state)
    
    def _empty(self):
        """Create empty grid"""
        cells = []
        for x in range(self.size):
            cells.append([])
            for y in range(self.size):
                cells[x].append(None)
        return cells
    
    def _from_state(self, state):
        """Restore grid from state"""
        cells = self._empty()
        for x in range(self.size):
            for y in range(self.size):
                if state[x][y]:
                    cells[x][y] = Tile({'x': x, 'y': y}, state[x][y]['value'])
        return cells
    
    def random_available_cell(self):
        """Get random available cell"""
        cells = self.available_cells()
        if cells:
            return random.choice(cells)
        return None
    
    def available_cells(self):
        """Get all available cells"""
        cells = []
        for x in range(self.size):
            for y in range(self.size):
                if not self.cells[x][y]:
                    cells.append({'x': x, 'y': y})
        return cells
    
    def cells_available(self):
        """检查是否有可用单元格"""
        return bool(self.available_cells())
    
    def cell_available(self, cell):
        """检查特定单元格是否可用"""
        return not self.cell_occupied(cell)
    
    def cell_occupied(self, cell):
        """检查特定单元格是否被占用"""
        return self.cell_content(cell) is not None
    
    def cell_content(self, cell):
        """获取单元格内容"""
        if self.within_bounds(cell):
            return self.cells[cell['x']][cell['y']]
        return None
    
    def within_bounds(self, cell):
        """检查单元格是否在边界内"""
        return (0 <= cell['x'] < self.size and 
                0 <= cell['y'] < self.size)
    
    def insert_tile(self, tile):
        """插入方块"""
        self.cells[tile.x][tile.y] = tile
    
    def remove_tile(self, tile):
        """移除方块"""
        self.cells[tile.x][tile.y] = None
    
    def each_cell(self, callback):
        """遍历每个单元格"""
        for x in range(self.size):
            for y in range(self.size):
                callback(x, y, self.cells[x][y])
    
    def serialize(self):
        """序列化网格"""
        cell_state = []
        for x in range(self.size):
            cell_state.append([])
            for y in range(self.size):
                cell_state[x].append(
                    self.cells[x][y].serialize() if self.cells[x][y] else None
                )
        return {
            'size': self.size,
            'cells': cell_state
        }


def test_fibonacci(value):
    """Check if a value is a Fibonacci number"""
    if value <= 0:
        return False
    
    # Generate Fibonacci sequence until exceeding the given value
    fib = [1, 1]
    while value > fib[-1]:
        fib.append(fib[-1] + fib[-2])
    
    # Check if value is in Fibonacci sequence
    return value in fib


def can_merge_fibonacci(value1, value2):
    """Check if the sum of two values is a Fibonacci number"""
    return test_fibonacci(value1 + value2)


FIB_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]

class GameManager:
    """Game manager class"""
    def __init__(self, size=3):
        self.size = size
        self.start_tiles = 2
        self.goal = 1597  # 斐波那契数列中的一个较大数作为目标
        # self.goal = 21  # 斐波那契数列中的一个较大数作为目标
        # self.goal = 89  # 斐波那契数列中的一个较大数作为目标
        self.setup()
    
    def setup(self):
        """Setup game"""
        self.grid = Grid(self.size)
        self.score = 0
        self.over = False
        self.won = False
        self.keep_playing = False
        
        # 添加初始方块
        self.add_start_tiles()
    
    def restart(self):
        """Restart game"""
        self.setup()
    
    def keep_playing_mode(self):
        """继续游戏模式"""
        self.keep_playing = True
    
    def is_game_terminated(self):
        """检查游戏是否结束"""
        return self.over or (self.won and not self.keep_playing)
    
    def add_start_tiles(self):
        """添加初始方块"""
        for _ in range(self.start_tiles):
            self.add_random_tile()
    
    def add_random_tile(self):
        """在随机位置添加方块"""
        if self.grid.cells_available():
            # 90% 概率生成1，10% 概率生成2（斐波那契数列的开始）
            value = 1 if random.random() < 0.9 else 2
            tile = Tile(self.grid.random_available_cell(), value)
            self.grid.insert_tile(tile)
    
    def prepare_tiles(self):
        """准备方块移动"""
        def callback(x, y, tile):
            if tile:
                tile.merged_from = None
                tile.save_position()
        
        self.grid.each_cell(callback)
    
    def move_tile(self, tile, cell):
        """移动方块"""
        self.grid.cells[tile.x][tile.y] = None
        self.grid.cells[cell['x']][cell['y']] = tile
        tile.update_position(cell)
    
    def move(self, direction):
        """移动方块"""
        # 0: 上, 1: 右, 2: 下, 3: 左
        if self.is_game_terminated():
            return
        
        vector = self.get_vector(direction)
        traversals = self.build_traversals(vector)
        moved = False
        
        self.prepare_tiles()
        
        # 遍历网格并移动方块
        for x in traversals['x']:
            for y in traversals['y']:
                cell = {'x': x, 'y': y}
                tile = self.grid.cell_content(cell)
                
                if tile:
                    positions = self.find_farthest_position(cell, vector)
                    next_tile = self.grid.cell_content(positions['next'])
                    
                    # 检查是否可以合并（使用斐波那契规则）
                    if (next_tile and 
                        can_merge_fibonacci(tile.value, next_tile.value) and 
                        not next_tile.merged_from):
                        
                        merged_value = tile.value + next_tile.value
                        merged = Tile(positions['next'], merged_value)
                        merged.merged_from = [tile, next_tile]
                        
                        self.grid.insert_tile(merged)
                        self.grid.remove_tile(tile)
                        
                        tile.update_position(positions['next'])
                        
                        # 更新分数
                        self.score += merged_value * merged_value
                        
                        # 检查是否达到目标
                        if merged_value == self.goal:
                            self.won = True
                    else:
                        self.move_tile(tile, positions['farthest'])
                    
                    if not self.positions_equal(cell, {'x': tile.x, 'y': tile.y}):
                        moved = True
        
        if moved:
            self.add_random_tile()
            
            if not self.moves_available():
                self.over = True
    
    def get_vector(self, direction):
        """获取移动方向向量"""
        vectors = {
            0: {'x': 0, 'y': -1},   # 上
            1: {'x': 1, 'y': 0},    # 右
            2: {'x': 0, 'y': 1},    # 下
            3: {'x': -1, 'y': 0}    # 左
        }
        return vectors[direction]
    
    def build_traversals(self, vector):
        """构建遍历顺序"""
        traversals = {
            'x': list(range(self.size)),
            'y': list(range(self.size))
        }
        
        # 根据方向调整遍历顺序
        if vector['x'] == 1:
            traversals['x'] = traversals['x'][::-1]
        if vector['y'] == 1:
            traversals['y'] = traversals['y'][::-1]
        
        return traversals
    
    def find_farthest_position(self, cell, vector):
        """找到最远位置"""
        previous = cell
        
        while True:
            current = {
                'x': previous['x'] + vector['x'],
                'y': previous['y'] + vector['y']
            }
            
            if not self.grid.within_bounds(current) or not self.grid.cell_available(current):
                break
            
            previous = current
        
        return {
            'farthest': previous,
            'next': {
                'x': previous['x'] + vector['x'],
                'y': previous['y'] + vector['y']
            }
        }
    
    def moves_available(self):
        """检查是否有可用移动"""
        return self.grid.cells_available() or self.tile_matches_available()
    
    def tile_matches_available(self):
        """检查是否有可合并的方块"""
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content({'x': x, 'y': y})
                
                if tile:
                    for direction in range(4):
                        vector = self.get_vector(direction)
                        cell = {'x': x + vector['x'], 'y': y + vector['y']}
                        other = self.grid.cell_content(cell)
                        
                        if other and can_merge_fibonacci(tile.value, other.value):
                            return True
        
        return False
    
    def positions_equal(self, first, second):
        """检查两个位置是否相等"""
        return first['x'] == second['x'] and first['y'] == second['y']
    
    def serialize(self):
        """序列化游戏状态"""
        return {
            'grid': self.grid.serialize(),
            'score': self.score,
            'over': self.over,
            'won': self.won,
            'keep_playing': self.keep_playing
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current game state for AI"""
        # Convert grid to simple 2D array representation
        grid_state = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content({'x': x, 'y': y})
                if tile:
                    # grid_state[y][x] = tile.value
                    grid_state[y][x] = FIB_SEQUENCE.index(tile.value) + 1  # 使用斐波那契数列索引表示
        
        return {
            'grid': grid_state,
            'score': self.score,
            'over': self.over,
            'won': self.won
        }
    
    def step(self, action: Action) -> Tuple[Dict[str, Any], int, bool]:
        """Take a step in the environment"""
        if self.is_game_terminated():
            return self.get_state(), 0, True
        
        prev_score = self.score
        prev_state = copy.deepcopy(self.get_state())
        
        # Execute the action
        self.move(action.value)
        
        # Calculate reward
        reward = self.score - prev_score
        
        # Check if game state actually changed (valid move)
        new_state = self.get_state()
        if prev_state['grid'] == new_state['grid']:
            reward = -1  # Penalty for invalid move
        
        return new_state, reward, self.is_game_terminated()
    
    def reset(self) -> Dict[str, Any]:
        """Reset the game and return initial state"""
        self.setup()
        return self.get_state()
    
    def clone(self):
        """Create a deep copy of the current game state"""
        new_game = GameManager(self.size)
        new_game.grid = Grid(self.size)
        
        # Copy grid state
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content({'x': x, 'y': y})
                if tile:
                    new_tile = Tile({'x': x, 'y': y}, tile.value)
                    new_game.grid.insert_tile(new_tile)
        
        new_game.score = self.score
        new_game.over = self.over
        new_game.won = self.won
        new_game.keep_playing = self.keep_playing
        
        return new_game


# Pygame 常量
WINDOW_SIZE = 1000
GRID_SIZE = 3
CELL_SIZE = 200  # 固定大小确保可预测布局
CELL_MARGIN = 10
GRID_OFFSET_X = (WINDOW_SIZE - (3 * CELL_SIZE + 4 * CELL_MARGIN)) // 2
GRID_OFFSET_Y = 120

# 颜色定义
COLORS = {
    'background': (187, 173, 160),
    'grid_background': (205, 193, 180),
    'empty_cell': (238, 228, 218),
    'text_dark': (119, 110, 101),
    'text_light': (249, 246, 242),
    'header_bg': (143, 122, 102),
    'game_over_bg': (238, 228, 218, 200),
    'winner_bg': (237, 194, 46, 200),
    'tiles': {
        0: (238, 228, 218),     # Empty
        1: (255, 240, 245),     # 1 - Light pink
        2: (230, 240, 255),     # 2 - Light blue
        3: (240, 255, 230),     # 3 - Light green
        5: (255, 250, 205),     # 5 - Light yellow
        8: (255, 228, 196),     # 8 - Peach
        13: (221, 160, 221),    # 13 - Purple
        21: (255, 182, 193),    # 21 - Light pink red
        34: (173, 216, 230),    # 34 - Light blue
        55: (144, 238, 144),    # 55 - Light green
        89: (255, 215, 0),      # 89 - Gold
        144: (255, 140, 0),     # 144 - Dark orange
        233: (220, 20, 60),     # 233 - Deep red
        377: (138, 43, 226),    # 377 - Blue purple
        610: (255, 20, 147),    # 610 - Deep pink
        987: (50, 205, 50),     # 987 - Lime green
        1597: (255, 69, 0),     # 1597 - Red orange
    }
}


class PygameDisplay:
    """Pygame game display class"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Fibonacci 2048")
        
        # 字体设置
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        self.font_tile = pygame.font.Font(None, 144)
        
        # 创建半透明表面用于游戏结束覆盖层
        self.overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        self.overlay.set_alpha(200)
        
    def draw_rounded_rect(self, surface, color, rect, radius):
        """绘制圆角矩形"""
        if len(color) == 4:  # 包含alpha通道
            # 创建临时表面处理alpha
            temp_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.rect(temp_surface, color, (0, 0, rect.width, rect.height), border_radius=radius)
            surface.blit(temp_surface, rect.topleft)
        else:
            pygame.draw.rect(surface, color, rect, border_radius=radius)
    
    def get_tile_color(self, value):
        """获取方块颜色"""
        return COLORS['tiles'].get(value, COLORS['tiles'][144])
    
    def get_text_color(self, value):
        """获取文字颜色"""
        # 统一使用深色文字，在浅色背景上更清晰
        return COLORS['text_dark']
    
    def draw_tile(self, value, x, y):
        """绘制单个方块"""
        rect = pygame.Rect(
            GRID_OFFSET_X + x * (CELL_SIZE + CELL_MARGIN),
            GRID_OFFSET_Y + y * (CELL_SIZE + CELL_MARGIN),
            CELL_SIZE,
            CELL_SIZE
        )
        
        # 绘制方块背景
        color = self.get_tile_color(value)
        self.draw_rounded_rect(self.screen, color, rect, 6)
        
        # 绘制数字
        if value > 0:
            text_color = self.get_text_color(value)
            text_surface = self.font_tile.render(str(value), True, text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
    
    def draw_grid(self, game):
        """绘制游戏网格"""
        # 绘制网格背景
        grid_rect = pygame.Rect(
            GRID_OFFSET_X - CELL_MARGIN,
            GRID_OFFSET_Y - CELL_MARGIN,
            GRID_SIZE * (CELL_SIZE + CELL_MARGIN) + CELL_MARGIN,
            GRID_SIZE * (CELL_SIZE + CELL_MARGIN) + CELL_MARGIN
        )
        self.draw_rounded_rect(self.screen, COLORS['grid_background'], grid_rect, 10)
        
        # 绘制所有方块
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                tile = game.grid.cell_content({'x': x, 'y': y})
                value = tile.value if tile else 0
                self.draw_tile(value, x, y)
    
    def draw_header(self, game):
        """绘制游戏头部信息"""
        # 标题
        title_text = self.font_large.render("Fibonacci 2048", True, COLORS['text_dark'])
        title_rect = title_text.get_rect(center=(WINDOW_SIZE // 2, 40))
        self.screen.blit(title_text, title_rect)
        
        # 分数
        score_text = self.font_medium.render(f"Score: {game.score}", True, COLORS['text_dark'])
        score_rect = score_text.get_rect(topleft=(50, 80))
        self.screen.blit(score_text, score_rect)
        
        # 目标
        goal_text = self.font_medium.render(f"Goal: {game.goal}", True, COLORS['text_dark'])
        goal_rect = goal_text.get_rect(topright=(WINDOW_SIZE - 50, 80))
        self.screen.blit(goal_text, goal_rect)
    
    def draw_footer(self):
        """绘制游戏底部说明"""
        y_offset = GRID_OFFSET_Y + GRID_SIZE * (CELL_SIZE + CELL_MARGIN) + 20
        
        # 控制说明
        controls = [
            "Arrow Keys or WASD to Move",
            "R to Restart | ESC to Quit",
            "Fibonacci: 1,1,2,3,5,8,13,21,34,55,89..."
        ]
        
        for i, text in enumerate(controls):
            control_surface = self.font_small.render(text, True, COLORS['text_dark'])
            control_rect = control_surface.get_rect(center=(WINDOW_SIZE // 2, y_offset + i * 25))
            self.screen.blit(control_surface, control_rect)
    
    def draw_game_over(self, game):
        """绘制游戏结束界面"""
        if not (game.over or (game.won and not game.keep_playing)):
            return
        
        # 半透明覆盖层
        if game.won:
            self.overlay.fill(COLORS['winner_bg'][:3])
        else:
            self.overlay.fill(COLORS['game_over_bg'][:3])
        self.screen.blit(self.overlay, (0, 0))
        
        # 主要消息
        if game.won and not game.keep_playing:
            main_text = "You Win!"
            sub_text = "Press C to Continue"
        else:
            main_text = "Game Over!"
            sub_text = "Press R to Restart"
        
        # 绘制文字
        main_surface = self.font_large.render(main_text, True, COLORS['text_dark'])
        main_rect = main_surface.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 - 30))
        self.screen.blit(main_surface, main_rect)
        
        sub_surface = self.font_medium.render(sub_text, True, COLORS['text_dark'])
        sub_rect = sub_surface.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 20))
        self.screen.blit(sub_surface, sub_rect)
    
    def display_game(self, game):
        """显示游戏界面"""
        # 清屏
        self.screen.fill(COLORS['background'])
        
        # 绘制各个部分
        self.draw_header(game)
        self.draw_grid(game)
        self.draw_footer()
        self.draw_game_over(game)
        
        # 更新显示
        pygame.display.flip()
    
    def handle_events(self):
        """处理pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    return 'up'
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    return 'down'
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    return 'left'
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    return 'right'
                elif event.key == pygame.K_r:
                    return 'restart'
                elif event.key == pygame.K_c:
                    return 'continue'
                elif event.key == pygame.K_ESCAPE:
                    return 'quit'
        return None


def main():
    """Main function"""
    game = GameManager()
    display = PygameDisplay()
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # 处理事件
        action = display.handle_events()
        
        if action == 'quit':
            running = False
        elif action == 'restart':
            game.restart()
        elif action == 'continue' and game.won:
            game.keep_playing_mode()
        elif not game.is_game_terminated():
            # 游戏进行中的移动操作
            if action == 'up':
                game.move(0)
            elif action == 'right':
                game.move(1)
            elif action == 'down':
                game.move(2)
            elif action == 'left':
                game.move(3)
        
        # 绘制游戏界面
        display.display_game(game)
        
        # 控制帧率
        clock.tick(60)
    
    pygame.quit()
    sys.exit()


def rollout(policy_fn: Callable[[Dict[str, Any]], Action], 
           initial_state: Optional[Dict[str, Any]] = None,
           render: bool = False,
           render_delay: float = 0.5) -> int:
    """Run a complete game using the given policy"""
    game = GameManager()
    
    if initial_state is not None:
        # TODO: Implement state restoration if needed
        pass
    
    if render:
        display = PygameDisplay()
        clock = pygame.time.Clock()
        pygame.event.set_allowed([pygame.QUIT])
    
    step_count = 0
    while not game.is_game_terminated():
        state = game.get_state()
        action = policy_fn(state)
        
        _, reward, done = game.step(action)
        print('reward:', reward, 'done:', done)
        step_count += 1
        
        if render:
            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return game.score
            
            display.display_game(game)
            time.sleep(render_delay)
    
    if render:
        # Show final state for a moment
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return game.score
        display.display_game(game)
        time.sleep(2)
        pygame.quit()
    
    print(f"Game finished after {step_count} steps with score: {game.score}")
    return game.score



def demonstrate_step_evolution():
    """Demonstrate state-action evolution"""
    print("\n=== State-Action Evolution Demo ===")
    game = GameManager()
    
    for i in range(5):
        state = game.get_state()
        action = Action.LEFT
        
        print(f"\nStep {i+1}:")
        print(f"Action: {action.name}")
        print(f"Grid before:")
        for row in state['grid']:
            print(f"  {row}")
        print(f"Score before: {state['score']}")
        
        new_state, reward, done = game.step(action)
        
        print(f"Grid after:")
        for row in new_state['grid']:
            print(f"  {row}")
        print(f"Score after: {new_state['score']}")
        print(f"Reward: {reward}")
        print(f"Terminated: {done}")
        
        if done:
            print("Game terminated!")
            break


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Fibonacci 2048 Game')
    parser.add_argument('--mode', choices=['human', 'demo'], 
                       default='human', help='Game mode')
    parser.add_argument('--render', action='store_true', 
                       help='Render the game during AI play')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between moves in rendered mode')
    
    args = parser.parse_args()
    
    if args.mode == 'human':
        # Original human gameplay
        game = GameManager()
        display = PygameDisplay()
        clock = pygame.time.Clock()
        
        running = True
        while running:
            action = display.handle_events()
            
            if action == 'quit':
                running = False
            elif action == 'restart':
                game.restart()
            elif action == 'continue' and game.won:
                game.keep_playing_mode()
            elif not game.is_game_terminated():
                if action == 'up':
                    game.move(0)
                elif action == 'right':
                    game.move(1)
                elif action == 'down':
                    game.move(2)
                elif action == 'left':
                    game.move(3)
            
            display.display_game(game)
            clock.tick(60)
        
        pygame.quit()
        sys.exit()
    
    elif args.mode == 'demo':
        demonstrate_step_evolution()

if __name__ == "__main__":
    main()