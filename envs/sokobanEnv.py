import numpy as np
import random
import re
import copy
import secrets
import os
from typing import Optional, Dict, Tuple, List
from gymnasium.spaces import Discrete, Box
from gymnasium.core import RenderFrame
from typing import Any
from numba import njit
# symbol to int
WALL = 0
FLOOR = 1
TARGET = 2
BOX_ON_TARGET = 3
BOX = 4
PLAYER = 5
PLAYER_ON_TARGET = 6


MODE_STANDARD = 0 # 4 actions: Up, Down, Left, Right 
MODE_EXPLICIT = 1 # 8 actions: 0-3 Move, 4-7 Push 

CHANGE_COORDINATES = { # (row_change, col_change)
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

ROOM_STATE_TO_CHAR = {
    0: '#', 1: ' ', 2: '?', 3: '*', 4: '$', 5: '@', 6: '+'
}

def str_board_to_int(str_board):
    char_map = {'#': WALL, ' ': FLOOR, '?': TARGET, '*': BOX_ON_TARGET, '$': BOX, '@': PLAYER, '+': PLAYER_ON_TARGET}
    h = len(str_board)
    w = len(str_board[0])
    arr = np.zeros((h, w), dtype=np.int8)
    for r in range(h):
        for c in range(w):
            arr[r, c] = char_map.get(str_board[r][c], FLOOR)
    return arr

@njit(fastmath=True)
def _numba_is_solved(board):
    h, w = board.shape
    for r in range(h):
        for c in range(w):
            if board[r, c] == BOX: 
                return False
    return True

@njit(fastmath=True)
def _numba_get_player_pos(board):
    h, w = board.shape
    for r in range(h):
        for c in range(w):
            val = board[r, c]
            if val == PLAYER or val == PLAYER_ON_TARGET:
                return r, c
    return -1, -1

@njit(fastmath=True)
def _numba_step_with_reward(board, action, action_mode):
    """
    Perform one environment step.

    action_mode:
      0 (Standard): actions 0-3, automatically push when hitting a box.
      1 (Explicit):
         0-3 (Move): only move; hitting a box is treated as illegal.
         4-7 (Push): only push a box; stepping into empty space is treated as illegal.

    Reward (same convention as `SokobanVariantEnv._calc_reward`):
        penalty_for_step            = -0.1
        penalty_for_illegal_action  = -2.0
        penalty_box_off_target      = -1.0
        reward_box_on_target        = 1.0
        reward_finished             = 10.0
    """
    new_board = board.copy()
    h, w = new_board.shape

    penalty_for_step = -0.1
    penalty_for_illegal_action = -2.0
    penalty_box_off_target = -1.0
    reward_box_on_target = 1.0
    reward_finished = 10.0

    # Count box statistics in the previous state (used for reward calculation)
    prev_boxes_on_target = 0
    prev_num_boxes = 0
    for r in range(h):
        for c in range(w):
            v = board[r, c]
            if v == BOX_ON_TARGET:
                prev_boxes_on_target += 1
            if v == BOX or v == BOX_ON_TARGET:
                prev_num_boxes += 1
    
    pr, pc = _numba_get_player_pos(new_board)
    if pr == -1: return new_board, False, False, True, -0.1
    
    # Parse basic direction (0:Up, 1:Down, 2:Left, 3:Right)
    direction = action % 4
    
    # Whether this is an explicit push action (only meaningful when action_mode == MODE_EXPLICIT)
    is_explicit_push = False
    if action_mode == MODE_EXPLICIT and action >= 4:
        is_explicit_push = True

    # Compute delta movement
    dr, dc = 0, 0
    if direction == 0: dr = -1
    elif direction == 1: dr = 1
    elif direction == 2: dc = -1
    elif direction == 3: dc = 1
    
    nr, nc = pr + dr, pc + dc
    
    if nr < 0 or nr >= h or nc < 0 or nc >= w:
        return new_board, False, False, False, penalty_for_step
    
    target_val = new_board[nr, nc]
    
    # 1. Wall collision
    if target_val == WALL:
        return new_board, False, False, False, penalty_for_step
        
    # 2. Move into floor or target (Move logic)
    if target_val == FLOOR or target_val == TARGET:
        # [Mode 1 Check]: explicit Push action into empty cell is illegal
        if action_mode == MODE_EXPLICIT and is_explicit_push:
             return new_board, False, False, False, penalty_for_step

        # Perform move
        current_val = new_board[pr, pc]
        new_board[pr, pc] = TARGET if current_val == PLAYER_ON_TARGET else FLOOR
        new_board[nr, nc] = PLAYER_ON_TARGET if target_val == TARGET else PLAYER
        return new_board, True, False, False, penalty_for_step
        
    # 3. Push box (Push logic)
    if target_val == BOX or target_val == BOX_ON_TARGET:
        # [Mode 1 Check]: explicit Move action (0-3) into a box is illegal
        if action_mode == MODE_EXPLICIT and not is_explicit_push:
            return new_board, False, False, False, penalty_for_step

        nnr, nnc = nr + dr, nc + dc
        if nnr < 0 or nnr >= h or nnc < 0 or nnc >= w:
            return new_board, False, False, False, penalty_for_step
            
        behind_val = new_board[nnr, nnc]
        
        if behind_val == FLOOR or behind_val == TARGET:
            # Move box
            new_board[nnr, nnc] = BOX_ON_TARGET if behind_val == TARGET else BOX
            # Move player
            current_val = new_board[pr, pc]
            new_board[pr, pc] = TARGET if current_val == PLAYER_ON_TARGET else FLOOR
            new_board[nr, nc] = PLAYER_ON_TARGET if target_val == BOX_ON_TARGET else PLAYER

            # ====== Reward calculation (aligned with SokobanEnv._calc_reward and
            # get_reward_from_state_action_sokoban) ======
            # Reward can accumulate, e.g.:
            #   - push box onto target: -0.1 (step) + 1.0 (box on target) = 0.9
            #   - push box onto target and finish: -0.1 (step) + 1.0 (box on target)
            #     + 10.0 (finished) = 10.9

            # Base step penalty (applied to all successful actions)
            reward = penalty_for_step

            # Count box statistics in current state
            current_boxes_on_target = 0
            boxes_off_target_after = 0
            for r in range(h):
                for c in range(w):
                    v = new_board[r, c]
                    if v == BOX_ON_TARGET:
                        current_boxes_on_target += 1
                    if v == BOX:
                        boxes_off_target_after += 1

            # Change in boxes on target (can be >1 if multiple boxes move onto targets)
            if current_boxes_on_target > prev_boxes_on_target:
                reward += reward_box_on_target * (current_boxes_on_target - prev_boxes_on_target)
            elif current_boxes_on_target < prev_boxes_on_target:
                reward += penalty_box_off_target * (prev_boxes_on_target - current_boxes_on_target)

            # When all boxes are on targets, add terminal reward (in addition to box-on-target reward)
            if prev_num_boxes > 0 and boxes_off_target_after == 0 and current_boxes_on_target == prev_num_boxes:
                reward += reward_finished
            # ====== End of reward calculation ======

            is_solved = _numba_is_solved(new_board)
            is_deadlock = False

            return new_board, True, is_solved, is_deadlock, reward
            
    return new_board, False, False, False, penalty_for_step

@njit(fastmath=True)
def _numba_get_value_from_action_first(board_arr, first_action, rollout_time, max_steps, gamma, num_actions, action_mode):
    if rollout_time <= 0 or max_steps <= 0:
        return 0.0

    total_value = 0.0

    for _ in range(rollout_time):
        curr_board = board_arr.copy()
        value = 0.0
        discount = 1.0

        next_board, moved, solved, deadlock, reward = _numba_step_with_reward(
            curr_board, first_action, action_mode
        )

        value += reward * discount
        discount *= gamma

        if not moved:
            total_value += value

        curr_board = next_board

        if solved:
            total_value += value
            continue

        # Subsequent steps: random rollout
        for _step in range(1, max_steps):
            # Use numba-supported random interface
            action_int = np.random.randint(0, num_actions)

            next_board, moved, solved, deadlock, reward = _numba_step_with_reward(
                curr_board, action_int, action_mode
            )

            value += reward * discount
            discount *= gamma

            curr_board = next_board

            if solved:
                break

        total_value += value

    return total_value / rollout_time


@njit(fastmath=True)
def _numba_get_value_from_state_sokoban(board_arr, rollout_time, max_steps, gamma, num_actions, action_mode):
    """
    Use numba to run the random rollout loop at C speed, significantly reducing
    the overhead of Python for-loops and random calls.
    """
    if rollout_time <= 0 and max_steps <= 0:
        return 0.0

    total_value = 0.0

    for _ in range(rollout_time):
        # Each rollout starts from the same initial state
        curr_board = board_arr
        value = 0.0
        discount = 1.0

        for _step in range(max_steps):
            # Use numba-supported random interface
            action_int = np.random.randint(0, num_actions)

            next_board, moved, solved, deadlock, reward = _numba_step_with_reward(
                curr_board, action_int, action_mode
            )

            # Accumulate discounted return by iteratively multiplying gamma
            value += reward * discount
            discount *= gamma

            # Update current board
            curr_board = next_board

            # Terminate this rollout once the puzzle is solved
            if solved:
                break

        total_value += value

    return total_value / rollout_time

class SokobanVariantEnv:
    def __init__(self, map_file_path, variant="origin", max_steps_episode=200):
        self.sokoban_symbol_to_char = {
            "origin": {
                0: "#",  # wall
                1: " ",  # floor
                2: "?",  # target
                3: "*",  # box on target
                4: "$",  # box
                5: "@",  # player
                6: "+",  # player on target
            },
            "symbol": {
                0: "#",  # wall
                1: "_",  # floor
                2: "O",  # target
                3: "√",  # box on target
                4: "B",  # box
                5: "P",  # player
                6: "+",  # player on target
            }
        }
        self.variant = variant
        assert self.variant in ["origin", "action", "symbol", "extra"], f"Invalid variant: {self.variant}"

        self.action_idx = {
            "up": 5,
            "down": 6,
            "left": 7,
            "right": 8,
            "push up": 1,
            "push down": 2,
            "push left": 3,
            "push right": 4,
        }
        self.max_steps_episode = max_steps_episode

        self.previous_boxes_on_target_for_perf: int = 0
        self.current_episode_cumulative_perf_score: float = 0.0
        self.cumulative_boxes_on_target_completed: int = 0
        self.room_fixed: Optional[np.ndarray] = None
        self.room_state: Optional[np.ndarray] = None
        self.player_position: Optional[np.ndarray] = None
        self.num_boxes_current: int = 0 # Number of boxes in the current level
        self.boxes_on_target: int = 0
        self.num_env_steps: int = 0
        self.current_reward_last_step: float = 0.0
        self.box_mapping: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None

        self.predefined_levels = {}
        self._load_predefined_levels(map_file_path)

        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1.0 
        self.reward_box_on_target = 1.0
        self.reward_finished = 10.0
        self.current_stuck_steps = 0

    def _load_predefined_levels(self, level_txt_file_path: str):
        if not os.path.exists(level_txt_file_path): return
        with open(level_txt_file_path, 'r') as f: content = f.read()
        
        level_blocks = re.split(r"Level\s+\d+", content)
        current_level_num = 0
        for block in level_blocks:
            if not block.strip(): continue
            current_level_num +=1
            # Corrected line splitting and stripping
            level_lines_raw = block.strip().splitlines()
            level_lines = [line.strip() for line in level_lines_raw if line.strip()]
            
            if level_lines:
                first_line_width = len(level_lines[0])
                if not all(len(line) == first_line_width for line in level_lines):
                    continue
                self.predefined_levels[current_level_num] = level_lines

    def _parse_level_data(self, level_str_lines: List[str]) -> bool:
        rows = len(level_str_lines)
        if rows == 0: return False
        cols = len(level_str_lines[0])
        if cols == 0: return False

        self.dim_room = (rows, cols)
        _room_fixed = np.ones(self.dim_room, dtype=np.uint8) # Default to floor (1)
        _player_pos_list = []
        _num_boxes = 0

        # First pass: Set up _room_fixed (walls, targets, default floor for dynamic items)
        for r in range(rows):
            for c in range(cols):
                char = level_str_lines[r][c]
                if char == '#': # Wall
                    _room_fixed[r, c] = 0
                elif char == '?': # Target
                    _room_fixed[r, c] = 2
                # For '$', '*', '@', the underlying _room_fixed is floor (1), which is the default.
                # If '*' implies a target, it's handled in the second pass by explicitly setting _room_fixed.

        _room_state = _room_fixed.copy() # Initialize _room_state based on fixed layout

        # Second pass: Place dynamic entities (player, boxes) onto _room_state
        for r in range(rows):
            for c in range(cols):
                char = level_str_lines[r][c]
                if char == '$': # Box
                    if _room_fixed[r, c] == 2: # Box placed on a target square (e.g. '$' on a '?')
                        _room_state[r, c] = 3 # Box on Target
                    else: # Box on Floor
                        _room_state[r, c] = 4 # Box
                    _num_boxes += 1
                elif char == '*': # Box explicitly on Target
                    _room_fixed[r, c] = 2 # Ensure underlying fixed map shows a target
                    _room_state[r, c] = 3 # Box on Target
                    _num_boxes += 1
                elif char == '@': # Player
                    if _room_fixed[r, c] == 2: # Player on a Target square (e.g. '@' on a '?')
                        _room_state[r, c] = 6 # Player on Target
                    else: # Player on Floor
                        _room_state[r, c] = 5 # Player on Floor
                    _player_pos_list.append(np.array([r, c]))
                # The character '+' for player on target is not used by the provided levels.txt
        
        self.room_fixed = _room_fixed
        self.room_state = _room_state
        self.num_boxes_current = _num_boxes

        if not _player_pos_list: # No player '@' found in level string
            # print("[SokobanEnv] Warning: No player '@' found in level. Attempting to place player on an available floor or target square.")
            # Try to find a floor or target square to place the player
            available_squares = np.argwhere((self.room_state == 1) | (self.room_state == 2)) # Floor or Empty Target
            if available_squares.size > 0:
                # Place player randomly on one of the available squares
                self.player_position = available_squares[np.random.choice(len(available_squares))]
            else: # Absolute fallback: try center; this might be a wall.
                # print("[SokobanEnv] Warning: No floor or target squares available for fallback player placement. Placing at center.")
                self.player_position = np.array([self.dim_room[0] // 2, self.dim_room[1] // 2])

            r_p, c_p = self.player_position
            # Check if chosen position is valid and not a wall before placing player state
            if self.room_fixed[r_p, c_p] != 0: # If not a wall
                 self.room_state[r_p, c_p] = 6 if self.room_fixed[r_p, c_p] == 2 else 5
            else: # Chosen fallback is a wall, this is bad. Player is effectively stuck or invalid.
                pass
                #  print(f"[SokobanEnv] CRITICAL: Fallback player position ({r_p},{c_p}) is a wall. Level may be unplayable.")
                 # Keep player_position, but room_state at wall remains wall. Agent might get stuck immediately.
        elif len(_player_pos_list) > 1:
            # print(f"[SokobanEnv] Warning: Multiple players ({len(_player_pos_list)}) found in level. Using the first one.")
            self.player_position = _player_pos_list[0]
        else: # Exactly one player found
            self.player_position = _player_pos_list[0]

        # Update observation space based on actual loaded level dimensions
        self.observation_space = Box(low=0, high=6, shape=self.dim_room, dtype=np.uint8)
        return True


    def reset(self, level_idx: int):
        if level_idx not in self.predefined_levels:
            raise ValueError(f"Level {level_idx} not found")

        self.num_env_steps = 0
        self.current_reward_last_step = 0.0
        
        # Reset Sokoban-specific performance score trackers
        self.previous_boxes_on_target_for_perf = 0

        level_loaded_ok = False
        current_level_to_load = level_idx
        level_data_str = self.predefined_levels[current_level_to_load]
        if self._parse_level_data(level_data_str):
            level_loaded_ok = True

    def _internal_push_or_move(self, action_mapped_idx: int) -> Tuple[bool, bool]:
        # action_mapped_idx: 0-no_op, 1-push_up, ..., 8-move_right
        # We need to map this to simple directions 0-up, 1-down, 2-left, 3-right
        # and determine if it's a push or move.
        
        moved_player, moved_box = False, False
        if action_mapped_idx == 0: return False, False # no_op

        is_push_action = 1 <= action_mapped_idx <= 4
        is_move_action = 5 <= action_mapped_idx <= 8
        
        direction_idx = -1
        if is_push_action: direction_idx = action_mapped_idx - 1 # 1-4 -> 0-3
        elif is_move_action: direction_idx = action_mapped_idx - 5 # 5-8 -> 0-3
        else: return False, False # Should not happen if action mapping is correct

        change = CHANGE_COORDINATES[direction_idx]
        player_r, player_c = self.player_position
        next_player_r, next_player_c = player_r + change[0], player_c + change[1]

        # Boundary checks for next_player_pos
        if not (0 <= next_player_r < self.dim_room[0] and 0 <= next_player_c < self.dim_room[1]):
            return False, False # Player would move out of bounds

        if is_push_action:
            box_target_r, box_target_c = next_player_r + change[0], next_player_c + change[1]
            # Boundary checks for box_target_pos
            if not (0 <= box_target_r < self.dim_room[0] and 0 <= box_target_c < self.dim_room[1]):
                return False, False # Box would be pushed out of bounds

            is_box_at_next = self.room_state[next_player_r, next_player_c] in [3, 4] # Box or BoxOnTarget
            can_box_move_to_target = self.room_state[box_target_r, box_target_c] in [1, 2] # Floor or Target

            if is_box_at_next and can_box_move_to_target:
                # Move Box
                self.room_state[box_target_r, box_target_c] = 3 if self.room_fixed[box_target_r, box_target_c] == 2 else 4
                moved_box = True
                # Move Player (follows box)
                self.room_state[next_player_r, next_player_c] = 6 if self.room_fixed[next_player_r, next_player_c] == 2 else 5
                self.room_state[player_r, player_c] = self.room_fixed[player_r, player_c] # Restore original player tile
                self.player_position = np.array([next_player_r, next_player_c])
                moved_player = True
            # else: # Cannot push, try to move if agent intended push but couldn't
            # This behavior (falling back to move if push fails) can be complex.
            # Let's make it explicit: if a push action is chosen, it either pushes or does nothing.
            # If agent wants to "try push then move", it should send a "move" action if push is known to fail.
            # For now, a failed push is just a failed push.
            
        elif is_move_action:
            if self.room_state[next_player_r, next_player_c] in [1, 2]: # Floor or Target
                self.room_state[next_player_r, next_player_c] = 6 if self.room_fixed[next_player_r, next_player_c] == 2 else 5
                self.room_state[player_r, player_c] = self.room_fixed[player_r, player_c] # Restore original player tile
                self.player_position = np.array([next_player_r, next_player_c])
                moved_player = True
        
        return moved_player, moved_box



    def _calc_reward(self) -> float:
        # Adapted from sokoban_env_old.py
        reward = self.penalty_for_step
        
        current_boxes_on_target = np.count_nonzero(self.room_state == 3) # state 3 is box_on_target
        
        if current_boxes_on_target > self.boxes_on_target:
            reward += self.reward_box_on_target * (current_boxes_on_target - self.boxes_on_target)
        elif current_boxes_on_target < self.boxes_on_target:
            # This implies a box was moved OFF a target.
            reward += self.penalty_box_off_target * (self.boxes_on_target - current_boxes_on_target)
        
        self.boxes_on_target = current_boxes_on_target
        
        if self._check_if_all_boxes_on_target():
            reward += self.reward_finished
        
        return float(reward)
    
    def _check_if_all_boxes_on_target(self) -> bool:
        if self.room_state is None or self.num_boxes_current == 0: return True # No boxes means solved
        num_boxes_off_target = np.count_nonzero(self.room_state == 4) # Count boxes not on target (state 4)
        return num_boxes_off_target == 0 and self.boxes_on_target == self.num_boxes_current
        
    def _check_if_maxsteps(self) -> bool:
        return self.num_env_steps >= self.max_steps_episode

    def _get_raw_board_obs(self) -> np.ndarray:
        return self.room_state.copy() if self.room_state is not None else np.zeros(self.dim_room, dtype=np.uint8)

    def calculate_perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        """
        Calculates a performance score for the current step for Sokoban.
        This is based on the cumulative count of newly placed boxes on targets 
        within the current episode.

        Args:
            reward (float): The reward received for the step (not directly used here).
            info (Dict[str, Any]): Additional information from the environment, expected
                                   to contain "boxes_on_target".

        Returns:
            float: The cumulative performance score for the episode up to this step.
        """
        current_boxes_on_target = info.get("boxes_on_target", 0)
        
        delta_boxes = 0
        if current_boxes_on_target > self.previous_boxes_on_target_for_perf:
            delta_boxes = current_boxes_on_target - self.previous_boxes_on_target_for_perf
        
        # Only add positive delta to the cumulative score
        if delta_boxes > 0:
            self.current_episode_cumulative_perf_score += float(delta_boxes)
            
        self.previous_boxes_on_target_for_perf = current_boxes_on_target
        
        return self.current_episode_cumulative_perf_score

    def get_text_obs(self, state=None):
        if state is None:
            state = self.room_state
        return self.get_symbol_board_from_state(state)

    def get_symbol_board_from_state(self, state: np.ndarray) -> str:
        if self.variant == "symbol":
            state_to_char_dict = self.sokoban_symbol_to_char[self.variant]
        else:
            state_to_char_dict = self.sokoban_symbol_to_char["origin"]
        char_board_2d_list = [[state_to_char_dict.get(tile, '?') for tile in row] for row in state.tolist()]
        return "\n".join(["".join(row) for row in char_board_2d_list])

    def _get_info(self) -> Dict[str, Any]:
        return {
            "num_env_steps": self.num_env_steps,
            "player_position": self.player_position.tolist() if self.player_position is not None else None,
            "boxes_on_target": self.boxes_on_target,
            "num_boxes": self.num_boxes_current,
            "all_boxes_on_target": self._check_if_all_boxes_on_target(),
            "reward_last_step": self.current_reward_last_step,
            "total_score": int(self.cumulative_boxes_on_target_completed + self.boxes_on_target),
        }


    def step(self, agent_action_str: Optional[str], thought_process: str = "", time_taken_s: float = 0.0):
        if self.variant == "action":
            agent_action_str = self.action_variant_mapping(self.room_state, agent_action_str)
        
        env_action_idx = self.action_idx.get(agent_action_str, None)

        reward = 0.0
        terminated = False
        truncated = False
        
        if env_action_idx is not None:
            self._internal_push_or_move(env_action_idx)
            reward = self._calc_reward()
            terminated = self._check_if_all_boxes_on_target()

        else: # Invalid or no action from agent
            reward = self.penalty_for_step
            terminated = self._check_if_all_boxes_on_target()

        self.num_env_steps += 1
        truncated = self._check_if_maxsteps()
        self.current_reward_last_step = reward

        raw_board_obs = self._get_raw_board_obs()
        info_dict = self._get_info()
        current_perf_score = self.calculate_perf_score(reward, info_dict)
        
        symbol_board = self.get_symbol_board_from_state(raw_board_obs)

        return symbol_board, reward, terminated, truncated, info_dict, current_perf_score

    def action_variant_mapping(self, board: np.ndarray, action_str: str) -> str:
        if type(action_str) != str:
            return action_str
        if "push" in action_str:
            return f"illegal {action_str}"
        if action_str in ["up", "down", "left", "right"]:
            player_positions = np.argwhere((board == 5) | (board == 6))
            if player_positions.size == 0:
                return action_str

            pr, pc = player_positions[0]

            direction_delta = {
                "up": (-1, 0),
                "down": (1, 0),
                "left": (0, -1),
                "right": (0, 1),
            }
            dr, dc = direction_delta[action_str]
            next_r, next_c = pr + dr, pc + dc

            if next_r < 0 or next_r >= board.shape[0] or next_c < 0 or next_c >= board.shape[1]:
                return action_str

            if board[next_r, next_c] in [4, 3]:
                return f"push {action_str}"
        return action_str

    def get_action_from_response(self, response):
        if not response:
            return {"action": None, "thought": "No response received"}
        
        # Initialize result with defaults
        result = {
            "action": None,
            "thought": None
        }
        
        # Use regex to find thought and action sections
        # Match patterns like "thought:", "# thought:", "Thought:", etc.
        thought_pattern = r'(?:^|\n)(?:#\s*)?thought:(.+?)(?=(?:\n(?:#\s*)?(?:action|move):)|$)'
        action_pattern = r'(?:^|\n)(?:#\s*)?(?:action|move):(.+?)(?=(?:\n(?:#\s*)?thought:)|$)'
        
        # Find thought section using regex (case insensitive)
        thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Find action section using regex (case insensitive)
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # If no structured format was found, treat the whole response as thought
        if not result["thought"] and not result["action"]:
            result["thought"] = response.strip()
        elif not result["thought"]:  # If only action was found
            # Look for any text before the action as thought
            pre_action = re.split(r'(?:^|\n)(?:#\s*)?(?:action|move):', response, flags=re.IGNORECASE)[0]
            if pre_action and pre_action.strip():
                result["thought"] = pre_action.strip()
        
        return result

    # use for mc-critic
    def get_return_for_state_all_action(self, state, rollout_time=1000, gamma=0.95, max_steps=200):
        board_arr = state
        if isinstance(board_arr, list):
            # It may be a list of strings (level layout) or a 2D int list
            if len(board_arr) > 0 and isinstance(board_arr[0], str):
                board_arr = str_board_to_int(board_arr)
            else:
                board_arr = np.asarray(board_arr, dtype=np.int8)
        else:
            board_arr = np.asarray(board_arr, dtype=np.int8)

        action_mode = MODE_EXPLICIT
        num_actions = 8  # Explicit mode: 0-3 Move, 4-7 Push

        # Compute value for each of the 8 actions
        action_values = []
        for action in range(num_actions):
            value = _numba_get_value_from_action_first(
                board_arr, action, rollout_time, max_steps, gamma, num_actions, action_mode
            )
            action_values.append(value)

        return {
            "up": action_values[0],
            "down": action_values[1],
            "left": action_values[2],
            "right": action_values[3],
            "push up": action_values[4],
            "push down": action_values[5],
            "push left": action_values[6],
            "push right": action_values[7],
        }

    def get_reward_for_state_action(self, state, action):
        """
        Compute immediate reward for a given state and action.

        The `state` can be:
          - numpy array
          - 2D list of ints
          - list of strings (board layout)

        The `action` can be:
          - integer in [0, 7] (explicit mode index)
          - string in {"up", "down", "left", "right",
                       "push up", "push down", "push left", "push right"}
        which will be mapped to the corresponding explicit-mode index.
        """
        # Normalize board to numba-friendly int8 ndarray
        board_arr = state
        if isinstance(board_arr, list):
            if len(board_arr) > 0 and isinstance(board_arr[0], str):
                board_arr = str_board_to_int(board_arr)
            else:
                board_arr = np.asarray(board_arr, dtype=np.int8)
        else:
            board_arr = np.asarray(board_arr, dtype=np.int8)

        # Map string action to explicit-mode index if needed
        if isinstance(action, str):
            a = action.lower()
            action_mapping = {
                "up": 0,
                "down": 1,
                "left": 2,
                "right": 3,
                "push up": 4,
                "push down": 5,
                "push left": 6,
                "push right": 7,
            }
            if a not in action_mapping:
                raise ValueError(f"Unknown action string for mc-critic: {action}")
            action_idx = action_mapping[a]
        else:
            action_idx = int(action)

        _, _, _, _, reward = _numba_step_with_reward(
            board_arr, action_idx, MODE_EXPLICIT
        )
        return float(reward)

    def get_value_from_state_sokoban(self, state, rollout_time=1000, max_steps=20, gamma=0.95):
        """
        Estimate the value of the current board state using random rollout.
        Uses explicit action mode and the same reward function as
        `get_reward_from_state_action_sokoban`.
        """
        # Convert to a numba-friendly numpy int8 array
        board_arr = state
        if isinstance(board_arr, list):
            # It may be a list of strings (level layout) or a 2D int list
            if len(board_arr) > 0 and isinstance(board_arr[0], str):
                board_arr = str_board_to_int(board_arr)
            else:
                board_arr = np.asarray(board_arr, dtype=np.int8)
        else:
            board_arr = np.asarray(board_arr, dtype=np.int8)

        action_mode = MODE_EXPLICIT
        num_actions = 8  # Explicit mode: 0-3 Move, 4-7 Push

        # Delegate the core computation to the numba function to avoid Python
        # double for-loops and random calls
        value = _numba_get_value_from_state_sokoban(
            board_arr, rollout_time, max_steps, gamma, num_actions, action_mode
        )
        return float(value)


if __name__ == "__main__":
    map_file_path = "envs/levels_random.txt"
    variant = "symbol"
    env = SokobanVariantEnv(map_file_path=map_file_path, variant=variant, max_steps_episode=5)
    env.reset(level_idx=1)
    while True:
        current_board = env.get_symbol_board_from_state(env.room_state)
        print("current board:")
        print(current_board)
        action = input("Action: ")
        symbol_board, reward, terminated, truncated, info_dict, current_perf_score = env.step(agent_action_str=action)
        print(f"reward={reward}, terminated={terminated}, truncated={truncated}, info_dict={info_dict}, current_perf_score={current_perf_score}")
        get_return_for_state_all_action = env.get_return_for_state_all_action(env.room_state)
        print(f"get_return_for_state_all_action={get_return_for_state_all_action}")
        if terminated or truncated:
            break