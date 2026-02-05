import numpy as np
import random
import re
import copy
import secrets

class TwentyFortyEightVariantEnv:
    def __init__(self, size, base, max_stuck_steps=10):
        self.size = size
        self.base = base
        self.max_stuck_steps = max_stuck_steps
        self.action_2_value = {}
    
    def reset(self, seed=None, grid=None, unchange_steps=None):
        if seed is not None:
            np.random.seed(seed)
        if grid is None:
            self.grid = [[0] * self.size for _ in range(self.size)]
            self.spawn_tile()
            self.spawn_tile()
        else:
            self.grid = copy.deepcopy(grid)
        self.total_score = 0
        self.unchange_steps = 0
        self.step_idx = 0
        if unchange_steps is not None:
            self.unchange_steps = unchange_steps

    def get_reward_for_state_action(self, state, action):
        self.reset(grid=state)
        reward, _ = self.step(action)
        return reward
    
    # use for mc-critic
    def get_return_for_state_action(self, state, action, value_calc_way="mean", num_rollouts=1000, gamma=0.9, max_steps=1000):
        self.reset(grid=copy.deepcopy(state))
        self.step(action)
        first_action = action
        if not self.format_correct:
            first_action = "null"
        elif not self.is_legal_move:
            first_action = "illegal"
        if first_action in self.action_2_value:
            return self.action_2_value[first_action], first_action

        value_list = []
        for _ in range(num_rollouts):
            seed = secrets.randbelow(2**31)
            self.reset(grid=copy.deepcopy(state), seed=seed)
            first_reward, done = self.step(action)
            value = first_reward
            step = 1
            while not done and step <= max_steps:
                reward, done = self.step(random.choice(["left", "right", "up", "down"]))
                value += gamma ** step * reward
                step += 1
            value_list.append(value)
        if value_calc_way == "mean":
            result = sum(value_list) / len(value_list)
        elif value_calc_way == "max":
            result = max(value_list)
        self.action_2_value[first_action] = result
        return result, first_action

    def get_value_for_state(self, state, value_calc_way="mean", num_rollouts=1000, gamma=0.9, max_steps=1000, unchange_steps=None):
        value_list = []
        for _ in range(num_rollouts):
            self.reset(grid=state, unchange_steps=unchange_steps, seed=secrets.randbelow(2**31))
            step = 0
            value = 0
            while step < max_steps:
                reward, done = self.step(random.choice(["left", "right", "up", "down"]))
                value += gamma ** step * reward
                step += 1
                if done:
                    break
            value_list.append(value)
        # print(value_list)
        if value_calc_way == "mean":
            result = sum(value_list) / len(value_list)
        elif value_calc_way == "max":
            result = max(value_list)
        return result

    def spawn_tile(self):
        grid = np.array(self.grid)
        rows, cols = np.where(grid == 0)
        if not len(rows):
            return # No space to spawn
        index = np.random.choice(len(rows))
        value = self.base if np.random.rand() > 0.1 else self.base * 2
        grid[rows[index], cols[index]] = value
        self.grid = [[int(x) for x in row] for row in grid.tolist()]

    def step(self, direction):
        grid = np.array(self.grid)
        score = 0
        self.format_correct = True
        if isinstance(direction, str):
            direction = direction.lower()
        
        def merge_row(row):
            nonlocal score
            non_zero = [x for x in row if x != 0]
            merged = []
            i = 0
            
            while i < len(non_zero):
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged.append(non_zero[i] * 2)
                    score += non_zero[i] * 2
                    i += 2
                else:
                    merged.append(non_zero[i])
                    i += 1
            
            merged += [0] * (self.size - len(merged))
            return merged
        
        if direction == 'left':
            new_grid = [merge_row(row) for row in grid]
        elif direction == 'right':
            new_grid = [merge_row(row[::-1])[::-1] for row in grid]
        elif direction == 'up':
            transposed = grid.T
            new_transposed = [merge_row(row) for row in transposed]
            new_grid = np.array(new_transposed).T.tolist()
        elif direction == 'down':
            transposed = grid.T
            new_transposed = [merge_row(row[::-1])[::-1] for row in transposed]
            new_grid = np.array(new_transposed).T.tolist()
        else:
            new_grid = self.grid
            self.format_correct = False
        
        new_grid = [[int(x) for x in row] for row in new_grid]

        if new_grid != self.grid:
            self.grid = new_grid
            self.spawn_tile()
            self.unchange_steps = 0
            self.is_legal_move = True
        else:
            self.unchange_steps += 1
            self.is_legal_move = False
        
        # if self.unchange_steps >= self.max_stuck_steps:
        #     print(f"Terminating episode due to unchanged observation for {self.max_stuck_steps} steps. step_idx={self.step_idx} grid={self.grid}")

        terminate = self.is_game_over() or self.unchange_steps >= self.max_stuck_steps
        self.total_score += score
        self.step_idx += 1
        if not self.is_legal_move:
            score = -1
        if not self.format_correct:
            score = -10
        return score, terminate
    
    def is_game_over(self):
        if any(0 in row for row in self.grid):
            return False
        
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return False
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return False
        
        return True
    
    def get_text_obs(self):
        lines = []
        for row in self.grid:
            row_str = []
            for cell in row:
                if cell == 0:
                    row_str.append(".".rjust(5))
                else:
                    val = str(cell)
                    row_str.append(val.rjust(5))
            lines.append("[" + " ".join(row_str) + "]")
        board_str = "\n".join(lines)
        return str({'board': board_str})
    
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

# test mc-critic
if __name__ == "__main__":
    env = TwentyFortyEightVariantEnv(size=4, base=2)
    env.reset()

    state = \
    [[    2,    16,     4,    16],
     [    2,   128,    64,     4],
     [    8,    16,     8,     2],
     [    4,     4,     2,     4]]
    import time
    return_list = []
    for action in ["left", "right", "up", "down"]:
        start = time.time()
        r, _ = env.get_return_for_state_action(state, action)
        return_list.append(round(r, 4))
        end = time.time()
        print(f"action = {action}, value = {r}, time = {end - start}")
    print(return_list)
