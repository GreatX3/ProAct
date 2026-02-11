import asyncio
import os
import uuid
from typing import Callable

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.experimental.openai import ArealOpenAI

from envs.twentyFortyEightVariantEnv import TwentyFortyEightVariantEnv
from envs.sokobanEnv import SokobanVariantEnv
import hashlib
import random
import time
import math
import numpy as np
import secrets
import pickle
import json
import glob
import re


logger = logging.getLogger("Traj GRPO workflow")

def group_normalization(reward_list):
    reward_tensor = torch.tensor(reward_list)
    return ((reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-5)).tolist()

class TrajGRPOWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        max_turns: int = 10000,
        is_eval: bool = False,
        game_name: str = "twenty_forty_eight",
        rank: int = 0,
        world_size: int = 1,
        variant_games_prompt_template_path: str | None = None,
        train_turns_per_gpu: int = 1,
        fix_test_env_seed: bool = False,
        eval_trajs_per_gpu: int = 1,
    ):
        self.fix_test_env_seed = fix_test_env_seed
        self.eval_trajs_per_gpu = eval_trajs_per_gpu
        self.train_turns_per_gpu = train_turns_per_gpu
        self.variant_games_prompt_template_path = variant_games_prompt_template_path
        self.rank = rank
        self.world_size = world_size
        self.game_name = game_name
        self.is_eval = is_eval
        self.max_turns = max_turns
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        self.sokoban_level_txt_file_path = './envs/levels_random.txt'
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
    

        if self.is_eval:
            self.n_trajs = 1
        else:
            self.n_trajs = self.gconfig.n_samples
        
        self.gamename_to_runevalrollout_func = {
            "twenty_forty_eight": self.arun_eval_rollout_2048,
            "sokoban": self.arun_eval_rollout_sokoban,
        }

        self.gamename_to_variant_game_runeval_func = {
            "twenty_forty_eight": self.arun_one_episode_2048_variant,
            "sokoban": self.arun_one_episode_sokoban_variant,
        }

        self.gamename_to_runepisode_func = {
            "twenty_forty_eight": self.arun_one_episode_turn_independent_2048,
            "sokoban": self.arun_one_episode_turn_independent_sokoban,
        }

        self.preprocess()
        self.turn_sample_pool = []

    def preprocess(self):
        if self.game_name == "twenty_forty_eight":
            if self.variant_games_prompt_template_path:
                with open(self.variant_games_prompt_template_path, 'r') as f:
                    self.variant_games_prompt_template = json.load(f)
                self.system_prompt_template = self.variant_games_prompt_template["2048_4x4"]["system_prompt_template"]
                self.user_prompt_template = self.variant_games_prompt_template["2048_4x4"]["user_prompt_template"]

        elif self.game_name == "sokoban":
            if self.variant_games_prompt_template_path:
                with open(self.variant_games_prompt_template_path, 'r') as f:
                    self.variant_games_prompt_template = json.load(f)
                self.system_prompt_template = self.variant_games_prompt_template["origin"]["system_prompt_template"]
                self.user_prompt_template = self.variant_games_prompt_template["origin"]["user_prompt_template"]
    
    
    async def arun_one_episode_2048_variant(self, engine: InferenceEngine, game_name, data, traj_idx):
        if self.fix_test_env_seed:
            seed = data
        else:
            seed = secrets.randbelow(2**31)
        if game_name == "2048_3x3":
            size, base = 3, 2
        elif game_name == "3072_4x4":
            size, base = 4, 3
        else:
            raise ValueError(f"Unknown game name: {game_name}")

        game_env = TwentyFortyEightVariantEnv(size=size, base=base)
        game_env.reset(seed=seed)

        system_prompt = self.variant_games_prompt_template[game_name]["system_prompt_template"]
        user_prompt_template = self.variant_games_prompt_template[game_name]["user_prompt_template"]
        
        system_messages = [{"role": "system", "content": system_prompt}]
        system_input_ids = self.tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        system_tokens_len = len(system_input_ids)
        traj_dump_str = f"System Prompt=\n{system_prompt}\nsystem_tokens_len={system_tokens_len}\n"

        turn_idx = 0
        total_score = 0
        output_token_len_list = []
        request_time_list = []
        while True:
            text_obs = game_env.get_text_obs()
            user_prompt = user_prompt_template.format(textual_representation=text_obs)

            user_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            user_input_ids = self.tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            user_tokens_len = len(user_input_ids)

            start_time = time.time()
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=user_input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            assert resp.input_len == user_tokens_len, f"resp.input_len != user_tokens_len, {resp.input_len} != {user_tokens_len}"
            
            resp_str = self.tokenizer.decode(resp.output_tokens)

            origin_resp_str = resp_str
            if self.enable_thinking and "</think>" in resp_str:
                resp_str = resp_str.split("</think>")[-1].strip()

            action_dict = game_env.get_action_from_response(resp_str.replace("<|im_end|>", ""))
            
            end_time = time.time()
            time_taken_s = end_time - start_time
            action_str = None
            if action_dict and action_dict.get("action") is not None:
                action_str = action_dict.get("action")
                action_str = str(action_str).strip().lower()
    
            reward, terminated = game_env.step(action_str)

            total_score = game_env.total_score

            output_token_len_list.append(resp.output_len)
            request_time_list.append(time_taken_s)

            traj_dump_str += f"{"-"*80}\nTurn {turn_idx}\n{"-"*80}\nUser Prompt=\n{user_prompt}\nResponse=\n{origin_resp_str}\n"
            traj_dump_str += f"action_str={action_str}, user_tokens_len={user_tokens_len}, respone token len={resp.output_len}, time_taken_s={time_taken_s}\n"
            turn_idx += 1
            if terminated or turn_idx >= self.max_turns:
                break
        
        ave_output_token_len = sum(output_token_len_list) / len(output_token_len_list)
        ave_request_time = sum(request_time_list) / len(request_time_list)
        print(f"collect one traj for {game_name}: {self.rollout_stat_scope} turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}")
        # Log reward.
        log_dict = {f"total_score_{game_name}": total_score, 
                    f"turns_{game_name}": turn_idx, 
                    f"ave_output_token_len_{game_name}": ave_output_token_len, 
                    f"ave_request_time_{game_name}": ave_request_time}
        stats_tracker.get(self.rollout_stat_scope).scalar(**log_dict)
        traj_dump_str += f"Final Results:\n turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}"

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(engine.get_version()))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{game_name}_seed{seed}_traj{traj_idx}.txt")
            async with aiofiles.open(file_path, "a") as f:
                await f.write(traj_dump_str)

        return True

    async def arun_one_episode_sokoban_variant(self, engine: InferenceEngine, game_name, data, traj_idx):
        if self.fix_test_env_seed:
            seed = data
        else:
            seed = secrets.randbelow(2**31)
        if game_name not in ["extra", "action", "symbol"]:
            raise ValueError(f"Unknown game variant name: {game_name}")

        game_env = SokobanVariantEnv(map_file_path=self.sokoban_level_txt_file_path, variant=game_name)
        level_idx = random.randint(1, 6)
        if game_name == "extra":
            level_idx = random.randint(7, 8)
        game_env.reset(level_idx=level_idx)

        prompt_mode = game_name
        if game_name == "extra":
            prompt_mode = "origin"

        system_prompt = self.variant_games_prompt_template[prompt_mode]["system_prompt_template"]
        user_prompt_template = self.variant_games_prompt_template[prompt_mode]["user_prompt_template"]
        
        system_messages = [{"role": "system", "content": system_prompt}]
        system_input_ids = self.tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        system_tokens_len = len(system_input_ids)
        traj_dump_str = f"System Prompt=\n{system_prompt}\nsystem_tokens_len={system_tokens_len}\n"

        turn_idx = 0
        total_score = 0
        output_token_len_list = []
        request_time_list = []
        while True:
            text_obs = game_env.get_text_obs()
            user_prompt = user_prompt_template.format(textual_representation=text_obs)

            user_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            user_input_ids = self.tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            user_tokens_len = len(user_input_ids)

            start_time = time.time()
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=user_input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            assert resp.input_len == user_tokens_len, f"resp.input_len != user_tokens_len, {resp.input_len} != {user_tokens_len}"
            
            resp_str = self.tokenizer.decode(resp.output_tokens)

            origin_resp_str = resp_str
            if self.enable_thinking and "</think>" in resp_str:
                resp_str = resp_str.split("</think>")[-1].strip()

            action_dict = game_env.get_action_from_response(resp_str.replace("<|im_end|>", ""))
            
            end_time = time.time()
            time_taken_s = end_time - start_time
            action_str = None
            if action_dict and action_dict.get("action") is not None:
                action_str = action_dict.get("action")
                action_str = str(action_str).strip().lower()
    
            symbol_board, reward, terminated, truncated, info_dict, current_perf_score = game_env.step(action_str)

            total_score = max(info_dict["total_score"], total_score)

            output_token_len_list.append(resp.output_len)
            request_time_list.append(time_taken_s)

            traj_dump_str += f"{"-"*80}\nTurn {turn_idx}\n{"-"*80}\nUser Prompt=\n{user_prompt}\nResponse=\n{origin_resp_str}\n"
            traj_dump_str += f"action_str={action_str}, user_tokens_len={user_tokens_len}, respone token len={resp.output_len}, time_taken_s={time_taken_s}\n"
            turn_idx += 1
            if terminated or truncated or turn_idx >= self.max_turns:
                break
        
        ave_output_token_len = sum(output_token_len_list) / len(output_token_len_list)
        ave_request_time = sum(request_time_list) / len(request_time_list)
        print(f"collect one traj for {game_name}: {self.rollout_stat_scope} turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}")
        # Log reward.
        log_dict = {f"total_score_{game_name}": total_score, 
                    f"turns_{game_name}": turn_idx, 
                    f"ave_output_token_len_{game_name}": ave_output_token_len, 
                    f"ave_request_time_{game_name}": ave_request_time}
        stats_tracker.get(self.rollout_stat_scope).scalar(**log_dict)
        traj_dump_str += f"Final Results:\n turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}"

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(engine.get_version()))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{game_name}_seed{seed}_level{level_idx}_traj{traj_idx}.txt")
            async with aiofiles.open(file_path, "a") as f:
                await f.write(traj_dump_str)

        return True

    async def arun_one_episode_turn_independent_2048(self, engine: InferenceEngine, data, traj_idx):
        seed = data
        game_env = TwentyFortyEightVariantEnv(size=4, base=2)
        game_env.reset(seed=seed)
        system_prompt = self.system_prompt_template
        
        system_messages = [{"role": "system", "content": system_prompt}]
        system_input_ids = self.tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        system_tokens_len = len(system_input_ids)
        traj_dump_str = f"System Prompt=\n{system_prompt}\nsystem_tokens_len={system_tokens_len}\n"

        turn_idx = 0
        version = engine.get_version()
        traj_reward = 0
        total_score = 0
        reward_list = []
        output_token_len_list = []
        request_time_list = []
        results = []
        while True:
            text_obs = game_env.get_text_obs()
            user_prompt = self.user_prompt_template.format(textual_representation=text_obs)
            user_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            user_input_ids = self.tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            user_tokens_len = len(user_input_ids)

            start_time = time.time()
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=user_input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            assert resp.input_len == user_tokens_len, f"resp.input_len != user_tokens_len, {resp.input_len} != {user_tokens_len}"
            
            resp_str = self.tokenizer.decode(resp.output_tokens)

            time_taken_s = time.time() - start_time

            origin_resp_str = resp_str
            if self.enable_thinking and "</think>" in resp_str:
                resp_str = resp_str.split("</think>")[-1].strip()
            
            action_dict = game_env.get_action_from_response(resp_str.replace("<|im_end|>", ""))
            action_str = None
            if action_dict and action_dict.get("action") is not None:
                action_str = action_dict.get("action")
                action_str = str(action_str).strip().lower()
        
            reward, terminated = game_env.step(action_str)
            total_score = game_env.total_score

            reward_list.append(reward)

            input_ids_list = resp.input_tokens + resp.output_tokens
            loss_mask_list = [0] * resp.input_len + [1] * resp.output_len
            logprobs_list = [0.0] * resp.input_len + resp.output_logprobs
            versions_list = [-1] * resp.input_len + resp.output_versions
            result = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(input_ids_list).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask_list).unsqueeze(0),
                logprobs=torch.tensor(logprobs_list).unsqueeze(0),
                versions=torch.tensor(versions_list).unsqueeze(0),
                attention_mask=torch.ones(len(input_ids_list), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=torch.tensor([float(reward)]),
            )
            results.append(result)
            
            traj_reward += reward
            output_token_len_list.append(resp.output_len)
            request_time_list.append(time_taken_s)

            traj_dump_str += f"{'-'*80}\nTurn {turn_idx}\n{"-"*80}\nUser Prompt=\n{user_prompt}\nResponse=\n{origin_resp_str}\n"
            traj_dump_str += f"action_str={action_str}, user_tokens_len={user_tokens_len}, respone token len={resp.output_len}, reward={reward}\n"

            turn_idx += 1
            if terminated or turn_idx >= self.max_turns:
                break
        
        ave_reward = sum(reward_list) / len(reward_list)
        ave_output_token_len = sum(output_token_len_list) / len(output_token_len_list)
        ave_request_time = sum(request_time_list) / len(request_time_list)
        print(f"collect one traj: {self.rollout_stat_scope} turns={turn_idx} total_score={total_score} traj_reward={traj_reward} ave_step_reward={ave_reward} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}")
        # Log reward.
        stats_tracker.get(self.rollout_stat_scope).scalar(total_score=total_score, turns=turn_idx, traj_reward=traj_reward, ave_step_reward=ave_reward, ave_output_token_len=ave_output_token_len, ave_request_time=ave_request_time)
        traj_dump_str += f"Final Results:\n turns={turn_idx} total_score={total_score} traj_reward={traj_reward} ave_step_reward={ave_reward} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}"

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            if isinstance(data, int):
                qid = data
            else:
                for key in ["query_id", "id", "qid"]:
                    qid = data.get(key, None)
                    if qid is not None:
                        break
                qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}_traj{traj_idx}.txt")
            async with aiofiles.open(file_path, "a") as f:
                await f.write(traj_dump_str)

        return results, float(traj_reward)

    async def arun_one_episode_turn_independent_sokoban(self, engine: InferenceEngine, data, traj_idx):
        seed = data
        game_env = SokobanVariantEnv(map_file_path=self.sokoban_level_txt_file_path)
        level_idx = random.randint(1, 6)
        game_env.reset(level_idx=level_idx)
        system_prompt = self.system_prompt_template
        
        system_messages = [{"role": "system", "content": system_prompt}]
        system_input_ids = self.tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        system_tokens_len = len(system_input_ids)
        traj_dump_str = f"System Prompt=\n{system_prompt}\nsystem_tokens_len={system_tokens_len}\n"

        turn_idx = 0
        version = engine.get_version()
        traj_reward = 0
        total_score = 0
        reward_list = []
        output_token_len_list = []
        request_time_list = []
        results = []
        while True:
            text_obs = game_env.get_text_obs()
            user_prompt = self.user_prompt_template.format(textual_representation=text_obs)
            user_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            user_input_ids = self.tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            user_tokens_len = len(user_input_ids)

            start_time = time.time()
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=user_input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            assert resp.input_len == user_tokens_len, f"resp.input_len != user_tokens_len, {resp.input_len} != {user_tokens_len}"
            
            resp_str = self.tokenizer.decode(resp.output_tokens)

            time_taken_s = time.time() - start_time

            origin_resp_str = resp_str
            if self.enable_thinking and "</think>" in resp_str:
                resp_str = resp_str.split("</think>")[-1].strip()
            
            action_dict = game_env.get_action_from_response(resp_str.replace("<|im_end|>", ""))
            action_str = None
            if action_dict and action_dict.get("action") is not None:
                action_str = action_dict.get("action")
                action_str = str(action_str).strip().lower()
        
            symbol_board, reward, terminated, truncated, info_dict, current_perf_score = game_env.step(action_str)
            if action_str is None or action_str not in ['up', 'down', 'left', 'right', 'push up', 'push down', 'push left', 'push right']:
                reward = -2
            total_score = max(info_dict["total_score"], total_score)

            reward_list.append(reward)

            input_ids_list = resp.input_tokens + resp.output_tokens
            loss_mask_list = [0] * resp.input_len + [1] * resp.output_len
            logprobs_list = [0.0] * resp.input_len + resp.output_logprobs
            versions_list = [-1] * resp.input_len + resp.output_versions
            result = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(input_ids_list).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask_list).unsqueeze(0),
                logprobs=torch.tensor(logprobs_list).unsqueeze(0),
                versions=torch.tensor(versions_list).unsqueeze(0),
                attention_mask=torch.ones(len(input_ids_list), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=torch.tensor([float(reward)]),
            )
            results.append(result)
            
            traj_reward += reward
            output_token_len_list.append(resp.output_len)
            request_time_list.append(time_taken_s)

            traj_dump_str += f"{'-'*80}\nTurn {turn_idx}\n{"-"*80}\nUser Prompt=\n{user_prompt}\nResponse=\n{origin_resp_str}\n"
            traj_dump_str += f"action_str={action_str}, user_tokens_len={user_tokens_len}, respone token len={resp.output_len}, reward={reward}\n"

            turn_idx += 1
            if terminated or truncated or turn_idx >= self.max_turns:
                break
        
        ave_reward = sum(reward_list) / len(reward_list)
        ave_output_token_len = sum(output_token_len_list) / len(output_token_len_list)
        ave_request_time = sum(request_time_list) / len(request_time_list)
        print(f"collect one traj: {self.rollout_stat_scope} turns={turn_idx} total_score={total_score} traj_reward={traj_reward} ave_step_reward={ave_reward} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}")
        # Log reward.
        stats_tracker.get(self.rollout_stat_scope).scalar(total_score=total_score, turns=turn_idx, traj_reward=traj_reward, ave_step_reward=ave_reward, ave_output_token_len=ave_output_token_len, ave_request_time=ave_request_time)
        traj_dump_str += f"Final Results:\n turns={turn_idx} total_score={total_score} traj_reward={traj_reward} ave_step_reward={ave_reward} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}"

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            if isinstance(data, int):
                qid = data
            else:
                for key in ["query_id", "id", "qid"]:
                    qid = data.get(key, None)
                    if qid is not None:
                        break
                qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}_traj{traj_idx}.txt")
            async with aiofiles.open(file_path, "a") as f:
                await f.write(traj_dump_str)

        return results, float(traj_reward)

    async def arun_eval_rollout_2048(self, engine: InferenceEngine, data, traj_idx):
        if self.fix_test_env_seed:
            seed = data
        else:
            seed = secrets.randbelow(2**31)
        game_env = TwentyFortyEightVariantEnv(size=4, base=2)
        game_env.reset(seed=seed)
        system_prompt = self.system_prompt_template
        
        system_messages = [{"role": "system", "content": system_prompt}]
        system_input_ids = self.tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        system_tokens_len = len(system_input_ids)
        traj_dump_str = f"System Prompt=\n{system_prompt}\nsystem_tokens_len={system_tokens_len}\n"

        turn_idx = 0
        total_score = 0
        output_token_len_list = []
        request_time_list = []
        while True:
            text_obs = game_env.get_text_obs()
            user_prompt = self.user_prompt_template.format(textual_representation=text_obs)

            user_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            user_input_ids = self.tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            user_tokens_len = len(user_input_ids)

            start_time = time.time()
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=user_input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            assert resp.input_len == user_tokens_len, f"resp.input_len != user_tokens_len, {resp.input_len} != {user_tokens_len}"
            
            resp_str = self.tokenizer.decode(resp.output_tokens)

            origin_resp_str = resp_str
            if self.enable_thinking and "</think>" in resp_str:
                resp_str = resp_str.split("</think>")[-1].strip()
            
            action_dict = game_env.get_action_from_response(resp_str.replace("<|im_end|>", ""))
            
            end_time = time.time()
            time_taken_s = end_time - start_time
            action_str = None
            if action_dict and action_dict.get("action") is not None:
                action_str = action_dict.get("action")
                action_str = str(action_str).strip().lower()
    
            reward, terminated = game_env.step(action_str)
            total_score = game_env.total_score

            output_token_len_list.append(resp.output_len)
            request_time_list.append(time_taken_s)

            traj_dump_str += f"{"-"*80}\nTurn {turn_idx}\n{"-"*80}\nUser Prompt=\n{user_prompt}\nResponse=\n{origin_resp_str}\n"
            traj_dump_str += f"action_str={action_str}, user_tokens_len={user_tokens_len}, respone token len={resp.output_len}, time_taken_s={time_taken_s}\n"
            turn_idx += 1
            if terminated or turn_idx >= self.max_turns:
                break
        
        ave_output_token_len = sum(output_token_len_list) / len(output_token_len_list)
        ave_request_time = sum(request_time_list) / len(request_time_list)
        print(f"collect one traj: {self.rollout_stat_scope} turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}")
        # Log reward.
        stats_tracker.get(self.rollout_stat_scope).scalar(total_score=total_score, turns=turn_idx, ave_output_token_len=ave_output_token_len, ave_request_time=ave_request_time)
        traj_dump_str += f"Final Results:\n turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}"

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(engine.get_version()))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"seed{seed}_traj{traj_idx}.txt")
            async with aiofiles.open(file_path, "a") as f:
                await f.write(traj_dump_str)

        return True


    async def arun_eval_rollout_sokoban(self, engine: InferenceEngine, data, traj_idx):
        if self.fix_test_env_seed:
            seed = data
        else:
            seed = secrets.randbelow(2**31)
        level_idx = random.randint(1, 6)
        game_env = SokobanVariantEnv(map_file_path=self.sokoban_level_txt_file_path)
        game_env.reset(level_idx=level_idx)
        system_prompt = self.system_prompt_template
        
        system_messages = [{"role": "system", "content": system_prompt}]
        system_input_ids = self.tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        system_tokens_len = len(system_input_ids)
        traj_dump_str = f"System Prompt=\n{system_prompt}\nsystem_tokens_len={system_tokens_len}\n"

        turn_idx = 0
        total_score = 0
        output_token_len_list = []
        request_time_list = []
        while True:
            text_obs = game_env.get_text_obs()
            user_prompt = self.user_prompt_template.format(textual_representation=text_obs)

            user_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            user_input_ids = self.tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            user_tokens_len = len(user_input_ids)

            start_time = time.time()
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=user_input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            assert resp.input_len == user_tokens_len, f"resp.input_len != user_tokens_len, {resp.input_len} != {user_tokens_len}"
            
            resp_str = self.tokenizer.decode(resp.output_tokens)

            origin_resp_str = resp_str
            if self.enable_thinking and "</think>" in resp_str:
                resp_str = resp_str.split("</think>")[-1].strip()

            action_dict = game_env.get_action_from_response(resp_str.replace("<|im_end|>", ""))
            
            end_time = time.time()
            time_taken_s = end_time - start_time
            action_str = None
            if action_dict and action_dict.get("action") is not None:
                action_str = action_dict.get("action")
                action_str = str(action_str).strip().lower()

            symbol_board, reward, terminated, truncated, info_dict, current_perf_score = game_env.step(action_str)
            total_score = max(info_dict["total_score"], total_score)

            output_token_len_list.append(resp.output_len)
            request_time_list.append(time_taken_s)

            traj_dump_str += f"{"-"*80}\nTurn {turn_idx}\n{"-"*80}\nUser Prompt=\n{user_prompt}\nResponse=\n{origin_resp_str}\n"
            traj_dump_str += f"action_str={action_str}, user_tokens_len={user_tokens_len}, respone token len={resp.output_len}, time_taken_s={time_taken_s}\n"
            turn_idx += 1
            if terminated or truncated or turn_idx >= self.max_turns:
                break
        
        ave_output_token_len = sum(output_token_len_list) / len(output_token_len_list)
        ave_request_time = sum(request_time_list) / len(request_time_list)
        print(f"collect one traj: {self.rollout_stat_scope} turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}")
        # Log reward.
        stats_tracker.get(self.rollout_stat_scope).scalar(total_score=total_score, turns=turn_idx, ave_output_token_len=ave_output_token_len, ave_request_time=ave_request_time)
        traj_dump_str += f"Final Results:\n turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}"

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(engine.get_version()))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"seed{seed}_level{level_idx}_traj{traj_idx}.txt")
            async with aiofiles.open(file_path, "a") as f:
                await f.write(traj_dump_str)

        return True

    async def arun_episode(self, engine: InferenceEngine, data):
        if isinstance(data, dict):
            temp_input_ids = [1] * 10
            temp_result = dict(
                input_ids=torch.tensor(temp_input_ids).unsqueeze(0),
                loss_mask=torch.tensor(temp_input_ids).unsqueeze(0),
                attention_mask=torch.ones(len(temp_input_ids), dtype=torch.bool).unsqueeze(0),
            )
            if "eval_rollout" in data:
                result = await asyncio.gather(
                    *[
                        self.gamename_to_runevalrollout_func[self.game_name](
                            engine=engine,
                            data=data["seed"] + self.rank * self.eval_trajs_per_gpu,
                            traj_idx=i,
                        )
                        for i in range(self.n_trajs)
                    ]
                )
                return temp_result
            elif "variant_game_name" in data:
                result = await asyncio.gather(
                    *[
                        self.gamename_to_variant_game_runeval_func[self.game_name](
                            engine=engine,
                            game_name=data["variant_game_name"],
                            data=data["seed"] + self.rank * self.eval_trajs_per_gpu,
                            traj_idx=i,
                        )
                        for i in range(self.n_trajs)
                    ]
                )
                return temp_result


        if data is None:
            data = secrets.randbelow(2**31)

        while len(self.turn_sample_pool) < self.train_turns_per_gpu:
            traj_turn_list_with_trajreward = await asyncio.gather(
                *[
                    self.gamename_to_runepisode_func[self.game_name](
                        engine=engine,
                        data=data,
                        traj_idx=i,
                    )
                    for i in range(self.n_trajs)
                ]
            )
            
            traj_turn_lists = [traj[0] for traj in traj_turn_list_with_trajreward]
            trajrewards = [traj[1] for traj in traj_turn_list_with_trajreward]
            trajrewards_normalized = group_normalization(trajrewards)
            for i in range(len(traj_turn_lists)):
                for j in range(len(traj_turn_lists[i])):
                    traj_turn_lists[i][j]["rewards"] = torch.tensor([float(trajrewards_normalized[i])], device=traj_turn_lists[i][j]["rewards"].device)
                    self.turn_sample_pool.append(traj_turn_lists[i][j])
            random.shuffle(self.turn_sample_pool)
            print(f"turn_sample_pool cnt = {len(self.turn_sample_pool)}")
        
        trajs = self.turn_sample_pool[:self.train_turns_per_gpu]
        self.turn_sample_pool = self.turn_sample_pool[self.train_turns_per_gpu:]
        
        batch = concat_padded_tensors(trajs)
        return batch

            