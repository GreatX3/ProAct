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

def message_to_number(messages):
    """
    Convert the list of messages to a deterministic integer by hashing the JSON representation.
    """
    import json
    msg_str = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    md5_hash = hashlib.md5(msg_str.encode("utf-8")).hexdigest()
    # Convert hash to integer (take the low 8 digits for use as a seed/int_rid)
    return int(md5_hash[:8], 16)


logger = logging.getLogger("SFT workflow")


def default_get_input_ids_fn(data, tokenizer, enable_thinking):
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return input_ids


def default_data_extract_prompt_fn(data):
    return data["messages"]

def group_normalization(reward_list):
    reward_tensor = torch.tensor(reward_list)
    return ((reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-5)).tolist()

class SFTWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        max_tokens_per_trajectory: int = 32768,
        max_turns: int = 10000,
        is_eval: bool = False,
        game_name: str = "twenty_forty_eight",
        sft_dataset_path: str | None = None,
        rank: int = 0,
        world_size: int = 1,
        variant_games_prompt_template_path: str | None = None,
        fix_test_env_seed: bool = False,
        eval_trajs_per_gpu: int = 1,
    ):
        self.fix_test_env_seed = fix_test_env_seed
        self.eval_trajs_per_gpu = eval_trajs_per_gpu
        self.variant_games_prompt_template_path = variant_games_prompt_template_path
        self.rank = rank
        self.world_size = world_size
        self.sft_dataset_path = sft_dataset_path
        self.game_name = game_name
        self.is_eval = is_eval
        self.max_tokens_per_trajectory = max_tokens_per_trajectory
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

        self.gamename_to_getofflinedata_func = {
            "twenty_forty_eight": self.get_sft_with_pkl_data,
            "sokoban": self.get_sft_with_pkl_data,
        }

        self.preprocess()
        self.train_data_idx = 0

    def preprocess(self):
        if self.sft_dataset_path:
                with open(self.sft_dataset_path, 'rb') as f:
                    self.offline_datas = pickle.load(f)
                self.offline_datas = [e for i, e in enumerate(self.offline_datas) if i % self.world_size == self.rank]
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

    async def get_sft_with_pkl_data(self, engine: InferenceEngine, group_seed):
        start_idx = self.train_data_idx % len(self.offline_datas)
        end_idx = (self.train_data_idx + self.n_trajs) % len(self.offline_datas)
        if end_idx >= start_idx:
            trajs = self.offline_datas[start_idx:end_idx]
        else:
            trajs = self.offline_datas[start_idx:] + self.offline_datas[:end_idx]
        self.train_data_idx += self.n_trajs

        results = []
        for traj_idx, traj in enumerate(trajs):

            while True:
                try:
                    if not ("system_prompt" in traj and "user_prompt" in traj and  "ast_prompt" in traj):
                        traj = random.choice(self.offline_datas)
                        continue
                    messages = [
                        {"role": "system", "content": traj["system_prompt"]}, 
                        {"role": "user", "content": traj["user_prompt"]}, 
                        {"role": "assistant", "content": traj["ast_prompt"]}
                    ]
                    input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        enable_thinking=self.enable_thinking,
                    )
                    if len(input_ids) >= self.max_tokens_per_trajectory:
                        print(f"traj={traj} input_ids length {len(input_ids)} is longer than {self.max_tokens_per_trajectory}")
                        traj = random.choice(self.offline_datas)
                        continue
                    break
                except Exception as e:
                    print(f"failed to load data with error: {e}")
                    traj = random.choice(self.offline_datas)
                    continue

            system_messages = [{"role": "system", "content": traj["system_prompt"]}]

            traj_dump_str = f"System Prompt=\n{traj['system_prompt']}\n"

            system_input_ids = self.tokenizer.apply_chat_template(
                system_messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
            )
            system_tokens_len = len(system_input_ids)

            input_ids_list = system_input_ids.copy()
            loss_mask_list = [0] * system_tokens_len

            version = engine.get_version()

            user_prompt = traj['user_prompt']
            ast_prompt = traj['ast_prompt']
            user_ast_messages = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": ast_prompt}]
            
            user_ast_input_ids = self.tokenizer.apply_chat_template(
                user_ast_messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
            )
            user_ast_tokens_len = len(user_ast_input_ids)

            input_ids_list += user_ast_input_ids

            user_messages = [{"role": "user", "content": user_prompt}]
            user_input_ids = self.tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            user_tokens_len = len(user_input_ids)
            resp_len = user_ast_tokens_len - user_tokens_len

            assert user_input_ids == user_ast_input_ids[:user_tokens_len], f"{user_input_ids} != {user_ast_input_ids[:user_tokens_len]}"

            loss_mask_list += [0] * user_tokens_len + [1] * resp_len
            # versions_list += [-1] * user_tokens_len + [version] * resp_len
            
            traj_dump_str += f"User Prompt=\n{user_prompt}\nResponse=\n{ast_prompt}\n"

            result = dict(
                input_ids=torch.tensor(input_ids_list).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask_list).unsqueeze(0),
                attention_mask=torch.ones(len(input_ids_list), dtype=torch.bool).unsqueeze(0),
            )
            
            results.append(result)

            if self.dump_dir is not None:
                dump_path = os.path.join(self.dump_dir, str(version))
                await aiofiles.os.makedirs(dump_path, exist_ok=True)
                file_path = os.path.join(dump_path, f"{group_seed}_traj{traj_idx}.txt")
                async with aiofiles.open(file_path, "a") as f:
                    await f.write(traj_dump_str)

        return results
    
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
        
        trajs = await self.gamename_to_getofflinedata_func[self.game_name](engine=engine, group_seed=data)

        batch = concat_padded_tensors(trajs)
        return batch

            