import pickle
import os
import numpy as np

import os
import openai
import time
import random
import json
import concurrent.futures
import re
from envs.sokobanEnv import SokobanVariantEnv
import fcntl


def get_prompt_template(env_name):
    if env_name == "extra":
        env_name = "origin"
    with open("envs/prompt_sokoban.json", 'r') as f:
        prompt_template = json.load(f)
    return prompt_template[env_name]["system_prompt_template"], prompt_template[env_name]["user_prompt_template"]

def ask_llm(system_prompt, user_prompt, temperature=0.6):
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
    ]
    openai.api_key = "EMPTY"
    openai.base_url = f"http://127.0.0.1:8080/v1/"
    start = time.time()
    response = openai.chat.completions.create(
        model="ProAct",
        messages=messages,
        temperature=temperature,
    )
    req_time =  time.time() - start
    resp_text = response.choices[0].message.content
    output_tokens = response.usage.completion_tokens

    return resp_text, req_time, output_tokens


def process_single_item(env_name, episode_id, save_dir):
    os.makedirs(f"{save_dir}/{env_name}", exist_ok=True)
    totalscore_save_path = f"{save_dir}/{env_name}/score_summary.txt"

    map_file_path = 'envs/levels_random.txt'
    game_env = SokobanVariantEnv(map_file_path=map_file_path, variant=env_name, max_steps_episode=200)
    start_level_idx = 1
    end_level_idx = 6
    if env_name == "extra":
        start_level_idx = 7
        end_level_idx = 8
    total_score_list = []
    for level_idx in range(start_level_idx, end_level_idx + 1):
        traj_save_path = f"{save_dir}/{env_name}/traj_{episode_id}_level_{level_idx}.txt"
        game_env.reset(level_idx=level_idx)

        system_prompt_template, user_prompt_template = get_prompt_template(env_name)

        system_prompt = system_prompt_template

        traj_dump_str = f"System Prompt=\n{system_prompt}\n"
        with open(traj_save_path, "a") as f:
            f.write(traj_dump_str)

        turn_idx = 0
        output_token_len_list = []
        request_time_list = []
        total_score = 0
        while True:
            text_obs = game_env.get_text_obs()
            user_prompt = user_prompt_template.format(textual_representation=text_obs)

            while True:
                try:
                    resp_str, req_time, output_tokens = ask_llm(system_prompt, user_prompt)
                    break
                except Exception as e:
                    print(e)

            origin_resp_str = resp_str
            action_dict = game_env.get_action_from_response(resp_str.replace("<|im_end|>", ""))
            
            action_str = None
            if action_dict and action_dict.get("action") is not None:
                action_str = action_dict.get("action")
                action_str = str(action_str).strip().lower()

            symbol_board, reward, terminated, truncated, info_dict, current_perf_score = game_env.step(action_str)

            total_score = max(total_score, info_dict["total_score"])

            output_token_len_list.append(output_tokens)
            request_time_list.append(req_time)

            traj_dump_str = f"{'-'*80}\nTurn {turn_idx}\n{'-'*80}\nUser Prompt=\n{user_prompt}\nResponse=\n{origin_resp_str}\n"
            traj_dump_str += f"action_str={action_str}, respone token len={output_tokens}, time_taken_s={req_time} total_score={total_score}\n"
            with open(traj_save_path, "a") as f:
                f.write(traj_dump_str)
            turn_idx += 1
            if terminated or truncated:
                break
        total_score_list.append(total_score)
        ave_output_token_len = sum(output_token_len_list) / max(len(output_token_len_list), 1)
        ave_request_time = sum(request_time_list) / len(request_time_list)
        print(f"collect one traj for {env_name}: turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}")

        traj_dump_str = f"Final Results:\n turns={turn_idx} total_score={total_score} ave_output_token_len={ave_output_token_len} ave_request_time={ave_request_time}"
        with open(traj_save_path, "a") as f:
            f.write(traj_dump_str)
        
        with open(totalscore_save_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(f"episode {episode_id} level {level_idx} total_score {total_score}\n")
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    return env_name, np.mean(total_score_list)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_num_episode", type=int, default=40)
    parser.add_argument("--save_dir", type=str, default="./results")

    args = parser.parse_args()
    test_num_episode = args.test_num_episode
    env_name_list = ["origin", "action", "symbol", "extra"]  
    save_dir = args.save_dir
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for episode_id in range(test_num_episode):
            for env_name in env_name_list:
                future = executor.submit(process_single_item, env_name, episode_id, save_dir)
                futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    success_count = len(results)
    print(f"Processing completed. Success: {success_count}")

    result_dict = {}
    for result in results:
        env_name, total_score = result
        if env_name not in result_dict:
            result_dict[env_name] = []
        result_dict[env_name].append(total_score)
    
    for env_name in env_name_list:
        assert len(result_dict[env_name]) == test_num_episode
        ave_score = sum(result_dict[env_name]) / len(result_dict[env_name])
        print(f"env_name {env_name} ave_score {ave_score}")
        totalscore_save_path = f"{save_dir}/{env_name}/score_summary.txt"
        with open(totalscore_save_path, "a") as f:
            f.write(f"ave_score {ave_score}\n")

if __name__ == "__main__":
    main()
        
