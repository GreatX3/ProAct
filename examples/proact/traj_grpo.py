import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field

import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    cycle_dataloader,
    tensor_container_to,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.traj_grpo_workflow import TrajGRPOWorkflow


@dataclass
class TrajGRPOConfig(GRPOConfig):
    max_turns: int = field(
        default=32,
        metadata={
            "help": "Maximum number of turns per trajectory. By default max_turns=32."
        },
    )
    eval_steps: int = field(
        default=10,
        metadata={
            "help": "Evaluation steps interval."
        },
    )
    save_steps: int = field(
        default=500,
        metadata={
            "help": "Save steps interval."
        },
    )
    eval_trajs_per_gpu: int = field(
        default=96,
        metadata={
            "help": "The number of evaluation trajectories in every actor gpu."
        },
    )
    game_name: str | None = field(
        default=None,
        metadata={
            "help": "Game name."
        },
    )
    enable_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether enable thinking mode."
        },
    )
    test_variant_games: str | None = field(
        default=None,
        metadata={
            "help": "The name of the variant game during evaluation, split by comma."
        },
    )
    variant_games_prompt_template_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to prompt template for test variant games."
        },
    )
    train_turns_per_gpu: int = field(
        default=1,
        metadata={
            "help": "The number of trained turns in every training step on every actor gpu."
        },
    )
    fix_test_env_seed: bool = field(
        default=False,
        metadata={
            "help": "Whether fix the env seed during evaluation."
        },
    )

def main(args):
    config, _ = load_expr_config(args, TrajGRPOConfig)
    config: TrajGRPOConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=config.save_steps * 1, # len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=1,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = TrajGRPOWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=config.enable_thinking,
        rollout_stat_scope="rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        max_turns=config.max_turns,
        is_eval=False,
        game_name=config.game_name,
        variant_games_prompt_template_path=config.variant_games_prompt_template_path,
        train_turns_per_gpu=config.train_turns_per_gpu,
    )
    eval_workflow = TrajGRPOWorkflow(
        gconfig=config.gconfig.new(temperature=0.6, max_new_tokens=12800),
        tokenizer=tokenizer,
        enable_thinking=config.enable_thinking,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
        max_turns=config.max_turns,
        is_eval=True,
        game_name=config.game_name,
        variant_games_prompt_template_path=config.variant_games_prompt_template_path,
        rank=actor.data_parallel_rank,
        fix_test_env_seed=config.fix_test_env_seed,
        eval_trajs_per_gpu=config.eval_trajs_per_gpu,
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    start_step = 0

    steps_per_epoch = config.save_steps
    max_steps = 10000

    for global_step in range(start_step, max_steps):

        print(f"Start global_step {global_step} | max_steps {max_steps}")

        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch

        with stats_tracker.record_timing("eval_rollout"):
            if global_step % config.eval_steps == 0:
                if actor.is_data_parallel_head():
                    for i in range(config.eval_trajs_per_gpu):
                        eval_rollout.submit({"eval_rollout": 1, "seed": i}, eval_workflow)
                    eval_rollout.wait(config.eval_trajs_per_gpu, timeout=None)
        
        if config.test_variant_games:
            test_variant_games = config.test_variant_games.replace(" ", "")
            for variant_game_name in test_variant_games.split(","):
                with stats_tracker.record_timing(f"eval_rollout_{variant_game_name}"):
                    if global_step % config.eval_steps == 0:
                        if actor.is_data_parallel_head():
                            for i in range(config.eval_trajs_per_gpu):
                                eval_rollout.submit({"variant_game_name": variant_game_name, "seed": i}, eval_workflow)
                            eval_rollout.wait(config.eval_trajs_per_gpu, timeout=None)
        
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("rollout"):
            batch = None
            if actor.is_data_parallel_head():
                if config.async_training:
                    batch = rollout.prepare_batch(
                        # train_dataloader,
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                else:
                    batch = rollout.rollout_batch(
                        # next(data_generator),
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                batch = tensor_container_to(batch, actor.device)

            print(f"rollout batch info: {[(k, v.shape) for k, v in batch.items()]}")

            batch = broadcast_tensor_container(
                batch,
                src_rank=actor.current_data_parallel_head(),
                group=actor.context_and_model_parallel_group,
            )
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")
        
        print(f"train batch info: {[(k, v.shape) for k, v in batch.items()]}")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(
            stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        )

        stats_minibatch_ave = {}
        for key in stats[0]:
            value_list = []
            for stat_each_minibath in stats:
                if key in stat_each_minibath:
                    value_list.append(stat_each_minibath[key])
            stats_minibatch_ave[key] = sum(value_list) / len(value_list)

        stats_logger.commit(epoch, step, global_step, [stats_minibatch_ave])

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
