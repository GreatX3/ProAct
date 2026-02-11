import functools
from typing import Any, Dict, List, Optional

import torch

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils import stats_tracker
from areal.utils.data import (
    KLEstimator,
    Normalization,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import (
    dynamic_sampling,
    gather_logprobs,
    gather_logprobs_entropy,
    ppo_actor_loss_fn,
    reward_overlong_penalty,
)


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.group_size = config.group_size

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature
        self.dynamic_sampling = config.dynamic_sampling

    @torch.no_grad()
    def compute_logp(
        self,
        data: Dict[str, Any],
        temperature: Optional[float] = None,
    ) -> torch.Tensor | None:
        def calc_logprobs(logits, input_data):
            labels = torch.roll(input_data["input_ids"], shifts=-1, dims=-1)
            logprobs = gather_logprobs(logits, labels, temperature or 1.0)
            return logprobs

        self.engine.eval()
        return self.engine.forward(
            input_=data,
            post_hook=calc_logprobs,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    def compute_turn_level_advantages(
        self,
        data: Dict[str, Any],
        discount: float | None = None,
        gae_lambda: float | None = None,
        use_mc_critic: bool = False,
        mc_value_weight: float = 0.0,
        train_turns_per_gpu: int = 1,
    ) -> Dict[str, torch.Tensor]:
        values = data.get("values")
        rewards = data.get("rewards")
        loss_mask = data.get("loss_mask")
        attention_mask = data.get("attention_mask")
        terminated = data.get("terminated")
        if use_mc_critic:
            mc_values = data.get("mc_values")
        
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        if values is None:
            raise ValueError("values must be provided in data")
        if rewards is None:
            raise ValueError("rewards must be provided in data")
        if loss_mask is None:
            loss_mask = attention_mask
        if loss_mask is None:
            raise ValueError("loss_mask or attention_mask must be provided in data")

        num_turns, max_seqlen = values.shape
        assert num_turns == train_turns_per_gpu + 1
        device = values.device
        dtype = values.dtype
        
        if rewards.ndim != 1 or rewards.shape[0] != num_turns:
            raise ValueError(
                f"rewards should be 1D tensor with shape [num_turns], "
                f"got shape {rewards.shape}, expected [{num_turns}]"
            )

        discount = discount if discount is not None else self.discount
        gae_lambda = gae_lambda if gae_lambda is not None else self.gae_lambda

        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        print(f"reward_score before norm: shape={reward_score.shape}, max={reward_score.max()}, min={reward_score.min()}, mean={reward_score.mean()}")

        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)

        print(f"reward_score after norm: shape={reward_score.shape}, max={reward_score.max()}, min={reward_score.min()}, mean={reward_score.mean()}, reward={reward_score}")

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            old_logp = data["logprobs"] = data["prox_logp"]
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp", torch.zeros_like(old_logp))
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score


        turn_advantages = torch.zeros(train_turns_per_gpu + 1, dtype=dtype, device=device)
        lastgaelam = 0.0
        

        for turn_idx in reversed(range(train_turns_per_gpu)):
            turn_reward = reward_score[turn_idx]
            if not use_mc_critic:
                next_turn_value = values[turn_idx + 1, indices[turn_idx + 1]]
            else:
                next_turn_value = (1 - mc_value_weight) * values[turn_idx + 1, indices[turn_idx + 1]] + mc_value_weight * mc_values[turn_idx + 1]
        
            if not use_mc_critic:
                delta = turn_reward + discount * (1 - terminated[turn_idx]) * next_turn_value - values[turn_idx, indices[turn_idx]]
            else:
                delta = turn_reward + discount * (1 - terminated[turn_idx]) * next_turn_value - ((1 - mc_value_weight) * values[turn_idx, indices[turn_idx]] + mc_value_weight * mc_values[turn_idx])
            
            turn_advantages[turn_idx] = delta + discount * gae_lambda * (1 - terminated[turn_idx]) * lastgaelam
            lastgaelam = turn_advantages[turn_idx]
        
        if not use_mc_critic:
            turn_returns = turn_advantages + values[batch_indices, indices]
        else:
            turn_returns = turn_advantages + ((1 - mc_value_weight) * values[batch_indices, indices] + mc_value_weight * mc_values[batch_indices])
        print(f"reward_score:\n{reward_score}\nturn_advantages:\n{turn_advantages}\nturn_returns:\n{turn_returns}")

        print(f"turn_advantages before norm:\n{turn_advantages}")

        if self.adv_norm is not None:
            turn_advantages[:-1] = self.adv_norm(turn_advantages[:-1])

        print(f"turn_advantages after norm:\n{turn_advantages}")

        advantages = turn_advantages.unsqueeze(-1).expand(-1, max_seqlen)
        returns = turn_returns.unsqueeze(-1).expand(-1, max_seqlen)

        if loss_mask is not None:
            loss_mask_float = loss_mask.float()
            advantages = advantages * loss_mask_float
            returns = returns * loss_mask_float
        
        critic_loss_mask = torch.zeros_like(loss_mask, dtype=loss_mask.dtype, device=loss_mask.device)
        critic_loss_mask[batch_indices, indices] = 1
        data["critic_loss_mask"] = critic_loss_mask
        
        data["advantages"] = advantages
        data["returns"] = returns
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = old_logp

    def compute_advantages(self, data: Dict[str, Any]) -> None:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        print(f"reward_score before norm: shape={reward_score.shape}, max={reward_score.max()}, min={reward_score.min()}, mean={reward_score.mean()}, reward={reward_score}")

        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)
        
        print(f"reward_score after norm: shape={reward_score.shape}, max={reward_score.max()}, min={reward_score.min()}, mean={reward_score.mean()}, reward={reward_score}")

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            old_logp = data["logprobs"] = data["prox_logp"]
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp", torch.zeros_like(old_logp))
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

        print(f"advantages before norm: shape={advantages.shape}, max={advantages.max()}, min={advantages.min()}, mean={advantages.mean()}\nadv_max={advantages.max(-1)[0]}, adv_min={advantages.min(-1)[0]}, adv_mean={advantages.mean(-1)}\nadv_last3={advantages[:, -3:]}")

        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)
        
        print(f"advantages after norm: shape={advantages.shape}, max={advantages.max()}, min={advantages.min()}, mean={advantages.mean()}\nadv_max={advantages.max(-1)[0]}, adv_min={advantages.min(-1)[0]}, adv_mean={advantages.mean(-1)}\nadv_last3={advantages[:, -3:]}")

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp

    def ppo_update(self, data: Dict[str, Any]) -> List[Dict[str, float]]:
        if self.dynamic_sampling and len(data["rewards"]) % self.group_size == 0:
            data, sampling_stat = dynamic_sampling(data, self.group_size)

        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        all_stats = []
        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            assert (
                "begin_of_trajectory" in data
            ), "'begin_of_trajectory' is expected to log agent statistics"
            assert (
                len(self.config.log_agent_stats_keys) > 0
            ), "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=torch.ones_like(loss_mask, dtype=torch.bool),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = []
        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.behav_imp_weight_cap is not None:
            scalars["behav_imp_weight_cap"] = self.config.behav_imp_weight_cap
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )

        global_stats = stats_tracker.export(
            reduce_group=self.engine.data_parallel_group
        )
        for k in global_denominators:
            keys = list(global_stats.keys())
            for k2 in keys:
                if k2.endswith(k):
                    global_stats.pop(k2)
        ########## Logging code ends ##########

        for key in ["rewards", "tot_rewards", "kl_rewards", "versions"]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        for epoch_id in range(self.config.n_epoches):
            for mb in mb_inputs.mbs:

                print(f"epoch {epoch_id} mini batch info: {[(k, v.shape) for k, v in mb.items()]}")

                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_loss_fn,
                        temperature=self.temperature,
                        eps_clip=self.config.eps_clip,
                        eps_clip_higher=self.config.eps_clip_higher,
                        c_clip=self.config.c_clip,
                        behav_imp_weight_cap=self.config.behav_imp_weight_cap,
                        loss_type=self.config.loss_type,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)
                all_stats.append(
                    stats_tracker.export(reduce_group=self.engine.data_parallel_group)
                )
        all_stats[0].update(global_stats)
        return all_stats

    def sft_update(self, data):
        self.engine.train()
        return self.engine.train_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )


class FSDPPPOActor(FSDPEngine):
    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> None:
        self.actor.compute_advantages(*args, **kwargs)
    
    @torch.no_grad()
    def compute_turn_level_advantages(self, *args, **kwargs) -> None:
        self.actor.compute_turn_level_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> List[Dict[str, float]]:
        return self.actor.ppo_update(*args, **kwargs)
    
    def sft_update(self, data):
        return self.actor.sft_update(data)


def grpo_loss_fn(
    logits: torch.Tensor,
    input_data: Dict,
    temperature: float,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
    loss_type: str = "ppo",
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    labels = input_data.get(
        "rolled_input_ids",
        torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
    )
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    # Use unsliced/full loss_mask.
    # Ulysses SP will slice loss_mask in ulysses_prepare_inputs().
    loss_mask = input_data.get("full_loss_mask", input_data["loss_mask"]).bool()
    prox_logp = input_data["prox_logp"]

    logprobs, entropy = gather_logprobs_entropy(logits, labels, temperature)
    entropy = entropy.detach()
    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
        c_clip=c_clip,
        proximal_logprobs=prox_logp,
        behav_imp_weight_cap=behav_imp_weight_cap,
        loss_type=loss_type,
    )

    if loss_type == "ppo":
        # Log training statistics
        stats_tracker.denominator(
            n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
            n_valid_tokens=loss_mask.bool(),
            clipped_tokens=stat["clip_mask"],
            dual_clipped_tokens=stat["dual_clip_mask"],
        )

        stats_tracker.stat(
            importance_weight=stat["importance_weight"],
            approx_kl=stat["approx_kl"],
            new_logp=logprobs.detach(),
            old_logp=old_logp,
            entropy=entropy.float(),
            actor_loss=stat["loss"],
            clip_ratio=stat["clip_mask"].float(),
            dual_clip_ratio=stat["dual_clip_mask"].float(),
            denominator="n_valid_tokens",
        )
    else:
        stats_tracker.denominator(
            n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
            n_valid_tokens=loss_mask.bool(),
        )

        stats_tracker.stat(
            approx_kl=stat["approx_kl"],
            new_logp=logprobs.detach(),
            old_logp=old_logp,
            entropy=entropy.float(),
            actor_loss=stat["loss"],
            denominator="n_valid_tokens",
        )

    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    if "clip_mask" in stat:
        clip_mask = stat["clip_mask"]
        clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
        clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
        stats_tracker.stat(
            clipped_new_logp=clipped_new_logp,
            clipped_old_logp=clipped_old_logp,
            denominator="clipped_tokens",
        )
    return loss

def compute_packed_sft_loss(
    logits: torch.Tensor, input_: Dict[str, Any]
) -> torch.Tensor:
    packed_input_ids: torch.Tensor = input_["input_ids"]
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    loss_mask = input_["loss_mask"].bool()

    print(f"sft loss: logits={logits.shape} packed_input_ids={packed_input_ids.shape}")

    logprobs = gather_logprobs(logits, torch.roll(packed_input_ids, shifts=-1, dims=-1))
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
    logprobs = torch.where(loss_mask, logprobs, 0)

    loss = -logprobs.sum() / loss_mask.count_nonzero()
    with torch.no_grad():
        seqlogp = torch.zeros(
            cu_seqlens.shape[0] - 1, device=logits.device, dtype=torch.float64
        )
        for i in range(cu_seqlens.shape[0] - 1):
            m = loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]]
            logp = logprobs[cu_seqlens[i] : cu_seqlens[i + 1]]
            seqlogp[i] = torch.where(m, logp.detach(), 0.0).sum() / (m.count_nonzero())

    ## Loggin stats
    stats_tracker.denominator(
        n_seqs=torch.ones(
            cu_seqlens.shape[0] - 1, dtype=torch.bool, device=logprobs.device
        ),
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=loss_mask,
        prompt_tokens=loss_mask.logical_not(),
    )
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    stats_tracker.stat(loss=-logprobs.detach(), denominator="n_valid_tokens")
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    return loss