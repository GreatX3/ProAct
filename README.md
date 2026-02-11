<h1 align="center">
ProAct: Agentic Lookahead in Interactive Environments
</h1>

<p align="center">
| <a href="https://arxiv.org/abs/2602.05327"><b>Paper</b></a> | <a href="https://huggingface.co/biang889/ProAct"><b>🤗 Models</b></a> | 
</p>

## 📖 Introduction

We introduce **ProAct**, a novel framework designed to empower Large Language Model (LLM) agents with accurate, multi-turn lookahead reasoning capabilities in interactive environments. Existing agents often struggle with long-horizon planning due to "simulation drift," where minor internal prediction errors accumulate over time.

**ProAct addresses this challenge through a two-stage paradigm:**

1. **Grounded LookAhead Distillation (GLAD):** We utilize Monte-Carlo Tree Search (MCTS) to explore the environment and distill complex search trees into concise, causal reasoning chains. This allows the model to "internalize" the foresight of search algorithms via Supervised Fine-Tuning (SFT).
2. **Monte-Carlo Critic (MC-Critic):** To further refine the policy, we propose a plug-and-play auxiliary value estimator that leverages lightweight environment rollouts. This provides low-variance value estimates, significantly stabilizing multi-turn agentic Reinforcement Learning (RL) algorithms.

## 🎥 Demos

We compare **ProAct** against leading closed-source models and open-source models on two  long-horizon interactive environments (i.e., 2048 and Sokoban). ProAct exhibits superior foresight and strategic planning capabilities in interactive environments.

- Visualizations on 2048

  <a href="https://github.com/GreatX3/ProAct/raw/main/demos/2048_demo.mp4">
    <img src="demos/2048_demo.png" width="600" alt="Watch the demo">
  </a>

- Visualizations on Sokoban

  <a href="https://github.com/GreatX3/ProAct/raw/main/demos/sokoban_demo.mp4">
    <img src="demos/sokoban_demo.png" width="600" alt="Watch the demo">
  </a>

## 🚀 Inference & Testing

The testing pipeline consists of three steps: downloading the model, deploying it with vLLM, and running the evaluation script.

### 1. Download the Model

Download the pre-trained model from [Hugging Face](https://huggingface.co/biang889/ProAct).

```bash
huggingface-cli download --resume-download biang889/ProAct --local-dir /path/to/ProAct 
```

### 2. Deploy Model using vLLM

Start the vLLM server to serve the model compatible with OpenAI API protocol.

```bash
vllm serve /path/to/ProAct/2048_rl \  # 2048_stf, sokoban_rl, sokoban_sft
  --served-model-name ProAct \
  --host 0.0.0.0 \
  --port 8080 \
  --tensor-parallel-size 1
```

> **Note:** Keep this terminal open while running the tests in a separate terminal.

### 3. Run Testing Script

Once the server is up and running, execute the testing script in a **new terminal window**. This script will interact with the deployed model via the API and generate results.

```bash
# test on 2048
python3 test_proact_on_2048.py
# test on Sokoban
python3 test_proact_on_sokoban.py
```

## 🏋️ Training

Our training framework is developed based on [AReaL](https://github.com/inclusionAI/AReaL). Before running the training scripts, please follow the [AReaL Installation Guide](https://inclusionai.github.io/AReaL/tutorial/installation.html) to set up the environment and dependencies. All experiments reported in the paper were conducted on a node equipped with 8 NVIDIA H20 GPUs.

The training process of ProAct operates in a two-stage paradigm: **Supervised Fine-Tuning (SFT)** to internalize lookahead reasoning, followed by **Reinforcement Learning (RL)** to refine the policy for long-horizon optimization.

### Stage 1: Supervised Fine-Tuning (SFT) with GLAD

In the first stage, we employ **Grounded LookAhead Distillation (GLAD)**. Instead of training on raw, delusional hallucinations, we use the environment as an oracle to construct a high-quality dataset. To run SFT training on 2048 and sokoban: 

```bash
# 2048
python3 -m areal.launcher.local examples/proact/sft.py --config examples/proact/sft_2048.yaml
# sokoban
python3 -m areal.launcher.local examples/proact/sft.py --config examples/proact/sft_sokoban.yaml
```

### Stage 2: Reinforcement Learning (RL) with MC-Critic

In the second stage, we further align the model using Online RL. To address the high variance of value estimation in long-horizon tasks, we introduce the **Monte-Carlo Critic (MC-Critic)**. As reported in the paper, we implement five multi-turn agentic reinforcement learning algorithms, namely Traj-GRPO, Step-GRPO, MC-GRPO, Step-PPO, and MC-PPO. To run RL training on 2048 and sokoban: 

```bash
# Traj-GRPO on 2048
python3 -m areal.launcher.local examples/proact/traj_grpo.py --config examples/proact/traj_grpo_2048.yaml
# Step-GRPO on 2048
python3 -m areal.launcher.local examples/proact/step_grpo.py --config examples/proact/step_grpo_2048.yaml
# MC-GRPO on 2048
python3 -m areal.launcher.local examples/proact/step_grpo.py --config examples/proact/mc_grpo_2048.yaml
# Step-PPO on 2048
python3 -m areal.launcher.local examples/proact/step_ppo.py --config examples/proact/step_ppo_2048.yaml
# MC-PPO on 2048
python3 -m areal.launcher.local examples/proact/step_ppo.py --config examples/proact/mc_ppo_2048.yaml
# Traj-GRPO on sokoban
python3 -m areal.launcher.local examples/proact/traj_grpo.py --config examples/proact/traj_grpo_sokoban.yaml
# Step-GRPO on sokoban
python3 -m areal.launcher.local examples/proact/step_grpo.py --config examples/proact/step_grpo_sokoban.yaml
# MC-GRPO on sokoban
python3 -m areal.launcher.local examples/proact/step_grpo.py --config examples/proact/mc_grpo_sokoban.yaml
# Step-PPO on sokoban
python3 -m areal.launcher.local examples/proact/step_ppo.py --config examples/proact/step_ppo_sokoban.yaml
# MC-PPO on sokoban
python3 -m areal.launcher.local examples/proact/step_ppo.py --config examples/proact/mc_ppo_sokoban.yaml
```

## 📜 Citation

If you find this project useful in your research, please cite our paper:

```<BIBTEX>
@misc{yu2026proactagenticlookaheadinteractive,
      title={ProAct: Agentic Lookahead in Interactive Environments}, 
      author={Yangbin Yu and Mingyu Yang and Junyou Li and Yiming Gao and Feiyu Liu and Yijun Yang and Zichuan Lin and Jiafei Lyu and Yicheng Liu and Zhicong Lu and Deheng Ye and Jie Jiang},
      year={2026},
      eprint={2602.05327},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.05327}, 
}
```
