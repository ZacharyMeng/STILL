# STILL: Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs

<h5 align="center">
🚀 Welcome to the repo of STILL! 
🎉 STILL has been accepted to NeurIPS’26!


This repo contains the official code for STILL.

If you like our works, please support us with your stars⭐!

[![arXiv](https://img.shields.io/badge/Arxiv-2602.02180-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2602.02180)
</h5>


## Introduction

### Motivation

Large Language Models (LLMs) suffer from quadratic computational complexity ($\mathcal{O}(N^2)$) in standard softmax attention, which limits scalability to long sequences. Existing intra-layer hybrid attention methods for LLM linearization face two critical challenges:

1. **Position-based token routing**: Sliding-window-based token selection fails to capture token-specific global importance, discarding crucial non-local tokens from high-fidelity softmax attention (SA).
2. **Norm distortion in linear attention**: Learnable feature maps in linear attention (LA) distort pretrained feature magnitudes, breaking the "norm-aware" property of pretrained LLMs and degrading performance.

To address these limitations, we propose STILL, an intra-layer hybrid linearization framework that achieves linear complexity while preserving the expressive power of pretrained LLMs.

### Method

STILL introduces three core innovations to enable efficient and effective LLM linearization:

#### 1. Self-Saliency Score for Content-Aware Token Selection

We design a **Self-Saliency Score** with strong local–global consistency to estimate token importance using only sliding-window computation. This score quantifies how much a token relies on its self-attention term, enabling reliable selection of salient tokens for SA (high-fidelity modeling) while routing the rest to LA (efficient summarization) — replacing heuristic position-based routing.

Formally, the Self-Saliency Score is computed by comparing sliding-window attention distributions with/without the diagonal (self-attention) term:

<p align="center">
    <img src="figures/eq1.png" width= "300">
    <br>
</p>

where $W_t$ is the local window index set, $a^{diag}$ and $a^{nodiag}$ are attention distributions with/without self-attention term, and $ϵ$ is a small constant for numerical stability.

#### 2. Norm-Preserved Feature Map (NP-Map)

To preserve pretrained norm statistics, we propose **NP-Map** that decouples feature direction from magnitude and reinjects pretrained norms into linear attention feature maps:

<p align="center">
    <img src="figures/eq2.png" width= "300">
    <br>
</p>

where $f(·)$ is the learnable MLP in feature maps. NP-Map ensures linear attention aligns with the pretrained model’s representational intensity while satisfying non-negativity constraints.

#### 3. Unified Training-Inference Architecture

We adopt **chunk-wise parallelization** and **delayed selection** to improve hardware efficiency:

- Delayed selection: Token routing is performed at chunk granularity (instead of per-token) during decoding, reducing overhead and improving parallelism.
- Chunk-wise parallel form: Sequences are split into chunks for parallel computation of saliency scores and hybrid attention, unifying training and inference logic while maintaining linear complexity.

<p align="center">
    <img src="figures/mainfig.png" width= "700">
    <br>
</p>


### Results

STILL achieves state-of-the-art performance on both standard reasoning and long-context benchmarks, while delivering significant efficiency gains:

#### Key Performance Highlights

- **Reasoning Tasks**: Matches or surpasses the original pretrained LLM on commonsense/general reasoning tasks (PIQA, ARC, HellaSwag, Winogrande), with up to 10.5% gains on MMLU over prior linearized baselines.
- **Long-Context Tasks**: Recovers 86.2% of full-attention performance on long-context benchmarks (RULER, BABILong) — a regime where existing hybrid baselines largely fail.
- **Efficiency**: Cuts required training tokens from 1000+B (scratch training) to 0.04B; achieves 45% average memory reduction and 28% decoding speed-up for sequences over 8K tokens.

## ⚙️ Environment

Prepare your Python environment and install all dependencies:

```bash
# Create and activate conda environment
conda create -n still python=3.10
conda activate still

# Install project dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 📚 Training

Linearizing pre-trained models involves the following steps:

1. **Prepare Base Model**: Place your model weights (e.g., Llama-3.1-8B) into the `./checkpoints/` directory and rename it to `still_llama31_8B_base`

2. **Modify Configuration**: Edit `./checkpoints/still_llama31_8B_base/config.json` and update the `"architectures"` and `"model_type"` fields to match the linearized architecture

3. **Adjust Training Parameters**: Modify the configuration files under `./configs/` as needed (e.g., `still_at_step1.yaml` or `still_ar_step2.yaml`)

4. **Launch Training**: Execute the corresponding training scripts

   ```bash
   sh scripts/still_stage1.sh   # Stage 1
   sh scripts/still_stage2.sh   # Stage 2
   ```
## Acknowledgements

This code is developed on the top of [LoLCATs](https://github.com/HazyResearch/lolcats) and [Liger](https://github.com/OpenSparseLLMs/Linearization).
## Citation

If you find this repo helpful, please consider citing us.

```latex
@article{meng2026still,
  title={STILL: Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs},
  author={Meng, Weikang and Huo, Liangyu and Luo, Yadan and Guan, Jiawen and Zhang, Jingyi and Li, Yingjian and Zhang, Zheng},
  journal={arXiv preprint arXiv:2602.02180},
  year={2026}
}
```

