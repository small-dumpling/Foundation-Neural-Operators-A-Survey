# Foundation Neural Operators: A Survey on Pretraining Methods, Data Ecosystems, and Efficient Adaptation


A curated paper list for **"Foundation Neural Operators: A Survey on Pretraining Methods, Data Ecosystems, and Efficient Adaptation"**.

This repository organizes recent advances in **foundation-style neural operators** into three parts(Some articles overlap across domains):

1) **Pretraining Paradigms** (how foundation NOs are pretrained)  
2) **Data Ecosystem & Dataset Generation** (how large-scale PDE corpora and benchmarks are built)  
3) **Optimization & Efficient Adaptation/Inference** (how to train, adapt, and deploy efficiently)

> **Last updated:** 2026-01-18 (Europe/Amsterdam)

---

## Table of Contents

- [1. Pretraining Paradigms](#1-pretraining-paradigms)
- [2. Data Ecosystem & Dataset Generation](#2-data-ecosystem--dataset-generation)
- [3. Optimization & Efficient Adaptation](#3-optimization--efficient-adaptation)

---

## 1. Pretraining Paradigms

| Year | Model (Link) | Team | Key idea / paradigm | Pretraining I/O |
|:---:|:---|:---|:---|:---|
| 2025 | [GPhyT](https://arxiv.org/abs/2509.13805) | UVA / RWTH | In-context learning (ICL) for dynamics: imitate LLM-style prompting, infer future evolution from short trajectory prompts | short-trajectory prompt → future physical evolution |
| 2025 | [PDEformer-2](https://arxiv.org/abs/2507.15409) | Peking University | Formula-encoding paradigm: convert PDE into computation graph as input; align symbols with fields | PDE computation graph → solution query at arbitrary coordinates |
| 2025 | [PI-MFM](https://arxiv.org/abs/2512.23056) | Yale (Lu Lu) | Physics-informed multimodal paradigm: embed symbolic-driven PDE residual loss directly into self-supervised objectives | symbolic formula + IC/BC → operator solution field |
| 2025 | [Walrus](https://arxiv.org/abs/2511.15684) | Polymathic AI | Autoregressive evolution with harmonic stabilization (Patch Jittering) to suppress long-horizon error accumulation | spatiotemporal field sequence → next-step prediction |
| 2025 | [Flow Marching](https://arxiv.org/abs/2509.18611) | MIT | Flow Matching for operators: combine operator learning with probability paths to support uncertainty modeling | noisy state → clean physical field |
| 2025 | [DISCO](https://arxiv.org/abs/2504.19496) | ENS / Flatiron | Hypernetwork discovery: infer evolution-operator parameters dynamically from short trajectories | short state trajectory → operator network parameters |
| 2025 | [BCAT](https://arxiv.org/abs/2501.18972) | UCLA | Block-causal paradigm: treat fields as spatiotemporal causal “blocks” for sequence prediction | history block sequence → next physical block/frame |
| 2025 | [VICON](https://arxiv.org/abs/2411.16063) | UCLA | Vision ICL: patch-wise operations for efficient high-dimensional ICL | 2D function-pair sequence → next-step prediction |
| 2025 | [MORPH](https://arxiv.org/abs/2509.21670) | LANL | Shape-agnostic foundation operator: joint pretraining over 1D/2D/3D in one model | arbitrary mesh/point cloud → field response |
| 2025 | [PDE-PFN](https://openreview.net/forum?id=z7ilspv4uH) | - | Prior-data fitting (PFN): use noisy PINN simulations as universal prior; Bayesian ICL | grid/point cloud → posterior distribution |
| 2025 | [PDE-FM](https://arxiv.org/abs/2511.21861) | IBM | State-space (Mamba) paradigm: linear scaling for 3D complex physical systems | spatial–spectral tokens → operator solution |
| 2025 | [LOGLO-FNO](https://openreview.net/forum?id=B0E2yjrNb8) | - | Schwartz-operator paradigm: add mode interactions to address FNO pretraining expressivity bottleneck | discrete field → enhanced mapping field |
| 2025 | [OmniField](https://arxiv.org/pdf/2511.02205) | UCLA | Continuous neural field (CNF) paradigm: multimodal pretraining for sparse/noisy experimental data | sparse sensor streams → continuous-field reconstruction |
| 2025 | [OmniLearn](https://arxiv.org/abs/2502.14652) | - | Injection physics foundation model: general pretraining for high-energy particle trajectory evolution | particle configuration → physical property prediction |
| 2024 | [POSEIDON](https://arxiv.org/abs/2405.19101) | ETH Zurich | Multiscale operator Transformer: large-scale pretraining using PDE semigroup structure | initial condition + time step → evolution field |
| 2024 | [Aurora](https://arxiv.org/html/2405.13063v2) | Microsoft | Atmospheric foundation model: 3D Swin Transformer trained on massive global weather data | heterogeneous weather variables → high-res forecast |
| 2024 | [DPOT](https://arxiv.org/abs/2403.03542) | Tsinghua / Alibaba | Autoregressive denoising with Fourier Attention for large heterogeneous PDE corpora | history sequence → next-step prediction |
| 2024 | [Unisolver](https://arxiv.org/abs/2405.17527) | Tsinghua University | Conditional-guided paradigm: embed PDE symbols, BCs, etc. as deep conditions in Transformer | PDE component symbols → solution field |
| 2024 | [PROSE-FD](https://arxiv.org/abs/2409.09811) | UCLA | Multimodal fluids: fuse symbolic expressions and numerical fields for zero-shot fluid simulation  | symbolic description + numerical → future trajectory |
| 2024 | [UPTS](https://proceedings.neurips.cc/paper_files/paper/2024/file/2cd36d327f33d47b372d4711edd08de0-Paper-Conference.pdf) | - | Universal Physics Solver: adapt general LLM to time-dependent PDE solver | serialized PDE trajectory → solution prediction |
| 2024 | [PITT](https://arxiv.org/abs/2511.09729) | Harvard University&Google | Physics-Informed Token Transformer: tokenize symbols to learn analytic correction terms | system state + symbolic encoding → dynamics correction |
| 2023 | [ICON](https://www.pnas.org/doi/10.1073/pnas.2310142120) | Osher team | Pioneering in-context operator learning: identify new operators via function-pair sequences | function-pair sequence → new operator prediction |
| 2023 | [MPP](https://arxiv.org/abs/2310.02994) | Polymathic AI | Pioneering multi-physics pretraining: demonstrate transfer gain from joint training across equations | heterogeneous physical fields → spatiotemporal evolution |
| 2023 | [FactFormer](https://arxiv.org/abs/2305.17560) | - | Axial decomposition paradigm: scalable Transformer for multi-D PDE pretraining (DPOT precursor) | multi-D grid data → field prediction |
| 2025 | [OmniArch](https://arxiv.org/pdf/2402.16014v2) | Beihang University | Unified multi-physics architecture + cross-dim joint pretraining; PDEAligner for consistency | unified modeling format for multi-physics multi-input |
| 2025 | [OmniFluids](https://arxiv.org/abs/2506.10862) | Renmin University of China | Physics-first CFD foundation solver: physics pretraining → coarse-grid operator distillation → few-shot fine-tuning | unified modeling format for multi-physics multi-input |
| 2025 | [PDE-Transformer](https://arxiv.org/abs/2505.24717) | Technical University of Munich | Channel-independent interaction: embed physical variables independently as tokens + channel-level self-attention | unified modeling format for multi-physics multi-input |
| 2024 | [Text2PDE](https://arxiv.org/pdf/2410.01153) | Carnegie Mellon University | Text-to-PDE full-field visualization via diffusion conditioned on PDE text | text → complete PDE field generation |
| 2023 | [Scaling & Transfer Behavior](https://proceedings.neurips.cc/paper_files/paper/2023/file/e15790966a4a9d85d688635c88ee6d8a-Paper-Conference.pdf) | Lawrence Berkeley National Lab | Scaling + transfer behavior of pretraining→downstream | history trajectories → future trajectories |
| 2024 | [CoDA-NO](https://openreview.net/pdf?id=wSpIdUXZYX) | NVIDIA | Codomain attention pretraining for multiphysics neural operators | history trajectories → future trajectories |
| 2024 | [Data-Efficient Operator Learning (UP + ICL)](https://proceedings.neurips.cc/paper_files/paper/2024/file/0bac492172db3311c7e116098cfcf521-Paper-Conference.pdf) | Simon Fraser University | Unsupervised pretraining + downstream tuning | history trajectories → future trajectories |
| 2026 | [NCWNO](https://www.sciencedirect.com/science/article/pii/S0010465525003844) | UIUC | MoE + pretraining + fine-tuning to mitigate forgetting | - |
| 2025 | [MoE-POT](https://arxiv.org/abs/2510.25803) | USTC | MoE-based pretraining framework (DPOT-like I/O) | same as DPOT |
| 2025 | [LLMs as Multi-Modal DE Solvers](https://arxiv.org/pdf/2308.05061) | University of California | Multi-modal in-context operator learning (ICON + text prompting) | ICON-style + text prompts |
| 2025 | [ENMA](https://openreview.net/pdf?id=3CYXSMFv55) | NeurIPS 2025 | Token autoregressive diffusion, improving VICON | same as VICON |
| 2025 | [Zero-shot PDE](https://arxiv.org/abs/2509.06322) | - | Text-trained foundation models extrapolate spatiotemporal dynamics from PDE solutions, without fine-tuning or prompts. | Context → target|

---

## 2. Data Ecosystem & Dataset Generation


| Year | Technique / Dataset (Link) | Team | Key data innovation |
|:---:|:---|:---|:---|
| 2025 | [The Well (15 TB)](https://polymathic-ai.org/the_well/) | Polymathic AI | Largest physics corpus: standardized dataset spanning 19 heterogeneous scenarios |
| 2025 | [PaPQS](https://arxiv.org/abs/2511.00183) | TAMU / PNNL | Active query synthesis (AL): use information gain (EIG) to generate most informative PDE samples |
| 2025 | [UPTF-7 format](https://arxiv.org/abs/2509.21670) | LANL | Unified physics tensor format: standardize metadata & tokenization for 1D–3D heterogeneous data |
| 2025 | [Dimension augmentation](https://arxiv.org/abs/2511.15684) | Polymathic AI | Cross-dimension alignment: embed 2D trajectories into 3D space for joint pretraining (3D data scarcity) |
| 2025 | [Universal-form generation](https://arxiv.org/abs/2507.15409) | Peking University | Symbolic auto-synthesis: random coefficient masking/zeroing over 8 PDE templates to create massive corpora |
| 2024 | [All2All sampling (POSEIDON)](https://arxiv.org/abs/2405.19101) | ETH Zurich | Semigroup augmentation: mine O(T^2) training pairs from one trajectory using time separability |
| 2024 | [PDE Control Gym](https://proceedings.mlr.press/v242/bhan24a/bhan24a.pdf) | University of California | PDE control benchmark focusing on boundary-condition control |
| 2022 | [PDE Bench](https://arxiv.org/abs/2210.07182) | NEC Labs Europe | First large PDE benchmark |
| 2023 | [CFDBench](https://arxiv.org/abs/2310.05963) | Tsinghua University | Fluid dynamics benchmark |
| 2022 | [PDEArena](https://arxiv.org/abs/2209.15616) | Microsoft | Large PDE corpus & benchmark suite |
| 2024 | [POSEIDON (Downstream suite)](https://arxiv.org/abs/2405.19101) | ETH Zurich | 15 downstream tasks across broad PDE types (elliptic/parabolic/hyperbolic/mixed, linear/nonlinear, etc.) |
| 2025 | [Realistic Spatiotemporal Multiphysics Flows](https://arxiv.org/pdf/2512.18595) | - | Realistic multiphysics flow benchmark |
| 2026 | [REALPDEBENCH](https://arxiv.org/pdf/2601.01829) | Westlake University | Benchmark for complex physical systems with real-world data |
| 2025 | [Multiphysics Bench](https://arxiv.org/pdf/2505.17575) | HKU / Shanghai AI Lab | Multiphysics datasets & benchmarking |
| 2023 | [BubbleML](https://arxiv.org/pdf/2307.14623) | UC Irvine | Bubble multiphysics dataset & benchmarks |
| 2025 | [Open-CK](https://openreview.net/pdf?id=A23C57icJt) | USTC | Large multi-physics field-coupling benchmark in combustion kinetics |

---

## 3. Optimization & Efficient Adaptation


| Year | Technique (Link) | Team | Key optimization idea (efficient finetuning/inference) |
|:---:|:---|:---|:---|
| 2025 | [F-Adapter](https://arxiv.org/abs/2509.23173) | - | Frequency-adaptive PEFT: allocate parameters dynamically by spectral complexity; fix high-frequency finetuning collapse |
| 2025 | [Transolver++](https://arxiv.org/abs/2502.02414) | Tsinghua | Eidetic state parallelism: first linear scaling on single GPU for 1.2M grid points |
| 2025 | [MoE-POT](https://arxiv.org/abs/2510.25803) | USTC | Hierarchical routing: activate only 1/4 parameters, reduce inference cost by ~60% |
| 2025 | [Mondrian](https://arxiv.org/pdf/2506.08226) | University of California Irvine | Sliding-window attention: shifted-window mechanism makes 3D operator learning linear-time  |
| 2025 | [PhysicsCorrect](https://arxiv.org/abs/2507.02227) | University of Pennsylvania | Jacobian-cached correction: precompute PDE Jacobians to fix long-horizon drift at inference |
| 2025 | [PITA](https://arxiv.org/abs/2505.10930) | USTC | Alternating-direction optimization: jointly optimize predictor & equation features, reduce shortcut bias |
| 2024 | [MATEY](https://arxiv.org/abs/2412.20601) | Oak Ridge National Laboratory | Adaptive tokenization: adjust patch size by local variance (e.g., shocks), save ~8× compute |
| 2024 | [SPUS](https://arxiv.org/abs/2510.01370) | CAI | Extreme parameter-efficiency: lightweight residual U-Net for strong generalization pretraining |
| 2025 | [DeepONet as a Multi-Operator Extrapolation Model](https://www.sciencedirect.com/science/article/pii/S0021999125008198) | - | Distributed NO training (D2NO/MODNO) + physics-informed losses for zero-shot adaptation |
| 2025 | [PI-MFM](https://arxiv.org/abs/2512.23056) | Yale | Compute PDE residual losses via AD + finite differences from symbolic PDE inputs |
| 2025 | [SC-FNO](https://arxiv.org/abs/2505.08740) | Penn State University | Sensitivity-constrained training to improve robustness in few-shot settings |
| 2023 | [Scaling & Transfer Behavior](https://proceedings.neurips.cc/paper_files/paper/2023/file/e15790966a4a9d85d688635c88ee6d8a-Paper-Conference.pdf) | Lawrence Berkeley National Lab | Study pretraining→downstream transfer |
| 2024 | [CoDA-NO](https://openreview.net/pdf?id=wSpIdUXZYX) | NVIDIA | Codomain attention pretraining for multiphysics NOs |
| 2024 | [Data-Efficient Operator Learning](https://proceedings.neurips.cc/paper_files/paper/2024/file/0bac492172db3311c7e116098cfcf521-Paper-Conference.pdf) | - | Unsupervised pretraining + ICL for data efficiency |
| 2026 | [NCWNO](https://www.sciencedirect.com/science/article/pii/S0010465525003844) | UIUC | MoE + pretraining + finetuning to mitigate forgetting |
