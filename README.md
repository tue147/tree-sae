# Tree SAE: Learning Hierarchical Feature Structures in Sparse Autoencoders

Official code for **Tree SAE** ([arXiv:2605.07922](https://arxiv.org/abs/2605.07922)).

A Tree SAE splits its dictionary into *privilege layers*. Every feature of a layer has one parent
among the features of earlier layers (or an always-active root), and

* it may only fire on tokens where its parent fires (**activation coverage**, Eq. 6), and
* the cumulative reconstruction of every layer must reconstruct the input on its own
  (**reconstruction condition**, Eq. 7),

so parent/child pairs are learned directly in the feature set. Children are dynamically
re-allocated to parents during training (Algorithms 1-2).

The repository contains the Tree SAE and the baselines of the paper (Matryoshka, Top-k,
Matching-Pursuit and ReLU SAEs), their training code, all evaluations (hierarchy, absorption,
splitting, composition, reconstruction, AutoInterp, sibling co-occurrence), the analyses of
Section 6 (child-subspace geometry, tree visualisation) and the code to redraw the paper figures.

## Installation

```bash
conda create -n tree-sae python=3.11 -y && conda activate tree-sae
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # or your CUDA/CPU build
pip install -e ".[dev,autointerp,wandb]"
```

The exact versions used to produce the released results are pinned in `environment.yml`.

## Released SAEs

All SAEs of the paper are trained on the residual stream before block 5 of GPT-2 small
(`blocks.5.hook_resid_pre`) with 500M Pile tokens, and are available at
[`tueminh/tree-sae`](https://huggingface.co/tueminh/tree-sae):

```python
from tree_sae.models.io import from_pretrained

sae = from_pretrained("gpt2-small/tree_sae_4layer_l0_32", device="cuda")
latents = sae.encode(activations)[0]  # (..., 24576) feature activations
recons = sae(activations)  # reconstruction
sae.children_of(5)  # children of feature 5 in the tree
sae.parent_index(1)  # allocation vector a_1 (root = sae.root_index(1))
```

| Paper name | Released names | Features per layer | L0 per layer |
|---|---|---|---|
| Tree SAE 2 layers | `tree_sae_2layer_l0_{32,48,64,80}` | 6144, 18432 | Table 1 |
| Tree SAE 4 layers | `tree_sae_4layer_l0_{32,48,64,80}` | 1536, 3072, 9216, 10752 | Table 1 |
| Matryoshka 2 / 4 layers | `matryoshka_{2,4}layer_l0_{32,48,64,80}` | as above | total k = L0 |
| Top-k SAE | `topk_l0_{32,48,64,80}` | 24576 | k = L0 |
| MP-SAE | `mp_sae_k_{32,48,64,80}` (max. k active; average L0 31.5-75.4) | 24576 | adaptive |
| Scaling (Sec. 5.4) | `{tree_sae,matryoshka}_{2,4}layer_{6k,49k}` | Table 2 | Table 2 |

Every released SAE has a matching training config in `configs/` (e.g.
`configs/main/tree_sae_4layer_l0_32.yaml`).

## Training

```bash
python train.py --config configs/main/tree_sae_4layer_l0_32.yaml --device cuda:0
```

Training streams `monology/pile-uncopyrighted`, computes GPT-2 activations on the fly and writes
`checkpoints/<name>/{sae.safetensors,config.json}`. Logging uses Weights & Biases
(`--no_wandb` to disable). A run is 97,657 steps of 5 x 1024 tokens.

Paper hyperparameters (Appendix C/G), all set in the configs: Adam with lr 1e-4, batches of
5 x 1024 tokens, AuxK with k_aux = 256 and alpha = 1/32 (for the Tree SAE on the first privilege
layer only), a feature is dead after 10M tokens without firing, a parent must have fired within the
last 50k tokens to receive children, reallocation every 3000 steps (interval x2 after each
reallocation, capped at 10,000 steps), and a root reset at step 50,000.

## Evaluation

`evaluate.py` takes `name=location` pairs, where `location` is a saved SAE directory or `hf:<name>`:

```bash
python evaluate.py hierarchy      --sae tree_sae_4layer_l0_32=hf:gpt2-small/tree_sae_4layer_l0_32
python evaluate.py absorption     --sae topk_l0_32=hf:gpt2-small/topk_l0_32
python evaluate.py composition    --sae topk_l0_32=hf:gpt2-small/topk_l0_32
python evaluate.py reconstruction --sae topk_l0_32=hf:gpt2-small/topk_l0_32
python evaluate.py cooccurrence   --sae tree_sae_2layer_l0_32=hf:gpt2-small/tree_sae_2layer_l0_32
OPENAI_API_KEY=... python evaluate.py autointerp --sae topk_l0_32=hf:gpt2-small/topk_l0_32
```

| Paper | Command | Reported value |
|---|---|---|
| Hierarchy, Figs. 4, 5 | `hierarchy` | `hierarchy_mcs` / `hierarchy_tree_structure` |
| Absorption, Splitting, Figs. 3, 8 | `absorption` | `absorption` (SAEBench `mean_full_absorption_score`), `splitting` (`mean_num_split_features`) |
| Composition, Figs. 3, 8 | `composition` | mean max. cosine similarity between decoder vectors |
| Variance explained, CE loss, Figs. 6, 7 | `reconstruction` | `variance_explained`, `downstream_ce_loss` |
| AutoInterp, Figs. 3, 8 | `autointerp` | SAEBench AutoInterp score, judge `gpt-4o-mini`, 200 latents |
| Table 3 | `cooccurrence` | mean sibling IoU |
| MCS variants, Fig. 16 | `hierarchy --mcs_scaled` / `--no-mcs_binary` | |

Hierarchy, co-occurrence and the analyses use the first 10,000 x 128 tokens of the Pile validation
split, with GPT-2 and the SAE in bfloat16 as in the paper.

### Figures

```bash
python scripts/figures/plot_metrics.py                         # Figs. 3-8, 16 from results/paper_results.json
python scripts/collect_results.py --outputs outputs --out results/mine.json
python scripts/figures/plot_metrics.py --results results/mine.json   # same figures from your runs

python scripts/figures/probe_correlation.py --sae hf:gpt2-small/tree_sae_2layer_l0_32 --parent 1343 \
    --children 7487 12625 15275                                 # Fig. 1 (and Figs. 10-12)
python scripts/figures/child_rank.py --saes hf:gpt2-small/topk_l0_{32,48,64,80}   # Fig. 13
python scripts/figures/child_geometry.py --sae hf:gpt2-small/tree_sae_4layer_l0_32 --parent 5   # Fig. 9
python scripts/figures/feature_tree.py --sae hf:gpt2-small/tree_sae_2layer_l0_48 --root 3410     # Figs. 17-20
```

`results/paper_results.json` holds every number plotted in the paper. `notebooks/` contains short
interactive versions of the analyses.

## Repository layout

```
tree_sae/
  models/     ReLUSAE, TopKSAE, MatryoshkaSAE, MPSAE, TreeSAE; Algorithm 1 (allocation.py);
              save/load (io.py); conversion of the research checkpoints (convert.py)
  training/   Lightning module, losses, streaming data, config, trainer
  evals/      hierarchy, composition, reconstruction, absorption/ and autointerp/ (from SAEBench)
  analysis/   activations, co-occurrence, probe correlation, geometry, tree visualisation, plotting
configs/      one YAML per released SAE          scripts/  figures, checkpoint conversion/upload
results/      numbers of the paper               tests/    unit tests and golden regression tests
```

## Implementation details not stated in the paper

These follow the code that trained the released SAEs:

* **Capacity of a parent (Eq. 10).** After every step, the step's total loss (reconstruction +
  auxiliary) is split equally among all candidate parents that fired at least once in the batch,
  the root always counting as active, and added to their capacities `C_p`. Capacities are reset
  after each reallocation. (The paper describes adding the loss once per token a parent fires on.)
* **Reallocation.** Only dead children move; they are taken from parents holding more children
  than their quota and given to parents holding fewer. Parents with zero capacity get no quota.
  After the root reset, no further reallocation happens.
* **Losses.** Inputs are standardised per token; each reconstruction MSE is divided by `Var(x)`;
  the AuxK MSE is divided by the per-token variance of the residual. The AuxK reconstruction
  includes the decoder bias, so the auxiliary loss is non-zero (and trains `b_dec`) even before
  any feature is dead.
* **Optimisation.** `b_dec` is initialised to the geometric median of the first batch; decoder
  vectors are renormalised to unit norm after every step and the gradient component parallel to
  them is removed; Adam uses eps = 6.25e-10; the SAE trains in 16-bit mixed precision while GPT-2
  runs in fp32. Lightning keeps the periodic checkpoint (every 1000 steps) with the lowest training
  loss and stops early if the loss stops improving by 0.005.
* **Matryoshka SAE.** Top-k over the full dictionary; the loss adds the (variance-normalised) MSE of
  every strict prefix to the full-dictionary loss.

## Tests

```bash
pytest tests/unit                  # fast CPU tests
pytest tests/golden -m "not gpu"   # bit-exact regression against the research code (CPU)
pytest tests/golden -m gpu         # real GPT-2 training steps and SAEBench evals (GPU)
```

## Citation

```bibtex
@article{cao2026treesae,
  title   = {Tree SAE: Learning Hierarchical Feature Structures in Sparse Autoencoders},
  author  = {Cao, Tue M. and Nhat, Hoang X. and Alharbi, Raed and Nguyen, Phi Le and Thai, My T.},
  journal = {arXiv preprint arXiv:2605.07922},
  year    = {2026}
}
```

## Acknowledgements

The absorption/splitting and AutoInterp evaluations are adapted from
[SAEBench](https://github.com/adamkarvonen/SAEBench) and the probe code from
[sae-spelling](https://github.com/lasr-spelling/sae-spelling) (both MIT). The SAE implementation
builds on [openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder) and
[EleutherAI/sae](https://github.com/EleutherAI/sae).
