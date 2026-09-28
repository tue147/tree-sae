---
license: mit
library_name: tree-sae
tags:
  - sparse-autoencoder
  - interpretability
  - gpt2
---

# Tree SAE checkpoints

Sparse autoencoders from **Tree SAE: Learning Hierarchical Feature Structures in Sparse
Autoencoders** ([arXiv:2605.07922](https://arxiv.org/abs/2605.07922)); code at
[github.com/tue147/tree-sae](https://github.com/tue147/tree-sae).

All SAEs are trained on the residual stream before block 5 of GPT-2 small
(`blocks.5.hook_resid_pre`) on 500M tokens of the Pile.

```python
from tree_sae.models.io import from_pretrained

sae = from_pretrained("gpt2-small/tree_sae_4layer_l0_32", device="cuda")
```

| Folder | Paper name |
|---|---|
| `gpt2-small/tree_sae_{2,4}layer_l0_{32,48,64,80}` | Tree SAE, 2 / 4 privilege layers, 24k features |
| `gpt2-small/matryoshka_{2,4}layer_l0_{32,48,64,80}` | Matryoshka SAE, 2 / 4 prefixes, 24k features |
| `gpt2-small/topk_l0_{32,48,64,80}` | Top-k SAE, 24k features |
| `gpt2-small/mp_sae_k_{32,48,64,80}` | Matching Pursuit SAE (at most k active features), 24k features |
| `gpt2-small/{tree_sae,matryoshka}_{2,4}layer_{6k,49k}` | Scaling experiments (Section 5.4) |

Each folder holds `sae.safetensors` and `config.json` (constructor arguments; `metadata.source_checkpoint`
names the original training checkpoint). For Tree SAEs, `parent_index_{l}` is the allocation vector of
privilege layer `l`: the parent of each feature of that layer, where the value equal to the layer's first
feature index denotes the root.

```bibtex
@article{cao2026treesae,
  title   = {Tree SAE: Learning Hierarchical Feature Structures in Sparse Autoencoders},
  author  = {Cao, Tue M. and Nhat, Hoang X. and Alharbi, Raed and Nguyen, Phi Le and Thai, My T.},
  journal = {arXiv preprint arXiv:2605.07922},
  year    = {2026}
}
```
