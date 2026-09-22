# LoHa and LoKr for original Qwen-Image

These alternative adapters use the same Qwen training, caching, sampling and saving workflow. Implementations are based on [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) by [KohakuBlueleaf](https://github.com/KohakuBlueleaf). LoHa uses the Hadamard product of low-rank matrices ([FedPara](https://arxiv.org/abs/2108.06098)); LoKr uses a Kronecker product with optional low-rank decomposition ([reference](https://arxiv.org/abs/2309.14859)). Existing experimental status is retained.

```toml
network_module = "networks.loha"
network_dim = 8
network_alpha = 8
network_args = ["rank_dropout=0.1", "module_dropout=0.05"]
```

For LoKr use `network_module="networks.lokr"` and, for example, `network_args=["factor=2", "rank_dropout=0.1"]`. `factor=-1` retains automatic factorization; positive values select the existing factorization parameter. Large dimensions can use full-matrix mode as before. Both target Qwen linear layers, auto-detecting `QwenImageTransformerBlock`; no other architecture dispatch remains.

The equivalent qualified names are `musubi_tuner.networks.loha` and `musubi_tuner.networks.lokr`; standard LoRA uses `networks.lora_qwen_image` or its qualified form. Dimension, alpha, neuron/rank/module dropout, pattern matching, LR ratios, weight IO and dtype bridges use the retained adapter engine. Only consumed argument names are accepted; `factor` belongs to LoKr. See [advanced settings](advanced_config.md) for literal/pattern syntax and [Qwen workflow](qwen_image.md) for both caches and training.

## 日本語

LoHaとLoKrは、KohakuBlueleaf氏のLyCORISに基づく実験的な代替アダプターです。同じQwen学習フローで`network_module`を切り替えます。LoKrの`factor`は-1で自動、または正の整数です。元の重み保存と学習処理を維持しています。
