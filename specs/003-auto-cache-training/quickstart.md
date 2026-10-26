# Quickstart: Automatic cache preparation

Use the existing installed environment and provide real model paths and captioned images. The main config enables experiment_mode=true and auto_cache=true.

From the repository root:
~~~sh
accelerate launch --num_processes 1 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml
~~~

First launch prepares missing train/val latent and text caches, then trains. Relaunch reuses valid files. From another directory, use absolute script and training-TOML paths; experiment-relative data/model paths retain their meaning.

Disable automatic preparation for a manual-cache launch:
~~~sh
accelerate launch --num_processes 1 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml --no_auto_cache
~~~

Missing/invalid models and present corrupt/stale cache files stop before training. Correct the reported cause; regenerate invalid caches deliberately using the existing cache commands. Automatic mode neither overwrites such files nor deletes unrelated files.

Local acceptance covers real source/config/cache logic and native entrypoints with substituted expensive encoders/model execution. No GPU or remote run is part of this feature's acceptance.
