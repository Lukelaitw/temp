#!/bin/sh
export WANDB_MODE=offline

# sbm.yaml 從 paths.dataset_paths.replogle 讀取 source（control/P0）與 train、test。
# base_repo_path 只影響預設路徑；若檔案不在 ${base_repo_path}/datasets/replogle/ 下，務必覆寫這三行。
srun /gpfs/projects/p32572/Luke/.venv/bin/python src/train_sbm.py \ # modify the path
  --config-name sbm_training \
  paths.base_repo_path=./ \
  paths.base_data_path=./data \
  paths.base_release_path=./ \
  paths.base_output_path=./outputs \
  paths.dataset_paths.replogle.source=./data/replogle/replogle_source.h5ad \
  paths.dataset_paths.replogle.train=./data/replogle/replogle_train.h5ad \
  paths.dataset_paths.replogle.test=./data/replogle/replogle_test.h5ad \
  experiment_name=sbm_replogle \
  seed=12345 \
  datamodule.dataset=replogle \
  datamodule.datamodule.control_label_key=gene \
  datamodule.datamodule.control_label_value=non-targeting \
  datamodule.datamodule.n_samples_per_epoch=200000 \
  datamodule.datamodule.num_workers=8 \
  model.batch_size=128 \
  model.test_batch_size=128 \
  model.module.vae_as_tokenizer.train=false \
  model.module.vae_as_tokenizer.load_from_checkpoint.ckpt_path=./ \
  model.module.vae_as_tokenizer.load_from_checkpoint.job_name=replogle_vae_2 \
  model.module.vae_as_tokenizer.load_from_checkpoint.epoch=null \
  model.module.sbm_model.condition_vocab_sizes.gene=2023 \
  model.module.sbm_model.condition_vocab_sizes.cell_line=4 \
  model.module.sbm_model.cond_embed_dim=64 \
  model.module.sbm_model.cond_hidden_dim=128 \
  model.module.sbm_optimizer.lr=1e-3 \
  +model.module.sbm_optimizer.betas=[0.9,0.95] \
  model.module.euler_maruyama_steps=100 \
  training.num_epochs=1000 \
  training.trainer.max_steps=113000 \
  training.trainer.enable_progress_bar=true \
  training.trainer.check_val_every_n_epoch=5 \
  training.trainer.gradient_clip_val=5.0 \
  training.callbacks.model_checkpoints.save_top_k=3 \
  training.callbacks.model_checkpoints.save_last=true