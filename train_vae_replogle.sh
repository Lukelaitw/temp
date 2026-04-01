#!/bin/sh
export WANDB_MODE=offline
#srun /gpfs/projects/p32572/Luke/.venv/bin/python /projects/p32572/Luke/src/train_vae.py \
srun /gpfs/projects/p32572/Luke/.venv/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  /projects/p32572/Luke/src/train_vae.py \
  seed=12345 \
  model.module.vae_model.encoder.n_layer=2 \
  model.module.vae_model.encoder.n_inducing_points=128 \
  model.module.vae_model.encoder.n_embed=128 \
  model.module.vae_model.encoder.n_embed_latent=16 \
  model.module.vae_model.encoder.n_head=4 \
  model.module.vae_model.encoder.n_head_cross=1 \
  model.module.vae_optimizer.lr=0.008 \
  paths.base_data_path=/projects/p32572/Luke/_artifacts/datasets \
  paths.base_release_path=/projects/p32572/Luke/_artifacts \
  datamodule.dataset=replogle \
  experiment_name=replogle_vae \
  training.num_epochs=1000 \
  training.trainer.check_val_every_n_epoch=5 \
  training.callbacks.model_checkpoints.save_top_k=3 \
  datamodule.datamodule.train_adata_path=/projects/p32572/Luke/datasets/replogle/replogle_train_hvg.h5ad \
  datamodule.datamodule.test_adata_path=/projects/p32572/Luke/datasets/replogle/replogle_test_hvg.h5ad \
  datamodule.datamodule.sample_genes=expressed \
  datamodule.datamodule.prefetch_factor=8 \
  model.test_batch_size=128 \
  datamodule.datamodule.val_as_test=false \
  datamodule.datamodule.num_workers=8 \
  datamodule.datamodule.persistent_workers=true
