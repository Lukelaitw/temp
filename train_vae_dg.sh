export WANDB_MODE=offline
srun /gpfs/projects/p32572/Luke/.venv/bin/python src/train_vae.py \
  model.module.vae_model.encoder.n_layer=2 \
  model.module.vae_model.encoder.n_inducing_points=128 \
  model.module.vae_model.encoder.n_embed=128 \
  model.module.vae_model.encoder.n_embed_latent=16 \
  model.module.vae_model.encoder.n_head=4 \
  model.module.vae_model.encoder.n_head_cross=1 \
  paths.base_data_path=/projects/p32572/Luke/_artifacts/datasets \
  paths.base_release_path=/projects/p32572/Luke/_artifacts \
  datamodule.dataset=dentate_gyrus \
  experiment_name=my_vae_experiment \
  training.num_epochs=100 \
  model.test_batch_size=128