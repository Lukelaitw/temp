export WANDB_MODE=offline
srun /gpfs/projects/p32572/Luke/.venv/bin/python src/train_vae.py   paths.base_data_path=/projects/p32572/Luke/_artifacts/datasets   paths.base_release_path=/projects/p32572/Luke/_artifacts   experiment_name=my_vae_experiment   training.num_epochs=100
