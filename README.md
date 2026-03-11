# Train VAE


## env
```bash
pip install -r requirements.txt
```

## dataset

```bash
python download.py
```

## visualize training step

uncomment the tqdm_progress_bar in [configs/training/default.yaml](configs/training/default.yaml)

```bash
#  tqdm_progress_bar:
#    _target_: pytorch_lightning.callbacks.TQDMProgressBar
```

## training

```bash
python src/train_vae.py
```

or 

```bash
bash train_vae.sh
```

remember to change the data path

## Bug

resume training might occur error
