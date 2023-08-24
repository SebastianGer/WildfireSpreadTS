# WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction

This repository contains the code corresponding to the paper with the name above. 

## Setup the environment

``` pip3 install -r requirements.txt ```

## Preparing the dataset

The dataset is freely available on Zenodo (link will be added upon acceptance of the paper). After downloading it, it should be converted to HDF5 files. Be aware that they take up about twice as much space as the original dataset! At the same time, they allow for much faster access, without which training is not feasible. 

To convert the dataset, run:
```python3 src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR```
 substituting the path to your local dataset and where you want the HDF5 version of the dataset to be created. 

You can skip this step, and simply pass `--data.load_from_hdf5=False` on the command line, but be aware that you won't be able to perform training at any reasonable speed. 

## Re-running the baseline experiments

We use wandb to log experimental results. This can be turned off by setting the environment variable `WANDB_MODE=disabled`. The results will then be logged to a local directory instead.

Experiments are parameterized via yaml files in the `cfgs` directory. Arguments are parsed via the [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

Grid searches and repetitions of experiments were done via WandB sweeps. Those are parameterized via yaml files in the `cfgs` directory prefixed with `wandb_`. WandB configuration files ending in `_repetition` contain the configurations for the runs that were used in the paper. They only needed to be _repeated_ five times with varying seeds to estimate the variance of results. We omit explanations of how to use wandb sweeps to run experiments and refer the readers to the [original documentation](https://docs.wandb.ai/guides/sweeps). To run the same experiments without WandB, the parameters specified in the WandB sweep configuration file can simply be passed via the command line. 

For example, to train the U-net architecture on one day of observations, which is specified in `cfgs/unet/wandb_monotemporal_repetition.yaml`, we could simply copy and paste the WandB parameters to the command line:

```
python3 train.py --config=cfgs/unet/res18_monotemporal.yaml --trainer=cfgs/trainer_single_gpu.yaml --data=cfgs/data_monotemporal_full_features.yaml --seed_everything=0 --trainer.max_epochs=200 --do_test=True --data.data_dir YOUR_DATA_DIR
```
where you replace `YOUR_DATA_DIR` with the path to your local HDF5 dataset. Alternatively, you can permanently set the data directory in the respective data config files. Later arguments overwrite previously given arguments, including parameters defined in config files. 

## Re-creating the dataset

The code to create the dataset using Google Earth Engine is available at [https://github.com/SebastianGer/WildfireSpreadTSCreateDataset](https://github.com/SebastianGer/WildfireSpreadTSCreateDataset).


## Using the dataset for your own experiments

To use the dataset outside of the baseline experiments, you can use the Lightning Datamodule at `src/dataloader/FireSpreadDataModule.py` which directly provides dataset loaders for train/val/test set. Alternatively, you can use the PyTorch dataset at `src/dataloader/FireSpreadDataset.py`. 

## Citation

Will be added upon acceptance of the paper.