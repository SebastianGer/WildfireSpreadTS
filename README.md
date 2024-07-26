# WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction

This repository contains the code for recreating the experiments in the WildfireSpreadTS paper. 

- [Link to main paper](https://openreview.net/pdf?id=RgdGkPRQ03)
- [Link to supplementary material](https://openreview.net/attachment?id=RgdGkPRQ03&name=supplementary_material)

**Note: After publishing the paper, we discovered a bug in the dataset class.** Based on initial experiments, the corrected dataset class leads to slightly higher performance, but the trends in the results are basically the same as those reported in the paper. The bug was fixed in commit `ab3c8f35c5ec8c52c306a4488eaeb71a5a13d0de`, in case you want to roll-back the change to compare with the results in the paper.

## Setup the environment

``` pip3 install -r requirements.txt ```

## Preparing the dataset

The dataset is freely available at [https://doi.org/10.5281/zenodo.8006177](https://doi.org/10.5281/zenodo.8006177) under CC-BY-4.0. For training, you will need to convert them to HDF5 files, which take up twice as much space but allow for much faster training.

To convert the dataset to HDF5, run:
```python3 src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR```
 substituting the path to your local dataset and where you want the HDF5 version of the dataset to be created. 

You can skip this step, and simply pass `--data.load_from_hdf5=False` on the command line, but be aware that you won't be able to perform training at any reasonable speed. 

## Re-running the baseline experiments

We use wandb to log experimental results. This can be turned off by setting the environment variable `WANDB_MODE=disabled`. The results will then be logged to a local directory instead.

Experiments are parameterized via yaml files in the `cfgs` directory. Arguments are parsed via the [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

Grid searches and repetitions of experiments were done via WandB sweeps. Those are parameterized via yaml files in the `cfgs` directory prefixed with `wandb_`. For example, to run the experiments that Table 3 in the main paper is based on, you can run a wandb sweep with `cfgs/unet/wandb_table3.yaml`. For explanations on how to use wandb sweeps please refer to the [original documentation](https://docs.wandb.ai/guides/sweeps). To run the same experiments without WandB, the parameters specified in the WandB sweep configuration file can simply be passed via the command line. 

For example, to train the U-net architecture on one day of observations, you could pass arguments on the command line as follows:

```
python3 train.py --config=cfgs/unet/res18_monotemporal.yaml --trainer=cfgs/trainer_single_gpu.yaml --data=cfgs/data_monotemporal_full_features.yaml --seed_everything=0 --trainer.max_epochs=200 --do_test=True --data.data_dir YOUR_DATA_DIR
```
where you replace `YOUR_DATA_DIR` with the path to your local HDF5 dataset. Alternatively, you can permanently set the data directory in the respective data config files. Parameters defined in config files are overwritten by command-line arguments. Later arguments overwrite previously given arguments. 

## Re-creating the dataset

The code to create the dataset using Google Earth Engine is available at [https://github.com/SebastianGer/WildfireSpreadTSCreateDataset](https://github.com/SebastianGer/WildfireSpreadTSCreateDataset).


## Using the dataset for your own experiments

To use the dataset outside of the baseline experiments, you can use the Lightning Datamodule at `src/dataloader/FireSpreadDataModule.py` which directly provides dataset loaders for train/val/test set. Alternatively, you can use the PyTorch dataset at `src/dataloader/FireSpreadDataset.py`. 

## Citation

```
@inproceedings{
    gerard2023wildfirespreadts,
    title={WildfireSpread{TS}: A dataset of multi-modal time series for wildfire spread prediction},
    author={Sebastian Gerard and Yu Zhao and Josephine Sullivan},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=RgdGkPRQ03}
}
```