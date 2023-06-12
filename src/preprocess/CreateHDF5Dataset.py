from src.dataloader.FireSpreadDataset import FireSpreadDataset
import argparse
import os
import sys
from pathlib import Path

import h5py
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__).split("/src")[-2]))


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str,
                    help="Path to dataset directory", required=True)
parser.add_argument("--target_dir", type=str,
                    help="Path to directory where the HDF5 files should be stored", required=True)
args = parser.parse_args()

# Need to prevent some error with HDF5 files being locked and thereby inaccessible
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


years = [2018, 2019, 2020, 2021]

dataset = FireSpreadDataset(data_dir=args.data_dir,
                            included_fire_years=years,
                            # the following args are irrelevant here, but need to be set
                            n_leading_observations=1, crop_side_length=128, load_from_hdf5=False, is_train=True,
                            remove_duplicate_features=False)
data_gen = dataset.get_generator_for_hdf5()


base_dir = "/tmp"

for y in [2018, 2019, 2020, 2021]:
    target_dir = f"{args.target_dir}/{y}"
    Path(target_dir).mkdir(parents=True, exist_ok=True)

for year, fire_name, img_dates, lnglat, imgs in tqdm(data_gen):

    # For some reason creating HDF5 files directly where we want them doesn't work
    # year_dir = f"{base_dir}/{year}/"
    # Path(year_dir).mkdir(parents=True, exist_ok=True)
    # h5_path = year_dir + f"{fire_name}.hdf5"
    target_dir = f"{args.target_dir}/{year}"
    h5_path = f"{target_dir}/{fire_name}.hdf5"

    if Path(h5_path).is_file():
        print(f"File {h5_path} already exists, skipping...")
        continue

    with h5py.File(h5_path, "w") as f:
        dset = f.create_dataset("data", imgs.shape, data=imgs)
        dset.attrs["year"] = year
        dset.attrs["fire_name"] = fire_name
        dset.attrs["img_dates"] = img_dates
        dset.attrs["lnglat"] = lnglat

    # shutil.move(h5_path, target_dir)

    break
