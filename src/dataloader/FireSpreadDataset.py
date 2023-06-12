from pathlib import Path
from typing import List, Optional

import rasterio
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data.dataset import T_co
import glob
import warnings
from .utils import get_means_and_stds, get_indices_of_degree_features
import torchvision.transforms.functional as TF
import h5py
from datetime import datetime


class FireSpreadDataset(Dataset):
    def __init__(self, data_dir: str, included_fire_years: List[int], n_leading_observations: int,
                 crop_side_length: int, load_from_hdf5: bool, is_train: bool, remove_duplicate_features: bool,
                 features_to_keep: Optional[List[int]] = None, return_doy: bool = False, add_binary_fire_mask: bool = True):
        """

        :param return_pre_aug_imgs: Output preprocessed images x_i at the stage before augmentation. This is used to
        extract the data in a form that can be efficiently saved in HDF5 format. At this stage, nan has been replaced by
         0.0, active fire detections have been converted to their hour of detection, but features that span 360° are
         still in degrees, so corrections for augmentations are easy to apply.
        :param data_dir: Root directory of the dataset, should contain several folders, each corresponding to a different fire.
        :param included_fire_years: Names of the folders in dataset_root that should be used in this instance of the dataset.
        :param n_leading_observations: Number of days in the past for which observations are included.
        :param crop_side_length Crops will be of size crop_side_length x crop_side_length.
        :param load_from_hdf5 If True, load from HDF5 files, with one file per fire. If False, load from GeoTIFF files,
        with one file per day per fire.
        :param is_train: If True, apply geometric data augmentations. If False, only apply center crop to get the required dimensions.
        :param remove_duplicate_features: If True, remove features that are static over the entire time series. 
        Only keep them for the last image. Then flattent the whole remaining data.
        :param features_to_keep If not None, only keep the features with indices specified by this list.
        :param return_doy: If True, return the day of the year as an additional feature of the input data.

        """
        super().__init__()

        self.add_binary_fire_mask = add_binary_fire_mask
        self.return_doy = return_doy
        self.features_to_keep = features_to_keep
        self.remove_duplicate_features = remove_duplicate_features
        self.is_train = is_train
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.included_fire_years = included_fire_years
        self.data_dir = data_dir

        self.validate_inputs()

        self.imgs_per_fire = self.read_list_of_images()
        self.datapoints_per_fire = self.compute_datapoints_per_fire()
        self.length = sum([sum(self.datapoints_per_fire[fire_year].values())
                          for fire_year in self.datapoints_per_fire])

        self.index_to_feature_name = self.map_channel_index_to_features()

        # Used in preprocessing and normalization. Better to define it once than build/call for every data point
        self.one_hot_matrix = torch.eye(17)
        self.means_and_stds = get_means_and_stds()
        self.indices_of_degree_features = get_indices_of_degree_features()

    def find_image_index_from_dataset_index(self, target_id):
        if target_id < 0:
            target_id = self.length + target_id
        if target_id >= self.length:
            raise RuntimeError(
                f"Tried to access item {target_id}, but maximum index is {self.length - 1}.")

        # The index is relative to the length of the full dataset. However, we need to make sure that we know which
        # specific fire the queried index belongs to. We know how many data points each fire contains from
        # self.datapoints_per_fire.
        first_id_in_current_fire = 0
        found_fire_year = None
        found_fire_name = None
        for fire_year in self.datapoints_per_fire:
            for fire_name, datapoints_in_fire in self.datapoints_per_fire[fire_year].items():
                if target_id - first_id_in_current_fire < datapoints_in_fire:
                    found_fire_year = fire_year
                    found_fire_name = fire_name
                    break
                else:
                    first_id_in_current_fire += datapoints_in_fire

        in_fire_index = target_id - first_id_in_current_fire

        return found_fire_year, found_fire_name, in_fire_index

    def load_imgs(self, found_fire_year, found_fire_name, in_fire_index):

        end_index = (in_fire_index + self.n_leading_observations + 1)

        if self.load_from_hdf5:
            hdf5_path = self.imgs_per_fire[found_fire_year][found_fire_name][0]
            with h5py.File(hdf5_path, 'r') as f:
                imgs = f["data"][in_fire_index:end_index]
                if self.return_doy:
                    doys = f["data"].attrs["img_dates"][in_fire_index:(end_index-1)]
                    doys = self.img_dates_to_doys(doys)
                    doys = torch.Tensor(doys)
            x, y = np.split(imgs, [-1], axis=0)
            # Last image's active fire mask is used as label, rest is input data
            y = y[0, -1, ...]
        else:
            imgs_to_load = self.imgs_per_fire[found_fire_year][found_fire_name][in_fire_index:end_index]
            imgs = []
            for img_path in imgs_to_load:
                with rasterio.open(img_path, 'r') as ds:
                    imgs.append(ds.read())
            x = np.stack(imgs[:-1], axis=0)
            y = imgs[-1][-1, ...]

        if self.return_doy:
            return x, y, doys
        return x, y

    def __getitem__(self, index) -> T_co:

        found_fire_year, found_fire_name, in_fire_index = self.find_image_index_from_dataset_index(
            index)
        loaded_imgs = self.load_imgs(found_fire_year, found_fire_name, in_fire_index)
        
        if self.return_doy:
            x, y, doys = loaded_imgs
        else:
            x, y = loaded_imgs

        x, y = self.preprocess_and_augment(x, y)
                
        if self.add_binary_fire_mask:
            # After normalization, fire values of 0 are transformed into -0.03, so this is a safe threshold.
            x = torch.cat([x, (x[:,-1:, ...] > 0).float()], axis=1)

        if self.remove_duplicate_features:
            x = self.flatten_and_remove_duplicate_features_(x)
       
        if self.features_to_keep is not None:
            if self.remove_duplicate_features or len(x.shape) != 4:
                raise NotImplementedError()
            x = x[:, self.features_to_keep, ...]

        if self.return_doy:
            return x, y, doys
        return x, y

    def __len__(self):
        return self.length

    def validate_inputs(self):
        if self.n_leading_observations < 1:
            raise ValueError("Need at least one day of observations.")
        if self.return_doy and not self.load_from_hdf5:
            raise NotImplementedError(
                "Returning day of year is only implemented for hdf5 files.")

    def read_list_of_images(self):
        """
        :return: Returns a dictionary mapping integer years to dictionaries. These map names of fires that happened
        within the respective year to either
        a) the corresponding list of image files (in case hdf5 files are not used) or
        b) the individual hdf5 file for each fire.
        """
        imgs_per_fire = {}
        for fire_year in self.included_fire_years:
            imgs_per_fire[fire_year] = {}

            if not self.load_from_hdf5:
                fires_in_year = glob.glob(f"{self.data_dir}/{fire_year}/*/")
                fires_in_year.sort()
                for fire_dir_path in fires_in_year:
                    fire_name = fire_dir_path.split("/")[-2]
                    fire_img_paths = glob.glob(f"{fire_dir_path}/*.tif")
                    fire_img_paths.sort()

                    imgs_per_fire[fire_year][fire_name] = fire_img_paths

                    if len(fire_img_paths) == 0:
                        warnings.warn(f"In dataset preparation: Fire {fire_year}: {fire_name} contains no images.",
                                      RuntimeWarning)
            else:
                fires_in_year = glob.glob(
                    f"{self.data_dir}/{fire_year}/*.hdf5")
                fires_in_year.sort()
                for fire_hdf5 in fires_in_year:
                    fire_name = Path(fire_hdf5).stem
                    imgs_per_fire[fire_year][fire_name] = [fire_hdf5]

        return imgs_per_fire

    def compute_datapoints_per_fire(self):
        datapoints_per_fire = {}
        for fire_year in self.imgs_per_fire:
            datapoints_per_fire[fire_year] = {}
            for fire_name, fire_imgs in self.imgs_per_fire[fire_year].items():
                if not self.load_from_hdf5:
                    n_fire_imgs = len(fire_imgs)
                else:
                    # Catch error case that there's no file
                    if not fire_imgs:
                        n_fire_imgs = 0
                    else:
                        with h5py.File(fire_imgs[0], 'r') as f:
                            n_fire_imgs = len(f["data"])
                # If we have two days of observations, and a lead of one day,
                # we can only predict the second day's fire mask, based on the first day's observation
                datapoints_in_fire = n_fire_imgs - self.n_leading_observations
                if datapoints_in_fire <= 0:
                    warnings.warn(
                        f"In dataset preparation: Fire {fire_year}: {fire_name} does not contribute data points. It contains "
                        f"{len(fire_imgs)} images, which is too few for a lead of {self.n_leading_observations} observations.",
                        RuntimeWarning)
                    datapoints_per_fire[fire_year][fire_name] = 0
                else:
                    datapoints_per_fire[fire_year][fire_name] = datapoints_in_fire
        return datapoints_per_fire

    def normalize(self, x):

        # x has shape (n_imgs, n_features, height, width)
        # means and stds have shape (n_features)
        # We want to subtract the mean and divide by the std for each feature, for each image.
        means, stds = self.means_and_stds
        means = means[None, :, None, None]
        stds = stds[None, :, None, None]
        x = (x - means) / stds

        return x

    def preprocess_and_augment(self, x, y):

        x, y = torch.Tensor(x), torch.Tensor(y)

        # Preprocessing that has been done in HDF files already
        if not self.load_from_hdf5:

            # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
            # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
            # them with 0 instead.
            x[:, -1, ...] = torch.nan_to_num(x[:, -1, ...], nan=0)
            y = torch.nan_to_num(y, nan=0.0)

            # Turn active fire detection time from hhmm to hh.
            x[:, -1, ...] = torch.floor_divide(x[:, -1, ...], 100)

        y = (y > 0).long()

        # Augmentation has to come before normalization, because we have to correct the angle features when we change
        # the orientation of the image.
        if self.is_train:
            x, y = self.augment(x, y)
        else:
            x, y = self.center_crop_x32(x, y)

        # Some features take values in [0,360] degrees. By applying sin, we make sure that values near 0 and 360 are
        # close in feature space, since they are also close in reality.
        x[:, self.indices_of_degree_features, ...] = torch.sin(
            torch.deg2rad(x[:, self.indices_of_degree_features, ...]))

        x = self.normalize(x)

        # Replace NaN values with 0, thereby essentially setting them to the mean of the respective feature.
        x = torch.nan_to_num(x, nan=0.0)

        # Create land cover class one-hot encoding, put it where the land cover integer was
        new_shape = (x.shape[0], x.shape[2], x.shape[3],
                     self.one_hot_matrix.shape[0])
        landcover_classes_flattened = x[:, 16, ...].long().flatten() -1 # -1 because land cover classes start at 1
        landcover_encoding = self.one_hot_matrix[landcover_classes_flattened].reshape(
            new_shape).permute(0, 3, 1, 2)
        x = torch.concatenate(
            [x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1)

        return x, y

    def augment(self, x, y):
        """
        Applies geometric transformations: 
          1. random square cropping, preferring images with a) fire pixels in the output and b) (with much less weight) fire pixels in the input
          2. rotate by multiples of 90°
          3. flip horizontally and vertically
        Adjustment of angles is done as in https://github.com/google-research/google-research/blob/master/simulation_research/next_day_wildfire_spread/image_utils.py
        :param x:
        :param y:
        :return:
        """

        # Need square crop to prevent rotation from creating/destroying data at the borders, due to uneven side lengths.
        # Try several crops, prefer the ones with most fire pixels in output, followed by most fire_pixels in input
        best_n_fire_pixels = -1
        best_crop = (None, None)

        for i in range(10):
            top = np.random.randint(0, x.shape[-2] - self.crop_side_length)
            left = np.random.randint(0, x.shape[-1] - self.crop_side_length)
            x_crop = TF.crop(
                x, top, left, self.crop_side_length, self.crop_side_length)
            y_crop = TF.crop(
                y, top, left, self.crop_side_length, self.crop_side_length)

            # We really care about having fire pixels in the target. But if we don't find any there,
            # we care about fire pixels in the input, to learn to predict that no new observations will be made,
            # even though previous days had active fires.
            n_fire_pixels = x_crop[:, -1, ...].mean() + \
                1000 * y_crop.float().mean()
            if n_fire_pixels > best_n_fire_pixels:
                best_n_fire_pixels = n_fire_pixels
                best_crop = (x_crop, y_crop)

        x, y = best_crop

        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))
        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = 360 - \
                x[:, self.indices_of_degree_features, ...]

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (
                180 - x[:, self.indices_of_degree_features, ...]) % 360

        if rotate != 0:
            angle = rotate * 90
            x = TF.rotate(x, angle)
            y = torch.unsqueeze(y, 0)
            y = TF.rotate(y, angle)
            y = torch.squeeze(y, 0)

            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (x[:, self.indices_of_degree_features,
                                                          ...] - 90 * rotate) % 360

        return x, y

    def center_crop_x32(self, x, y):
        """
        Crops the center of the image to side lengths that are a multiple of 32, 
        which the ResNet U-net architecture requires.
        Only used for computing the test performance.
        :param x: input image data
        :param y: target active fire segmentation map
        :return: cropped input and target data
        """
        T,C,H,W = x.shape
        H_new = H//32 * 32
        W_new = W//32 * 32

        x = TF.center_crop(x, (H_new, W_new))
        y = TF.center_crop(y, (H_new, W_new))
        return x, y

    def flatten_and_remove_duplicate_features_(self, x):
        """
        For a simple U-Net, static and forecast features can be removed everywhere but in the last time step
        to reduce the number of features. Since that would result in different numbers of channels for different
        time steps, we flatten the temporal dimension. 
        """
        # ids_to_remove = [12,13,14] + list(range(16,34))
        ids_to_keep = torch.Tensor(
            list(range(12)) + [15] + list(range(34, x.shape[1]))).int()

        x_deduped = x[:-1, ids_to_keep, :, :].flatten(start_dim=0, end_dim=1)
        x_ = torch.cat([x_deduped, x[0, ...]], axis=0)

        return x_

    @staticmethod
    def img_dates_to_doys(img_dates):
        """
        Converts a list of date strings to day of year values.
        :param img_dates: list of datetime objects
        :return: list of day of year values
        """
        date_format = "%Y-%m-%d"
        return [datetime.strptime(img_date.replace(".tif", ""), date_format).timetuple().tm_yday for img_date in img_dates]

    @staticmethod
    def map_channel_index_to_features():
        """
        Computes a dictionary of
            feature_index: feature_name
        to make it easier to understand which transformations are being applied to which feature.
        """
        return {0: 'VIIRS band M11',
                1: 'VIIRS band I2',
                2: 'VIIRS band I1',
                3: 'NDVI',
                4: 'EVI2',
                5: 'total precipitation',
                6: 'wind speed',
                7: 'wind direction',
                8: 'minimum temperature',
                9: 'maximum temperature',
                10: 'energy release component',
                11: 'specific humidity',
                12: 'slope',
                13: 'aspect',
                14: 'elevation',
                15: 'pdsi',
                16: 'Landcover_Type1',
                17: 'forecast total_precipitation',
                18: 'forecast wind speed',
                19: 'forecast wind direction',
                20: 'forecast temperature',
                21: 'forecast specific humidity',
                22: 'active fire'}

    def get_generator_for_hdf5(self):
        """
        :return: generator that yields tuples of (year, fire_name, img_dates, lnglat, img_array) 
        where img_array contains all images available for the respective fire, preprocessed such 
        that active fire detection times are converted to hours. lnglat contains longitude and latitude
        of the center of the image.
        """

        for year, fires_in_year in self.imgs_per_fire.items():
            for fire_name, img_files in fires_in_year.items():
                imgs = []
                lnglat = None
                for img_path in img_files:
                    with rasterio.open(img_path, 'r') as ds:
                        imgs.append(ds.read())
                        if lnglat is None:
                            lnglat = ds.lnglat()
                x = np.stack(imgs, axis=0)

                # Get dates from filenames
                img_dates = [img_path.split("/")[-1].split("_")[0]
                             for img_path in img_files]

                # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
                # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
                # them with 0 instead.
                x[:, -1, ...] = np.nan_to_num(x[:, -1, ...], nan=0)

                # Turn active fire detection time from hhmm to hh.
                x[:, -1, ...] = np.floor_divide(x[:, -1, ...], 100)
                yield year, fire_name, img_dates, lnglat, x
