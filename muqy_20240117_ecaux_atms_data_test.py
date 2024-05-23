# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2024-01-18 10:44:58

import os

import numpy as np
from muqy_20240101_2Bgeoprof_reader import Reader
from muqy_20240104_filter_anvil_insitu_cirrus import CloudSatProcessor


# ----------------------------------------------------------------
# Extract cirrus in-situ and anvil-cirrus clouds mask
# ----------------------------------------------------------------
file_idx = 3
base_paths = {
    "cld_type": "../Data_python/CloudSat_data/2B_CLDCLASS_LIDAR/",
    "cld_ice": "../Data_python/CloudSat_data/2C_ICE/",
    "cld_fraction": "../Data_python/CloudSat_data/2B_GEOPROF_LIDAR/",
}

# structure = np.ones((7, 7)) means 3 profiles on each side + the current profile = 7 profiles
processor = CloudSatProcessor(
    file_idx, base_paths, structure=np.ones((7, 7))
)

processor.read_data()
processor.create_aux_cld_data()
processor.apply_connected_component_labeling()
processor.filter_cloud_clusters_connected_to_cirrus_or_DC()

processor.identify_anvil_cirrus()
processor.extend_anvil_cirrus()
processor.identify_insitu_cirrus()

# Extract in-situ cirrus and anvil-cirrus clouds
overall_data = processor.data
cirrus_insitu_mask = processor.cirrus_insitu_mask
cirrus_anvil_mask = processor.extented_cirrus_anvil_mask

# ----------------------------------------------------------------
# Read EC_AUX data to test
# ----------------------------------------------------------------


def read_data(self):
    """
    Reads data from multiple file paths and updates the data dictionary.

    Returns:
        None
    """
    file_paths = {
        key: os.path.join(
            path, sorted(os.listdir(path))[self.file_idx]
        )
        for key, path in self.base_paths.items()
    }
    for _, file_path in file_paths.items():
        self.data.update(self.read_cloudsat_data(file_path))


@staticmethod
def read_cloudsat_data(file_path):
    """
    Read CloudSat data from the specified file.

    Args:
        file_path (str): The path to the CloudSat data file.

    Returns:
        dict: A dictionary containing the CloudSat data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Implement the reading logic based on file format
    data = {}
    f = Reader(file_path)
    try:
        if "CLDCLASS" in file_path:
            data["lon"], data["lat"], data["elv"] = f.read_geo()
            data["time"] = f.read_time(datetime=False)
            data["start_time"], data["end_time"] = f.read_time(
                datetime=True
            )[[0, -1]]
            data["cld_layer_base"] = f.read_sds("CloudLayerBase")
            data["cld_layer_top"] = f.read_sds("CloudLayerTop")
            data["cld_layer_type"] = f.read_sds("CloudLayerType")
        elif "ICE" in file_path:
            data["re"] = f.read_sds("re")
            data["IWC"] = f.read_sds("IWC", process=False)
        elif "GEOPROF" in file_path:
            data["cf"] = f.read_sds("CloudFraction", process=False)
            data["cf"] = np.where(data["cf"] <= 0, np.nan, data["cf"])
            data["cf"] = np.where(
                data["cf"] >= 100, np.nan, data["cf"]
            )
            data["height"] = f.read_sds("Height")
            data["hgt"] = np.nanmean(data["height"], axis=0) / 1000
        elif "EC_AUX" in file_path:
            data["Temperature"] = f.read_sds(
                "Temperature", process=False
            )
            data["Specific_humidity"] = f.read_sds(
                "Specific_humidity", process=False
            )
            data["Skin_temperature"] = f.read_vdata("Temperature_2m")
    finally:
        f.close()

    return data


file_idx = 3
base_paths = {
    "EC_AUX": "../Data_python/CloudSat_data/EC_AUX/",
    "cld_type": "../Data_python/CloudSat_data/2B_CLDCLASS_LIDAR/",
    "cld_ice": "../Data_python/CloudSat_data/2C_ICE/",
    "cld_fraction": "../Data_python/CloudSat_data/2B_GEOPROF_LIDAR/",
}

file_paths = {
    key: os.path.join(path, sorted(os.listdir(path))[file_idx])
    for key, path in base_paths.items()
}

f = Reader(file_paths["EC_AUX"])

Temperature = f.read_sds("Temperature", process=False)

Temperature_2m = f.read_vdata("Temperature_2m")


# EC_AUX data has vdata(0 level only) :
# Latitude, Longitude, DEM_elevation, Skin_temperature,
# Surface_pressure, Temperature_2m, Sea_surface_temperature,
# U10_velocity, V10_velocity
# sdata(all levels) :
# Extrapolation_flag, Pressure, Temperature, Specific_humidity,
# Ozone, U_velocity, V_velocity
