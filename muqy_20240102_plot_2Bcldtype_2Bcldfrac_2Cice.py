# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2024-01-18 17:28:50

import gc
import os

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.gridspec import GridSpec
from muqy_20240101_2Bgeoprof_reader import Reader
from muqy_20240101_plot_2Bgeoprof_test import draw_cross_section


def cld_type_processor(
    cloud_type,
    cloud_base,
    cloud_top,
    required_types=[0, 1, 2, 3, 4, 5, 6, 7, 8],
):
    """
    Modify cloud data arrays by setting unrequired types to NaN, while keeping the original shapes.

    Args:
        cloud_type (np.ndarray): Array of cloud types.
        cloud_base (np.ndarray): Array of cloud base heights.
        cloud_top (np.ndarray): Array of cloud top heights.
        cf (np.ndarray): Cloud fraction data.
        hgt (np.ndarray): Heights of each floor in cf.
        required_types (list): List of cloud types to retain.

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray: Modified arrays of cloud types, bases, tops, and cf data.
    """
    # Mask for required cloud types
    type_mask = np.isin(cloud_type, required_types)

    # Applying mask to cloud type, base, and top arrays
    # Unrequired types set to NaN
    filtered_cloud_type = np.where(type_mask, cloud_type, np.nan)
    filtered_cloud_base = np.where(type_mask, cloud_base, np.nan)
    filtered_cloud_top = np.where(type_mask, cloud_top, np.nan)

    return (
        filtered_cloud_type,
        filtered_cloud_base,
        filtered_cloud_top,
    )


def set_map(ax):
    """
    Set up the map on the provided axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to set up the map on.
    """
    # Set up the map
    proj = ccrs.PlateCarree()

    # Set up the coastlines
    ax.coastlines(lw=0.5)

    # Set up the gridlines
    xticks = np.arange(-180, 181, 60)
    yticks = np.arange(-90, 91, 30)

    # Set up the x and y tick labels
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Set up the font size
    ax.tick_params("both", labelsize=14)
    ax.set_global()


def draw_track(ax, lon1D, lat1D):
    """
    Draw the satellite track based on latitude and longitude.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to draw the track on.
    lon1D : np.ndarray
        Array of longitudes.
    lat1D : np.ndarray
        Array of latitudes.
    """
    # Draw the track
    ax.plot(lon1D, lat1D, lw=2, color="b", transform=ccrs.Geodetic())

    # Draw the start point
    ax.plot(
        lon1D[0], lat1D[0], "ro", ms=3, transform=ccrs.PlateCarree()
    )
    # Add the start label
    ax.text(
        lon1D[0] + 5,
        lat1D[0],
        "start",
        color="r",
        fontsize=14,
        transform=ccrs.PlateCarree(),
    )


def draw_cloud_profile(
    ax, time, base, top, cloud_type, hgt, cax=None
):
    """
    Draw the cloud profile based on cloud base, top, and type using a finer granularity.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to draw the cloud profile on.
    time : np.ndarray
        Array of time data.
    base : np.ndarray
        Array of cloud base heights.
    top : np.ndarray
        Array of cloud top heights.
    cloud_type : np.ndarray
        Array of cloud types.
    hgt : np.ndarray
        Array of height data.
    """
    # Define a discrete colormap for cloud types
    cloud_colors = [
        "white",  # 0: Not determined
        "lavender",  # 1: Cirrus
        "lightblue",  # 2: Altostratus
        "lightgreen",  # 3: Altocumulus
        "beige",  # 4: Stratus
        "yellow",  # 5: Stratocumulus
        "orange",  # 6: Cumulus
        "pink",  # 7: Nimbostratus
        "red",  # 8: Deep Convection
    ]
    # Cloud type labels
    cloud_labels = {
        0: "Not determined",
        1: "Cirrus",
        2: "Altostratus",
        3: "Altocumulus",
        4: "Stratus",
        5: "Stratocumulus",
        6: "Cumulus",
        7: "Nimbostratus",
        8: "Deep Convection",
    }

    cmap = mcolors.ListedColormap(cloud_colors)
    bounds = np.arange(-0.5, 9.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create a cloud type array compatible with pcolormesh
    cloud_type_mesh = np.full(
        (cloud_type.shape[0], hgt.shape[0]), np.nan
    )

    # Fill in the cloud type array
    for nray in range(cloud_type.shape[0]):  # nray
        for ncloud in range(cloud_type.shape[1]):  # ncloud
            if ~np.isnan(cloud_type[nray, ncloud]):
                # Get the cloud base and top heights
                base_hgt = base[nray, ncloud]
                top_hgt = top[nray, ncloud]
                # Find the height indices corresponding to the cloud layer
                height_indices = (base_hgt <= hgt[:]) & (
                    hgt[:] <= top_hgt
                )
                cloud_type_mesh[nray, height_indices] = cloud_type[
                    nray, ncloud
                ]

    # Plot the cloud types
    im = ax.pcolormesh(
        time, hgt, cloud_type_mesh.T, cmap=cmap, norm=norm
    )

    # Set plot limits and labels
    ax.set_ylim(
        0, 20
    )  # Assuming height is in km and max height is 20 km
    ax.set_xlim(time.min(), time.max())
    ax.tick_params(labelsize=11)
    ax.set_xlabel(
        "Seconds since the start of the granule [s]", fontsize=12
    )
    ax.set_ylabel("Height (km)", fontsize=12)

    # Add a colorbar
    cbar = plt.colorbar(
        im, ax=ax, cax=cax, boundaries=bounds, ticks=np.arange(0, 9)
    )
    cbar.ax.set_yticklabels(
        [cloud_labels.get(i, "") for i in range(9)]
    )  # Cloud type labels
    cbar.ax.tick_params(labelsize=11)


def draw_elevation(ax, time, elv):
    """
    Draw the elevation profile.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to draw the elevation on.
    time : np.ndarray
        Array of time data.
    elv : np.ndarray
        Array of elevation data.
    """
    ax.fill_between(time, elv, color="gray")


def filter_by_time_range(data, time, start, end):
    """
    Filter data based on a specified time range.

    Parameters:
        data (array-like): The data to be filtered.
        time (array-like): The time values corresponding to the data.
        start (float): The start time of the range.
        end (float): The end time of the range.

    Returns:
        tuple: A tuple containing the filtered data and corresponding time values.
    """
    mask = (time >= start) & (time <= end)
    return data[mask], time[mask]


def read_CloudSat_data(file_path):
    """
    Read CloudSat data from a file.

    Args:
        file_path (str): Path to the CloudSat data file.

    Returns:
        dict: Dictionary containing the read data.
    """
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
    finally:
        f.close()

    return data


def filter_and_plot_data(overall_data, time_range_custum):
    """
    Filter and plot CloudSat data.

    Args:
        overall_data (dict): Dictionary containing all CloudSat data.
        time_range_custum (int): Custom time range for filtering data.

    Returns:
        None
    """
    # ----------------------------------------------------------------
    # set the time range
    # Customizable start time (example: start from the first timestamp in the dataset)
    start_time_custom = overall_data["time"][0]
    time_range_end = start_time_custom + time_range_custum

    # Data names to filter
    data_to_filter = [
        "cf",
        "re",
        "IWC",
        "cld_layer_base",
        "cld_layer_top",
        "cld_layer_type",
        "lon",
        "lat",
        "elv",
    ]

    # Initialize a dictionary to store filtered data
    filtered_data = {}

    # Filter the data using the custom start time
    _, time_filtered = filter_by_time_range(
        overall_data["cf"],
        overall_data["time"],
        start_time_custom,
        time_range_end,
    )

    # Apply the filter to each data set
    for data_name in data_to_filter:
        filtered_data[data_name], _ = filter_by_time_range(
            overall_data[data_name],
            overall_data["time"],
            start_time_custom,
            time_range_end,
        )

    # Unpack the filtered data
    cf_time_filtered, re_time_filtered, IWC_time_filtered = (
        filtered_data["cf"],
        filtered_data["re"],
        filtered_data["IWC"],
    )
    (
        cld_layer_base_time_filtered,
        cld_layer_top_time_filtered,
        cld_layer_type_time_filtered,
    ) = (
        filtered_data["cld_layer_base"],
        filtered_data["cld_layer_top"],
        filtered_data["cld_layer_type"],
    )
    lon, lat, elv = (
        filtered_data["lon"],
        filtered_data["lat"],
        filtered_data["elv"],
    )

    # ----------------------------------------------------------------
    # Plot the data
    # set font
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]

    fig = plt.figure(figsize=(9, 14), dpi=330)

    # Set up the grid
    gs = GridSpec(
        5,
        2,
        figure=fig,
        width_ratios=[30, 1],
        wspace=0.09,
        hspace=0.5,
    )

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[4, 0])

    # Create additional axes for colorbars to the right of the subplots
    cbar_ax2 = fig.add_subplot(gs[1, 1])
    cbar_ax3 = fig.add_subplot(gs[2, 1])
    cbar_ax4 = fig.add_subplot(gs[3, 1])
    cbar_ax5 = fig.add_subplot(gs[4, 1])

    # Set up the map on the first subplot
    set_map(ax1)
    draw_track(ax1, lon, lat)

    # Calculate the start time of the filtered data
    start_str = overall_data["start_time"].strftime("%Y-%m-%d %H:%M")

    # Calculate the end time of the filtered data
    end_time_filtered = overall_data["start_time"] + pd.Timedelta(
        seconds=time_range_custum
    )
    end_str = end_time_filtered.strftime("%Y-%m-%d %H:%M")

    # Set title
    ax1.set_title(f"From {start_str} to {end_str}", fontsize=14)

    # Add a new subplot for cloud types
    cbar_ax2.set_visible(False)
    draw_cloud_profile(
        ax2,
        time_filtered,
        cld_layer_base_time_filtered,
        cld_layer_top_time_filtered,
        cld_layer_type_time_filtered,
    )

    # Add a new subplot for lidar-radar cross-section
    draw_cross_section(
        ax3,
        time_filtered,
        overall_data["hgt"],
        cf_time_filtered,
        colormap="Spectral_r",
        cbar_label="Cloud Fraction (%)",
        cbar_orientation="vertical",
        cax=cbar_ax3,
    )
    # Add a new subplot for microphysical properties for ice clouds
    draw_cross_section(
        ax4,
        time_filtered,
        overall_data["hgt"],
        re_time_filtered,
        colormap="Purples",
        cbar_label="Effective Radius (μm)",
        cbar_orientation="vertical",
        cax=cbar_ax4,
    )
    draw_cross_section(
        ax5,
        time_filtered,
        overall_data["hgt"],
        IWC_time_filtered,
        colormap="Blues",
        cbar_label="IWC (g/m" + r"$^3$)",
        cbar_orientation="vertical",
        cax=cbar_ax5,
    )

    # Draw elevation in each subplot
    draw_elevation(
        ax2, time_filtered, elv / 1000
    )  # Convert elevation to km
    ax2.set_title("Cloud Types", fontsize=14)
    draw_elevation(
        ax3, time_filtered, elv / 1000
    )  # Convert elevation to km
    ax3.set_title("Cloud Faction", fontsize=14)
    draw_elevation(
        ax4, time_filtered, elv / 1000
    )  # Convert elevation to km
    ax4.set_title("Retreived Effective Radius", fontsize=14)
    draw_elevation(
        ax5, time_filtered, elv / 1000
    )  # Convert elevation to km
    ax5.set_title("Retreived IWC", fontsize=14)

    # Save the figure
    start_str_date = overall_data["start_time"].strftime(
        "%Y_%m_%d_%H_%M_%S"
    )
    fig.savefig(
        f"../Fig_python/CloudSat_2Bcldtype_2Cice_2Bgeoprof_{start_str_date}.png",
        dpi=330,
        bbox_inches="tight",
        facecolor="w",
    )
    # totally close the figure
    plt.close("all")
    # close all the figure windows to save memory
    plt.close()


def plot_multi_CloudSat_data(
    base_paths, file_idx, time_range_custum=5000
):
    """
    Plot multiple CloudSat data variables.

    Args:
        file_idx (int): The index of the file to be plotted.
        time_range_custum (int, optional): The custom time range in seconds. Defaults to 5000.

    Returns:
        None
    """
    # Set the file index
    file_paths = {
        key: os.path.join(path, sorted(os.listdir(path))[file_idx])
        for key, path in base_paths.items()
    }

    overall_data = {}
    for _, file_path in file_paths.items():
        overall_data.update(read_CloudSat_data(file_path))

    (
        overall_data["cld_layer_type"],
        overall_data["cld_layer_base"],
        overall_data["cld_layer_top"],
    ) = cld_type_processor(
        overall_data["cld_layer_type"],
        overall_data["cld_layer_base"],
        overall_data["cld_layer_top"],
        required_types=[1, 2, 3, 4, 5, 6, 7, 8],
    )

    filter_and_plot_data(overall_data, time_range_custum)

    # delete the data in memory to save memory
    del overall_data
    gc.collect()  # Explicit garbage collection
    # close all the figure windows to save memory
    plt.close()
    plt.close("all")


if __name__ == "__main__":
    # Set the file base paths
    base_paths = {
        "cld_type": "../Data_python/2B_CLDCLASS_lidar/",
        "cld_ice": "../Data_python/2C_ICE/",
        "cld_fraction": "../Data_python/2B_GEOPROF_lidar/",
    }

    # Execute the function for each file
    # for idx in range(2, 4):
