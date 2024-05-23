# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-01-01 15:34:44
# @Last Modified by:   Muqy
# @Last Modified time: 2024-01-22 16:39:01

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from muqy_20240101_2Bgeoprof_reader import Reader


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


def draw_cross_section(
    ax,
    time,
    hgt,
    data,
    colormap="Spectral_r",
    cbar_label="Cloud Fraction (%)",
    cbar_orientation="vertical",
    cax=None,
    vmin=None,
    vmax=None,
):
    """
    Draw the time-height cross-section of the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to draw the cross-section on.
    time : np.ndarray
        Array of time data.
    hgt : np.ndarray
        Array of height data.
    data : np.ndarray
        The data to be plotted.
    colormap : str, optional
        The colormap used for the plot. Default is "Spectral_r".
    cbar_label : str, optional
        The label for the colorbar. Default is "Cloud Fraction (%)".
    cbar_orientation : str, optional
        Orientation of the colorbar ("vertical" or "horizontal"). Default is "vertical".
    """
    # Set up the colormap
    cmap = plt.get_cmap(colormap)
    cmap.set_bad("white", 1.0)
    im = ax.pcolormesh(
        time, hgt, data.T, cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_ylim(0, 22)
    ax.set_xlim(time.min(), time.max())
    ax.tick_params(labelsize=11)
    ax.set_xlabel(
        "Seconds since the start of the granule [s]", fontsize=12
    )
    ax.set_ylabel("Height (km)", fontsize=12)

    # Add a colorbar
    cbar = plt.colorbar(
        im, cax=cax, extend="both", orientation=cbar_orientation
    )
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(cbar_label, fontsize=12)


def draw_cross_section_lidar_radar(ax, time, hgt, data, uncertainty):
    """
    Draw the time-height cross-section of clouds captured by radar, lidar, or both.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to draw the cross-section on.
    time : np.ndarray
        Array of time data.
    hgt : np.ndarray
        Array of height data.
    data : np.ndarray
        The data to be plotted.
    uncertainty : np.ndarray
        The UncertaintyCF data indicating the source of detection.
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    # Create masked arrays for each category
    no_data = np.ma.masked_where(uncertainty != 0, data, copy=True)
    radar_only = np.ma.masked_where(uncertainty != 1, data, copy=True)
    lidar_only = np.ma.masked_where(uncertainty != 2, data, copy=True)
    both = np.ma.masked_where(uncertainty != 3, data, copy=True)

    # Define colors for each category
    color_none = "white"
    color_radar = "blue"
    color_lidar = "green"
    color_both = "orange"

    # Plot each category
    ax.pcolormesh(
        time, hgt, no_data.T, color=color_none, shading="nearest"
    )
    ax.pcolormesh(
        time,
        hgt,
        radar_only.T,
        color=color_radar,
        shading="nearest",
    )
    ax.pcolormesh(
        time,
        hgt,
        lidar_only.T,
        color=color_lidar,
        shading="nearest",
    )
    ax.pcolormesh(
        time, hgt, both.T, color=color_both, shading="nearest"
    )

    ax.set_ylim(0, 20)
    ax.set_xlim(0, time.max())
    ax.tick_params(labelsize=11)
    ax.set_xlabel(
        "Seconds since the start of the granule [s]", fontsize=12
    )
    ax.set_ylabel("Height (km)", fontsize=12)

    # Create a custom color bar
    colors = [color_none, color_radar, color_lidar, color_both]
    labels = ["None", "Radar Only", "Lidar Only", "Both"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="10%", pad=0.5)
    cbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=[0.5, 1.5, 2.5, 3.5],
    )
    cbar.ax.set_xticklabels(labels)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Cloud Detection Source", fontsize=12)


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


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Read the data
    fname = "../Data_python/2ZzlnD_0001_0303/2010038101625_20112_CS_2B-GEOPROF-LIDAR_GRANULE_P2_R05_E03_F00.hdf"

    # Create a Reader object
    f = Reader(fname)

    # Read the Geolocation fields
    lon, lat, elv = f.read_geo()

    # Read the time and height fields
    height = f.read_sds("Height")
    time = f.read_time(datetime=False)
    start_time, end_time = f.read_time(datetime=True)[[0, -1]]
    # Calculate the mean height and convert it to km
    hgt = np.nanmean(height, axis=0) / 1000  # Convert height to km

    # Read the Data fields
    cf = f.read_sds("CloudFraction")
    # Read the UncertaintyCF field
    uncertainty_cf = f.read_sds("UncertaintyCF")

    # Close the HDF file
    f.close()

    # ----------------------------------------------------------------
    # set the time range
    time_range = 200  # seconds

    # Filter data within the specified time range
    time_range_end = time[0] + time_range

    cf_time_filtered, time_filtered = filter_by_time_range(
        cf, time, time[0], time_range_end
    )
    uncertainty_cf_time_filtered = filter_by_time_range(
        uncertainty_cf, time, time[0], time_range_end
    )[0]
    lon, lat, elv = (
        filter_by_time_range(lon, time, time[0], time_range_end)[0],
        filter_by_time_range(lat, time, time[0], time_range_end)[0],
        filter_by_time_range(elv, time, time[0], time_range_end)[0],
    )
    hgt = np.nanmean(height, axis=0) / 1000  # Convert height to km

    # ----------------------------------------------------------------
    # Plot the data
    # set font
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]

    fig = plt.figure(figsize=(8, 10), dpi=330)
    ax1 = fig.add_subplot(311, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    set_map(ax1)
    draw_track(ax1, lon, lat)

    start_str = start_time.strftime("%Y-%m-%d %H:%M")
    end_str = end_time.strftime("%Y-%m-%d %H:%M")
    ax1.set_title(f"From {start_str} to {end_str}", fontsize=14)

    draw_cross_section(ax2, time_filtered, hgt, cf_time_filtered)
    draw_elevation(
        ax2, time_filtered, elv / 1000
    )  # Convert elevation to km
    ax2.set_title("Cloud Fraction", fontsize=14)

    # Add a new subplot for lidar-radar cross-section
    draw_cross_section_lidar_radar(
        ax3,
        time_filtered,
        hgt,
        cf_time_filtered,
        uncertainty_cf_time_filtered,
    )

    plt.tight_layout()
    plt.show()
