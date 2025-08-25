# -*- coding: utf-8 -*-
"""Plotting Module"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_2d(
    D,
    figsize=(12, 6),
    units="kW",
    clear_days=None,
    dates=None,
    year_lines=False,
    ax=None,
    color="red",
):
    """
    A function for plotting the power heat map for solar power data

    :param D: PV power data arranged as a matrix, typically the output of
        `data_transforms.make_2d()`
    :param figsize: the size of the desired figure (passed to `matplotlib`)
    :param units: the units of the power data
    :param clear_days: a boolean array marking the location of clear days in
        the data set, typically the output of `clear_day_detection.find_clear_days()`
    :return: `matplotlib` figure
    """
    if D is not None:
        with sns.axes_style("white"):
            if ax is None:
                fig, ax = plt.subplots(nrows=1, figsize=figsize)
            else:
                fig = ax.get_figure()
            foo = ax.imshow(D, cmap="plasma", interpolation="none", aspect="auto")
            ax.set_title("Measured power")
            plt.colorbar(foo, ax=ax, label=units)
            ax.set_xlabel("Day number")
            ax.set_yticks([])
            ax.set_ylabel("(sunset)        Time of day        (sunrise)")
            if clear_days is not None:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                use_day = clear_days
                days = np.arange(D.shape[1])
                y1 = np.ones_like(days[use_day]) * D.shape[0] * 0.999
                ax.scatter(days[use_day], y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.995 * y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.99 * y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.985 * y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.98 * y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.975 * y1, marker="|", color=color, s=2)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
        if dates is not None:
            if D.shape[1] >= 356 * 1.5:
                mask = np.logical_and(dates.month == 1, dates.day == 1)
                day_ticks = np.arange(D.shape[1])[mask]
                plt.xticks(day_ticks, dates[day_ticks].year)
                plt.xlabel("Year")
            else:
                mask = dates.day == 1
                day_ticks = np.arange(D.shape[1])[mask]
                plt.xticks(day_ticks, dates[day_ticks].month)
                plt.xlabel("Month")
            if year_lines:
                for d in day_ticks:
                    plt.axvline(d, ls="--", color="gray", linewidth=1)
        return fig
    else:
        return


def plot_bundt_cake(
    data,
    figsize=(12, 8),
    units="kW",
    inner_radius=1.0,
    slice_thickness=100,
    elev=45,
    azim=30,
    zoom=1.0,
    zscale=0.5,
    ax=None,
    cmap="coolwarm",
):
    """
    A function for plotting solar power data in a 3D “Bundt cake” style.
    Author: Mehmet Giray Ogut
    Date: June 11, 2025

    :param data: A 2D NumPy array of shape (365, N), where 365 is the number of days
    :param figsize: Size of the figure (width, height)
    :param units: Label for the z-axis
    :param inner_radius: Inner radius for the first slice
    :param slice_thickness: Total radial extent of all slices combined
    :param elev: Elevation angle for the 3D view
    :param azim: Azimuth angle for the 3D view
    :param zoom: Controls the box aspect ratio in x and y directions
    :param ax: Optional Matplotlib Axes object (3D projection). If None, a new figure is created
    :param cmap: Colormap used for the surface
    :return: Matplotlib Figure object
    """
    if data is None or data.shape[0] != 365:
        return

    with sns.axes_style("white"):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()

        num_days, slices_per_day = data.shape
        theta_days = np.linspace(0, 2 * np.pi, num_days, endpoint=True)
        theta_slices = np.linspace(0, 2 * np.pi, slices_per_day, endpoint=True)
        theta_grid, slice_grid = np.meshgrid(theta_days, theta_slices, indexing="xy")

        x = (inner_radius + slice_grid * (slice_thickness / slices_per_day)) * np.cos(
            theta_grid
        )
        y = (inner_radius + slice_grid * (slice_thickness / slices_per_day)) * np.sin(
            theta_grid
        )
        z = data.T

        ax.plot_surface(x, y, z, cmap=cmap, edgecolor="none")
        ax.grid(False)
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zlabel(units)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([zoom, zoom, zscale])
        plt.tight_layout()
        return fig
