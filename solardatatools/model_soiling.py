"""
simulate_PV_time_series: Model for generating PV soiling time series described in Deceglie et. al.
    "Numerical Validation of an Algorithm for Combined Soiling and Degradation Analysis of Photovoltaic
    Systems", Proceedings of the 2019 IEEE PVSC.
simulate_PV_time_series_gaussian_soiling: Model for generating PV soiling time series with soiling
    rates that vary from day to day according to a Gaussian distribution (Åsmund)
generate_my_soiling_signals: A function that generates 6 different time series based on the previous
    two functions. This was used to generate the datasets used in
    A. Skomedal, M. G. Deceglie, 2020, "Combined Estimation of Degradation and Soiling Losses in PV
    Systems"
Authors: Michael Deceglie and Åsmund Skomedal (Sep 2019)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simulate_PV_time_series(
    first_date,
    last_date,
    freq="1D",
    degradation_rate=-0.005,
    noise_scale=0.01,
    seasonality_scale=0.01,
    nr_of_cleaning_events=120,
    soiling_rate_low=0.0001,
    soiling_rate_high=0.003,
    smooth_rates=False,
    random_seed=False,
):

    if random_seed:  # Have seed for repeatability
        if not type(np.random.seed) == int:
            np.random.seed(int(random_seed))

    # Initialize time series and data frame
    times = pd.date_range(first_date, last_date, freq=freq)
    df = pd.DataFrame(
        index=times,
        columns=[
            "day",
            "noise",
            "seasonality",
            "degradation",
            "soiling",
            "cleaning_events",
        ],
    )
    n = len(times)

    df["day"] = range(n)
    df["noise"] = np.random.normal(1.0, scale=noise_scale, size=n)
    df["seasonality"] = seasonality_scale * np.sin(df["day"] / 365.25 * 2 * np.pi) + 1
    df["degradation"] = 1 + degradation_rate / 365.25 * df["day"]

    # simulate soiling
    cleaning_events = np.random.choice(
        df["day"], nr_of_cleaning_events, replace="False"
    )
    cleaning_events = np.sort(cleaning_events)

    x = np.full(n, 1)
    intervals = np.split(x, cleaning_events)
    soiling_rate = []
    for interval in intervals:
        rate = np.random.uniform(
            low=soiling_rate_low, high=soiling_rate_high, size=1
        ) * np.ones(len(interval))
        soiling_rate.append(rate)
    df["soiling_rate"] = np.concatenate(soiling_rate)
    if smooth_rates:
        df.soiling_rate = (
            df.soiling_rate.rolling(smooth_rates, center=True).mean().ffill().bfill()
        )
    derate = np.concatenate(
        [np.cumsum(si) for si in np.split(df["soiling_rate"].values, cleaning_events)]
    )
    df["soiling"] = 1 - derate

    # Generate Performance Indexes
    df["PI_no_noise"] = df["seasonality"] * df["degradation"] * df["soiling"]
    df["PI_no_soil"] = df["seasonality"] * df["degradation"] * df["noise"]
    df["PI_no_degrad"] = df["seasonality"] * df["soiling"]
    df["daily_norm"] = df["noise"] * df["PI_no_noise"]

    return df


def simulate_PV_time_series_seasonal_soiling(
    first_date,
    last_date,
    freq="1D",
    degradation_rate=-0.005,
    noise_scale=0.01,
    seasonality_scale=0.01,
    nr_of_cleaning_events=120,
    soiling_rate_center=0.002,
    soiling_rate_std=0.001,
    soiling_seasonality_scale=0.9,
    random_seed=False,
    smooth_rates=False,
    seasonal_rates=False,
):
    """As the name implies, this function models soiling rates that vary from day to day according
    to a Gaussian distribution"""
    if random_seed:  # Have seed for repeatability
        if not type(np.random.seed) == int:
            np.random.seed(int(random_seed))

    # Initialize time series and data frame
    times = pd.date_range(first_date, last_date, freq=freq)
    df = pd.DataFrame(index=times)
    n = len(times)

    df["day"] = range(n)
    df["noise"] = np.random.normal(1.0, scale=noise_scale, size=n)
    df["seasonality"] = seasonality_scale * np.sin(df["day"] / 365.25 * 2 * np.pi) + 1
    df["degradation"] = 1 + degradation_rate / 365.25 * df["day"]

    soiling_seasonality = (
        soiling_rate_center
        * soiling_seasonality_scale
        * np.sin(df["day"] / 365.25 * 2 * np.pi + np.random.uniform(10, size=1) * np.pi)
    )
    cleaning_probability = (
        2 - soiling_seasonality_scale
    ) * soiling_seasonality.max() + soiling_seasonality
    cleaning_probability /= cleaning_probability.sum()

    # simulate soiling
    cleaning_events = np.random.choice(
        df["day"].values,
        nr_of_cleaning_events,
        replace="False",
        p=cleaning_probability.values,
    )
    cleaning_events = np.sort(cleaning_events)
    df["cleaning_events"] = [
        True if day in cleaning_events else False for day in df.day
    ]

    x = np.full(n, 1)
    intervals = np.split(x, cleaning_events)
    soiling = []
    soiling_rate = []
    for interval in intervals:
        if len(soiling) > 0:
            pos = len(np.concatenate(soiling))
        else:
            pos = 0
        if seasonal_rates:
            if smooth_rates:
                soiling_season = soiling_seasonality[pos : pos + len(interval)]
                rate = np.random.normal(
                    soiling_rate_center + soiling_season,
                    soiling_rate_std,
                    size=len(interval),
                )
            else:
                soiling_season = soiling_seasonality[pos]
                rate = np.random.normal(
                    soiling_rate_center + soiling_season, soiling_rate_std, size=1
                ) * np.ones(len(interval))
        else:
            if smooth_rates:
                print("Smooth rates not possible without seasonal rates")
            soiling_season = 0
            rate = np.random.normal(
                soiling_rate_center + soiling_season, soiling_rate_std, size=1
            ) * np.ones(len(interval))
        derate = 1 - np.matmul(half_one_matrix(len(interval)), rate)
        soiling.append(derate)
        soiling_rate.append(rate)
    df["soiling"] = np.concatenate(soiling)
    df["soiling_rate"] = np.concatenate(soiling_rate)

    # Generate Performance Indexes
    df["PI_no_noise"] = df["seasonality"] * df["degradation"] * df["soiling"]
    df["PI_no_soil"] = df["seasonality"] * df["degradation"] * df["noise"]
    df["PI_no_degrad"] = df["seasonality"] * df["soiling"]
    df["daily_norm"] = df["noise"] * df["PI_no_noise"]

    return df


def generate_my_soiling_signals(
    high=0.02, low=0.01, num_years=10, index_form="Datetime"
):
    first_date = "2010/01/01"
    last_date = str(2010 + num_years) + "/01/01"
    noises = [low, low, high, low, low, low]
    seasonality_scales = [low, high, low, low, low, low]
    soiling_levels = [0.003, 0.001, 0.001, 0.0005, 0.0005, 0.0001]
    names = [
        "normal",
        "M Soil, H season",
        "M Soil, H noise",
        "Seasonal cleaning",
        "L/M Soil (.0005)",
        "Low Soil (.0001)",
    ]
    dfs = []
    for i in range(6):
        if i == 3:
            df = simulate_PV_time_series_seasonal_soiling(
                first_date,
                last_date,
                noise_scale=noises[i],
                seasonality_scale=seasonality_scales[i],
                soiling_rate_std=0,
                soiling_rate_center=soiling_levels[i],
                soiling_seasonality_scale=0.95,
                smooth_rates=False,
                seasonal_rates=False,
            )
        else:
            df = simulate_PV_time_series(
                first_date,
                last_date,
                noise_scale=noises[i],
                seasonality_scale=seasonality_scales[i],
                nr_of_cleaning_events=120,
                soiling_rate_low=0,
                soiling_rate_high=soiling_levels[i],
                degradation_rate=-0.005,
            )
        if index_form == "numeric":
            df.index = np.arange(len(df))

        dfs.append(df)

    return dfs, names


def half_one_matrix(size):
    dummy = np.zeros((size, size))
    dummy[np.triu_indices(size, k=1)] = 1
    return dummy.T


if __name__ == "__main__":
    # Visualize my soiling signals
    np.random.seed(21)
    dfs, names = generate_my_soiling_signals()
    names = [
        "Base case \n(max soiling rate = 0.3 %/d)",
        "Double seasonality \n(max soiling rate = 0.1 %/d)",
        "Double noise \n(max soiling rate = 0.1 %/d)",
        "Seasonal cleaning \n(constant soiling rate = 0.05 %/d)",
        "Little soiling \n(max soiling rate = 0.05 %/d)",
        "Very little soiling \n(max soiling rate = 0.01 %/d)",
    ]
    N = len(names)
    spn = 5 if N in [5, 9, 10] else (3 if N in [6] else 4)
    plt.close("all")
    for i in range(N):
        df = dfs[i]
        if i % spn == 0:
            fig, ax = plt.subplots(spn, 1, figsize=(10, 6), sharex=True)
        ax[i % spn].scatter(df.index, df.daily_norm, 5, alpha=0.5)
        ax[i % spn].plot(df.index, df.PI_no_noise, "k", alpha=0.5)
        ax[i % spn].set_ylim(0.8, 1.06)
        ax[i % spn].set_ylabel("PI")
        ax[i % spn].set_title(names[i])
        plt.tight_layout()
