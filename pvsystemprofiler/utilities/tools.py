import numpy as np
import pandas as pd


def random_initial_values(nrandom):
    lat_initial_value = np.random.uniform(low=-90, high=90, size=nrandom)
    tilt_initial_value = np.random.uniform(low=0, high=90, size=nrandom)
    azim_initial_value = np.random.uniform(low=-180, high=180, size=nrandom)
    return lat_initial_value, tilt_initial_value, azim_initial_value
