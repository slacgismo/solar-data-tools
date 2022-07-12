""" Equation of Time Module
The Equation of Time (EoT) describes the discrepancy between clock time and
solar time: https://en.wikipedia.org/wiki/Equation_of_time

This module contains two approaches to calculating the EoT: da_rosa and Duffie
"""
import numpy as np


def eot_da_rosa(day_of_year):
    """
    The equation of time as defined in:
        Haghdadi, Navid, et al. "A method to estimate the location and
        orientation of distributed photovoltaic systems from their generation
        output data." Renewable Energy 108 (2017): 390-400.

    These are equations (7) and (8) in the paper.

    :param day_of_year: the day of year, can be int, float, or numpy array
    :return: the difference between clock time and solar time for a given day of year
    """
    b = np.deg2rad((360 / 365) * (day_of_year - 81))
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    try:
        return eot.values
    except AttributeError:
        return eot


def eot_duffie(day_of_year):
    """
    The equation of time as defined in:
        Duffie, John A., and William A. Beckman. Solar engineering of thermal
        processes. New York: Wiley, 1991.

    These are equations (1.4.2) and (1.5.3) in the book

    :param day_of_year: the day of year, can be int, float, or numpy array
    :return: the difference between clock time and solar time for a given day of year
    """
    b = np.deg2rad((360 / 365) * (day_of_year - 1))
    A = 1440 / (2 * np.pi)  # book uses approximation of 229.2
    eot = A * (0.000075 + 0.001868 * np.cos(b) - 0.032077 * np.sin(b)
               - 0.014615 * np.cos(2 * b) - 0.04089 * np.sin(2 * b))
    try:
        return eot.values
    except AttributeError:
        return eot
