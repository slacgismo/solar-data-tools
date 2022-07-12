"""The function cos(theta) is  calculated using equation (1.6.2) in:
 Duffie, John A., and William A. Beckman. Solar engineering of thermal
 processes. New York: Wiley, 1991."""

import numpy as np


def func_costheta(x, phi, beta, gamma):
    delta = x[0]
    omega = x[1]

    gamma -= np.rint(gamma / 2 / np.pi) * 2 * np.pi

    a = np.sin(delta) * np.sin(phi) * np.cos(beta)
    b = np.sin(delta) * np.cos(phi) * np.sin(beta) * np.cos(gamma)
    c = np.cos(delta) * np.cos(phi) * np.cos(beta) * np.cos(omega)
    d = np.cos(delta) * np.sin(phi) * np.sin(beta) * np.cos(gamma) * np.cos(omega)
    e = np.cos(delta) * np.sin(beta) * np.sin(gamma) * np.sin(omega)
    return a - b + c + d + e
