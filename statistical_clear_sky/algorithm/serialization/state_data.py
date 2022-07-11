"""
This module defines a class that holds the current state of algorithm object.
"""
import numpy as np

class StateData(object):
    """
    Holds the data to be serialized.
    """

    def __init__(self):
        self._auto_fix_time_shifts = True

        self._power_signals_d = None
        self._rank_k = None
        self._matrix_l0 = None
        self._matrix_r0 = None
        self._l_value = np.array([])
        self._r_value = np.array([])
        self._beta_value = 0.0
        self._component_r0 = np.array([])
        self._mu_l = None
        self._mu_r = None
        self._tau = None

        self._is_solver_error = False
        self._is_problem_status_error = False
        self._f1_increase = False
        self._obj_increase = False

        self._residuals_median = None
        self._residuals_variance = None
        self._residual_l0_norm = None
        self._weights = np.array([])

    @property
    def auto_fix_time_shifts(self):
        return self._auto_fix_time_shifts

    @auto_fix_time_shifts.setter
    def auto_fix_time_shifts(self, value):
        self._auto_fix_time_shifts = value

    @property
    def power_signals_d(self):
        return self._power_signals_d

    @power_signals_d.setter
    def power_signals_d(self, value):
        self._power_signals_d = value

    @property
    def rank_k(self):
        return self._rank_k

    @rank_k.setter
    def rank_k(self, value):
        self._rank_k = value

    @property
    def matrix_l0(self):
        return self._matrix_l0

    @matrix_l0.setter
    def matrix_l0(self, value):
        self._matrix_l0 = value

    @property
    def matrix_r0(self):
        return self._matrix_r0

    @matrix_r0.setter
    def matrix_r0(self, value):
        self._matrix_r0 = value

    @property
    def l_value(self):
        return self._l_value

    @l_value.setter
    def l_value(self, value):
        self._l_value = value

    @property
    def r_value(self):
        return self._r_value

    @r_value.setter
    def r_value(self, value):
        self._r_value = value

    @property
    def beta_value(self):
        return self._beta_value

    @beta_value.setter
    def beta_value(self, value):
        self._beta_value = value

    @property
    def component_r0(self):
        return self._component_r0

    @component_r0.setter
    def component_r0(self, value):
        self._component_r0 = value

    @property
    def mu_l(self):
        return self._mu_l

    @mu_l.setter
    def mu_l(self, value):
        self._mu_l = value

    @property
    def mu_r(self):
        return self._mu_r

    @mu_r.setter
    def mu_r(self, value):
        self._mu_r = value

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = value

    @property
    def is_solver_error(self):
        return self._is_solver_error

    @is_solver_error.setter
    def is_solver_error(self, value):
        self._is_solver_error = value

    @property
    def is_problem_status_error(self):
        return self._is_problem_status_error

    @is_problem_status_error.setter
    def is_problem_status_error(self, value):
        self._is_problem_status_error = value

    @property
    def f1_increase(self):
        return self._f1_increase

    @f1_increase.setter
    def f1_increase(self, value):
        self._f1_increase = value

    @property
    def obj_increase(self):
        return self._obj_increase

    @obj_increase.setter
    def obj_increase(self, value):
        self._obj_increase = value

    @property
    def residuals_median(self):
        return self._residuals_median

    @residuals_median.setter
    def residuals_median(self, value):
        self._residuals_median = value

    @property
    def residuals_variance(self):
        return self._residuals_variance

    @residuals_variance.setter
    def residuals_variance(self, value):
        self._residuals_variance = value

    @property
    def residual_l0_norm(self):
        return self._residual_l0_norm

    @residual_l0_norm.setter
    def residual_l0_norm(self, value):
        self._residual_l0_norm = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value
