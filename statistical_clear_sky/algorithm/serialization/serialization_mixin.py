"""
This module defines Mixin for serialization.
"""
import json
import numpy as np
from statistical_clear_sky.algorithm.serialization.state_data import StateData

class SerializationMixin(object):
    """
    Mixin for IterativeClearSky, taking care of serialization.
    """

    def save_instance(self, filepath):
        save_dict = dict(
            auto_fix_time_shifts = self._state_data.auto_fix_time_shifts,
            power_signals_d = self._state_data.power_signals_d.tolist(),
            rank_k = self._state_data.rank_k,
            matrix_l0 = self._state_data.matrix_l0.tolist(),
            matrix_r0 = self._state_data.matrix_r0.tolist(),
            l_value = self._state_data.l_value.tolist(),
            r_value = self._state_data.r_value.tolist(),
            beta_value = float(self._state_data.beta_value),
            component_r0 = self._state_data.component_r0.tolist(),
            mu_l = self._state_data.mu_l,
            mu_r = self._state_data.mu_r,
            tau = self._state_data._tau,
            is_solver_error = self._state_data.is_solver_error,
            is_problem_status_error = self._state_data.is_problem_status_error,
            f1_increase = self._state_data.f1_increase,
            obj_increase = self._state_data.obj_increase,
            residuals_median = self._state_data.residuals_median,
            residuals_variance = self._state_data.residuals_variance,
            residual_l0_norm = self._state_data.residual_l0_norm,
            weights = self._state_data.weights.tolist()
        )
        with open(filepath, 'w') as file:
            json.dump(save_dict, file)

    @classmethod
    def load_instance(cls, filepath):
        with open(filepath, 'r') as file:
            load_dict = json.load(file)

        power_signals_d = np.array(load_dict['power_signals_d'])
        rank_k = load_dict['rank_k']

        instance = cls(np.array(power_signals_d), rank_k=rank_k)

        instance.state_data.power_signals_d = power_signals_d
        instance.state_data.rank_k = rank_k
        instance.state_data.matrix_l0 = np.array(load_dict['matrix_l0'])
        instance.state_data.matrix_r0 = np.array(load_dict['matrix_r0'])

        instance.state_data.l_cs_value = np.array(load_dict['l_value'])
        instance.state_data.r_cs_value = np.array(load_dict['r_value'])
        instance.state_data.beta_value = load_dict['beta_value']

        instance.state_data.component_r0 = np.array(load_dict['component_r0'])

        instance.state_data.mu_l = load_dict['mu_l']
        instance.state_data.mu_r = load_dict['mu_r']
        instance.state_data.tau = load_dict['tau']

        instance.state_data.is_solver_error = load_dict['is_solver_error']
        instance.state_data.is_problem_status_error = load_dict[
            'is_problem_status_error']
        instance.state_data.f1_increase = load_dict['f1_increase']
        instance.state_data.obj_increase = load_dict['obj_increase']

        instance.state_data.residuals_median = load_dict['residuals_median']
        instance.state_data.residuals_variance = load_dict['residuals_variance']
        instance.state_data.residual_l0_norm = load_dict['residual_l0_norm']

        instance.state_data.weights = np.array(load_dict['weights'])

        instance._keep_result_variables_as_properties(instance.state_data.l_cs_value,
                                                      instance.state_data.r_cs_value,
                                                      instance.state_data.beta_value)
        instance._keep_supporting_parameters_as_properties(instance.state_data.weights)

        return instance
