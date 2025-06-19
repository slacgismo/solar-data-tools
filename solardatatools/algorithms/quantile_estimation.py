""" PV Power Quantile Estimation Module

This module is for estimating time-varying quantiles of PV power data. 
This module also include the ability to use the fit quantiles to transform the data

Author: Bennet Meyers
Date: 6/19/25
"""

from solardatatools.algorithms import Dilation
from solardatatools.algorithms.dilation import undilate_quantiles

class PVQuantiles:
    def __init__(
        self, 
        data_handler,
        nvals_dil=101,
        num_harmonics=[10, 3], 
        regularization=0.1, 
        solver='CLARABEL', 
        verbose=False
    )
    """
    A class for estimating time-varying quantiles of PV power data

    :param data_handler: a DataHandler object with the initial pipeline run
    :type data_handler: solardatatools.DataHandler
    :param nvals_dil: the number of data points to use for daily dilation
    :type nvals_dil: int
    :param quantile_levels: the quantile levels to estimate, default is to estimate decades
    :type quantile_levels: list
    :param num_harmonics: the number of Fourier harmonics to use for the daily (first) and yearly (second) periods
    :type num_harmonics: list
    :param regularization: stiffness weight for quantile fits (larger is more stiff)
    :type regularization: float
    :param solver: the cvxpy solver to invoke, default is CLARABEL
    :type solver: string
    :param verbose: print solver status
    :type verbose: boolean
    """
    dil = Dilation(data_handler, nvals_dil=nvals_dil)
    self.dilation_object = dil
    self.sig_dilated = dil.signal_dil
    self.sig_original = dil.signal_ori
    self.nvals_dil = nvals_dil
    self.quantile_levels = None
    self.num_harmonics = num_harmonics
    self.regularization = regularization
    self.solver = solver
    self.verbose = verbose
    self.quantiles_original = {}
    self.quantiles_dilated = {}
    self.spq_object = None

    def estimate_quantiles(
        self,
        quantile_levels=[0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.98],
    ):
        spq = SmoothPeriodicQuantiles(
            num_harmonics=self.num_harmonics,
            periods=[self.nvals_dil, 365.24225*self.nvals_dil],
            standing_wave=[True, False],
            trend=False,
            quantiles=quantile_levels,
            weight=self.regularization,
            problem='sequential',
            solver=self.solver,
            verbose=self.verbose,
        )
        spq.fit(self.sig_dilated)
        if verbose:
            print("Quantiles estimated successfully.")
        quantiles_ori = undilate_quantiles(
             self.dilation_object.idx_ori,
             self.dilation_object.idx_dil,
             spq.fit_quantiles,
             nvals_dil=self.nvals_dil
        )
        for i, q in enumerate(quantile_levels):
            self.quantiles_dilated[q] = spq.fit_quantiles[:, i].squeeze()
            self.quantiles_original[q] = quantiles_ori[:, i].squeeze()
        self.spq_object = spq
        if self.quantile_levels is None:
            self.quantile_levels = quantile_levels
        else:
            self.quantile_levels = list(set(self.quantile_levels + quantile_levels))
            self.quantile_levels.sort()
    
    def plot_quantiles(self, dilated=True):
        q = self.quantile_levels
        if dilated:
            fq = np.empty((len(self.sig_dilated)), float))
            for i, q in enumerate(self.quantile_levels):
                fq[:, i] = self.quantiles_dilated[q]
            sig = self.sig_dilated
        else:
            fq = np.empty((len(self.sig_original)), float))
            for i, q in enumerate(self.quantile_levels):
                fq[:, i] = self.quantiles_original[q]
            sig = self.sig_original
        nq = fq.shape[1]
        nrows = (nq + 2) // 2
        fig, ax = plt.subplots(nrows, 2, figsize=(12, 3*nrows))
        if nrows > 1:
            for i in range(len(q)):
                sns.heatmap(fq[1:, i].reshape((nvals_dil, ndays), order='F'), ax=ax[i//2, i%2], cmap='plasma')
                ax[i//2, i%2].set_title(f'Quantile {q[i]}')
            sns.heatmap(sig[1:].reshape((nvals_dil, ndays), order='F'), ax=ax[-1, -1], cmap='plasma')
            ax[-1, -1].set_title('Dilated signal')
        else:
            for i in range(len(q)):
                sns.heatmap(fq[1:, i].reshape((nvals_dil, ndays), order='F'), ax=ax[i], cmap='plasma')
                ax[i].set_title(f'Quantile {q[i]}')
            sns.heatmap(sig[1:].reshape((nvals_dil, ndays), order='F'), ax=ax[-1], cmap='plasma')
            ax[-1].set_title('Dilated signal')
        plt.tight_layout()
        return plt.gcf()
