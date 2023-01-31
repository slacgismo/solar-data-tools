import matplotlib.pyplot as plt
import seaborn as sns

def plot_decomp(signal, gt=None):
    """Signal must be a list of one or more components.
    Ground truth (gt) if passed, must be a list of the same 
    number of components.
    """
    signal = list(signal)
    if gt is not None:
        gt = list(gt)
    subplots = len(signal)
    
    sns.set_theme()
    sns.set(font_scale=0.8)
    fig, axs = plt.subplots(subplots, sharex=True, figsize=(8,6))

    for i in range(subplots):
        if subplots == 1:
            axs = [axs]
        axs[i].plot(signal[i], label="Estimated", linewidth=0, marker=".",  markersize=3)
        if gt:
            axs[i].plot(gt[i], label="True", linewidth=0, marker=".",  markersize=3)
            axs[i].legend()
        axs[i].set_title(f"component $x^{i+1}$")
        
        
    plt.tight_layout()