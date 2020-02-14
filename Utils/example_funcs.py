import numpy as np
import pandas as pd


def plot_loss_and_initial_final_histograms(ax, loss_iter, p_samples, q_samples, q_samples_init,
                                           title: str, number_of_bins: int = 15):
    total_iterations = loss_iter.shape[0]
    hist_color = '#377eb8'
    label = 'IGR-SB'
    # hist_color = '#984ea3'
    # label = 'GS with K = 12'
    ylim = 0.3
    # xlim = 12
    xlim = 70
    ax[0].set_title(title)
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')
    ax[0].plot(np.arange(total_iterations), loss_iter, alpha=0.2)
    window = 100 if total_iterations >= 500 else 10
    loss_df = pd.DataFrame(data=loss_iter).rolling(window=window).mean()
    ax[0].plot(np.arange(total_iterations), loss_df, label=f'mean over {window} iter')
    ax[0].legend()

    ax[1].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p', density=True)
    ax[1].hist(q_samples_init, bins=np.arange(number_of_bins), color=hist_color, alpha=0.5,
               label=label, density=True)
    ax[1].set_ylim([0, ylim])
    ax[1].set_xlim([0, xlim])
    ax[1].set_title('Initial distribution')
    ax[1].legend()

    ax[2].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p', density=True)
    ax[2].hist(q_samples, bins=np.arange(number_of_bins), color=hist_color, alpha=0.5, label=label, density=True)
    ax[2].set_title('Final distribution')
    ax[2].set_ylim([0, ylim])
    ax[2].set_xlim([0, xlim])
    ax[2].legend()


def plot_histograms_of_gs(ax, p_samples, q_samples_list, q_samples_init_list, number_of_bins: int = 15):
    colors = ['#c5a6fa', '#4e17aa', '#2c0d61']
    k = [20, 40, 100]
    # y_lim = 0.35
    # k = [10]
    y_lim = 0.2
    x_lim = 70
    ax[0].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p',
               density=True)
    for i in range(len(q_samples_init_list)):
        ax[0].hist(q_samples_init_list[i], bins=np.arange(number_of_bins), color=colors[i], alpha=0.5,
                   label=f'GS with K = {k[i]:d}', density=True)
    ax[0].set_ylim([0, y_lim])
    ax[0].set_xlim([0, x_lim])
    ax[0].set_title('Initial distribution')
    # ax[0].set_ylabel('Normalized Counts')
    ax[0].legend()

    ax[1].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p',
               density=True)
    for i in range(len(q_samples_list)):
        ax[1].hist(q_samples_list[i], bins=np.arange(number_of_bins), color=colors[i], alpha=0.5,
                   label=f'GS with K = {k[i]:d}', density=True)
    ax[1].set_title('Final distribution')
    ax[1].set_ylim([0, y_lim])
    ax[1].set_xlim([0, x_lim])
    # ax[1].set_ylabel('Normalized Counts')
    ax[1].legend()
