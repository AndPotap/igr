import numpy as np
from matplotlib import pyplot as plt
from Utils.posterior_sample_funcs import sample_from_posterior
import seaborn as sns
import pandas as pd

dataset_name = 'mnist'
run_with_sample = True
plots_path = './Results/Outputs/'
path_to_results = './Results/Current_Model/'
output_plot = f'./Results/Outputs/posterior_samples_{dataset_name}_dis.png'
model_name = ['IGR_SB(0.5)']

diff_igr = sample_from_posterior(path_to_results=path_to_results, hyper_file='hyper_igr_sb_10_mnist.pkl',
                                 dataset_name=dataset_name, weights_file='vae_igr_sb_10.h5',
                                 model_type='IGR_SB_Dis', run_with_sample=run_with_sample)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Make boxplot
# ===========================================================================================================
plt.style.use(style='ggplot')
plt.figure(dpi=100)

rows_list = []
total_test_images = diff_igr.shape[0]
diff_list = [np.median(np.mean(diff_igr, axis=2), axis=1)]
for i in range(len(diff_list)):
    for s in range(total_test_images):
        entry = {'Model': model_name[i], 'Distance': diff_list[i][s]}
        rows_list.append(entry)
df = pd.DataFrame(rows_list)
ax = sns.boxplot(x='Model', y='Distance', data=df, color='royalblue', boxprops={'alpha': 0.5})

plt.ylabel('Euclidean Distance')
# plt.ylim([0.0, 0.5])
plt.xlabel('Models')
plt.legend()
plt.savefig(output_plot)
plt.tight_layout()
plt.show()
# ===========================================================================================================
