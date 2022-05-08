from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)

experiment_id = "Ab6ml7pAQeGrZvT0d5QVqw"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
print(type(experiment))
df = experiment.get_scalars()
options = ['tdt4265_focal_loss_res34']
#options = ['tdt4265_augmented']
#options = ['tdt4265_init_weights_res34_new_weights']
#options = ['tdt4265_focal_loss_res34','tdt4265_focal_loss_res50','tdt4265_focal_loss_res101']
#options = ['tdt4265','tdt4265_augmented','tdt4265_fpn_res34','tdt4265_focal_loss_res34', 'tdt4265_deeper_regression_heads_res34','tdt4265_init_weights_res34']
#options = ['tdt4265_init_weights_res34','tdt4265_init_weights_res50', 'tdt4265_init_weights_res101']
df['run'] = df["run"].apply(lambda x: x.replace("\\logs\\tensorboard", ""))

df_runs = df.loc[df['run'].isin(options)]
#df_map = df[df['tag']=='metrics/mAP']
df_map = df_runs[df_runs['tag']=='metrics/mAP']

def PlotmAPTag(df, options, tag):
    df_runs = df.loc[df['run'].isin(options)]
    df_map = df_runs[df_runs['tag']==tag]
    title = tag.replace('metrics/', '')
    sns.lineplot(data=df_map, x="step", y="value", hue='run').set_title(title.replace('_', ' '))
    save_as_eps = './dataset_exploration/' + options[0] + title + '.eps'
    save_as_png = './dataset_exploration/' + options[0] + title + '.png'
    plt.savefig(save_as_eps, bbox_inches="tight", dpi=200)
    plt.savefig(save_as_png, bbox_inches="tight", dpi=200)
    plt.show()

def maxmAPvalToLatex(df):
    df_metric = df[df['tag'].str.contains('metrics')].drop('step',axis=1)
    df_loss = df[df['tag'].str.contains('loss')].drop('step',axis=1)
    
    grouped_df_metrics = df_metric.groupby('tag')
    grouped_df_loss = df_loss.groupby('tag')
    print(grouped_df_loss.min().to_latex())
    print('------------------------------------------')
    print(grouped_df_metrics.max().to_latex())
    #grouped_df_metrics = grouped_df[grouped_df['tag'].contains('metrics')]
    #maximums_metrics = grouped_df_metrics.max()
    #min_loss = grouped_df_loss.min()
    #print(min_loss.to_latex(index=False))
    
    #print(maximums_metrics.to_latex(index=False))
    
maxmAPvalToLatex(df_runs)
#options = ['tdt4265']
#PlotmAPTag(df, options, 'metrics/AP_person')
#df_loss = df[df['tag']=='loss/']
#

# df_loss = df_runs.loc[df_runs.tag.str.contains('loss')]


# df_map_runs = df_map.loc[df_map['run'].isin(options)]
# # df_loss_runs = df_loss.loc[df_loss['run'].isin(options)]

# sns.relplot(data=df_loss, x="step", y="value", hue='run', col='tag', kind='line')
# plt.ylim(0,5)
# #sns.lineplot(data=df_map, x="step", y="value", hue='run').set_title("mAP@0.5:0.95")
# figure = plt.gcf()

# figure.set_size_inches(14, 14)
# plt.savefig('./dataset_exploration/all2_3_new_loss.png',bbox_inches="tight", dpi=200)
# plt.savefig('./dataset_exploration/all2_3_new_loss.eps',bbox_inches="tight", dpi=200)
# plt.show()