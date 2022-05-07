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

experiment_id = "hzh50lR3Skafw6l5IEzY7A"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
print(type(experiment))
df = experiment.get_scalars()
#options = ['tdt4265']
#options = ['tdt4265_augmented']
options = ['tdt4265','tdt4265_augmented']
options = ['tdt4265_updated_res34','tdt4265_updated_res101','tdt4265_deeper_regression_heads_res34']
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
#options = ['tdt4265']
PlotmAPTag(df, options, 'metrics/AP_person')
#df_loss = df[df['tag']=='loss/']
#

#df_loss = df_runs.loc[df_runs.tag.str.contains('loss')]


#df_map_runs = df_map.loc[df_map['run'].isin(options)]
#df_loss_runs = df_loss.loc[df_loss['run'].isin(options)]

#sns.relplot(data=df_loss, x="step", y="value", hue='run', col='tag', kind='line')
#plt.ylim(0,10)
#sns.lineplot(data=df_map, x="step", y="value", hue='run').set_title("mAP@0.5:0.95")
#figure = plt.gcf()

#figure.set_size_inches(12, 8)
#plt.savefig('./dataset_exploration/tdt4265andaugmented_map.png',bbox_inches="tight", dpi=200)
#plt.savefig('./dataset_exploration/tdt4265andaugmented_map.eps',bbox_inches="tight", dpi=200)
#plt.show()