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

experiment_id = "SaVgxERkT9KxtGAqhbp09g"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
print(type(experiment))
df = experiment.get_scalars()

options = ['tdt4265_fpn_res34','tdt4265_fpn_res50', 'tdt4265_fpn_v2_101']
df['run'] = df["run"].apply(lambda x: x.replace("\\logs\\tensorboard", ""))


df_map = df[df['tag']=='metrics/mAP']


df_map_runs = df_map.loc[df_map['run'].isin(options)]



sns.lineplot(data=df_map_runs, x="step", y="value", hue='run').set_title("mAP")

plt.show()