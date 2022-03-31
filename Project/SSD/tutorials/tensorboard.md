# Inspecting training logs:

### Via Tensorboard
To start logging on tensorboard, do:
```
tensorboard --logdir outputs
```
**NOTE:** If you want to view the tensorboard logs on the cluster you have three options:
1. Check the logs out in the jupyter lab notebook on the server, by clicking this:
![](https://raw.githubusercontent.com/chaoleili/jupyterlab_tensorboard/master/image/launcher.png)

2. Start a tensorboard server with the same port that you use for jupyter lab. This requires that jupyter lab is not running, as you cannot have access to both. To start tensorboard to a specific port, write `tensorboard --logdir outputs --port=18475`. Note that you have to replace the port 18475 with the port number that is printed when you get an allocated resource (in this case, 18475 matched the port as used in the [tutorial](https://github.com/TDT4265-tutorial/TDT4265_StarterCode/blob/main/tutorials/cluster_tutorial.md#tips-for-working-remotely)).
3. Use [tensorboard in VS Code](https://stackoverflow.com/questions/63938552/how-to-run-tensorboard-in-vscode)


### Plotting via jupyter notebook
Also, we included a notebook to export tensorboard logs to jupyter. See: ![](notebooks/plot_scalars.ipynb)