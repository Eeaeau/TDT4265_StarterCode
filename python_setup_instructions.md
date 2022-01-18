# Python Setup Instructions (Local)

## Setup

**Installing Anaconda:** We recommend using [Anaconda Python distribution](https://www.anaconda.com/download/) which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version.

**Anaconda Virtual environment:** Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run (in a terminal) 

```bash
conda create -n tdt4265 python=3.6 anaconda
```

to create a environment called `tdt4265`. 

Then, to activate and enter the environment, run

```bash
source activate tdt4265
```

To exit, you can simply close the window, or run

```bash
source deactivate tdt419tdt42655
```

Note that every time you want to work on the assignment, you should run `source activate tdt4265` (change to the name of your virtual env).

You may refer to [this page](https://conda.io/docs/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.

## Installing requirements for tdt4265
We use several python packages in this course. To install requirements you can use either pip or conda. First, activate and enter your environment with 

```bash
source activate tdt4265
```



Then install pytorch and torchvision by the following command

**MACOS:** 
```bash
conda install pytorch torchvision -c pytorch
```

**Linux or Windows:**
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

You can also follow the tutorial on [the pytorch website](https://pytorch.org/get-started/locally/).

**Note**, if you have a PC with NVIDIA GPU (Linux or windows), you need to install CUDA and CUDNN first if you want to utilize your GPU. Installing CUDA and CUDNN is outside of the scope for this tutorial.

Install tqdm.

```bash
conda install tqdm
```

Finally, install [scikit-image](https://scikit-image.org/docs/stable/install.html):
```bash
conda install -c conda-forge scikit-image
```



## Launching jupyter notebook

Once you have finished the environment setup, and installed the required packages, you can launch jupyter notebook with the command:

```bash
jupyter notebook
```

Then, if you open a jupyter notebook file (`.ipynb`), you will see the active environment in the top right corner. To change the kernel to the right environment select `kernel` -> `change kernel` -> `Python tdt4265`. 

If your environment is not showing up in the list of kernels, you can take a quick look on [this stackoverflow post](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook).
