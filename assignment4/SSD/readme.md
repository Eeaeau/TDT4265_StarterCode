# SSD300


## Tutorials
- [Introduction to code](notebooks/code_introduction.ipynb).
- [Dataset setup](tutorials/dataset_setup.md) (Not required for TDT4265 computers).
- [Running tensorboard to visualize graphs](tutorials/tensorboard.md).


## Install
Follow the installation instructions from previous assignments.
Then, install specific packages with

```
pip install -r requirements.txt
```


## Dataset exploration 
We have provided some boilerplate code for getting you started with dataset exploration. It can be found in `dataset_exploration/analyze_stuff.py`. We recommend making multiple copies of this file for different parts of your data exploration.

To run the script, do the following command from the SSD folder:

```
python -m dataset_exploration.analyze_stuff
```

## Dataset visualization

We have also created a script visualizing images with annotations. To run the script, do 

```
python -m dataset_exploration.save_images_with_annotations
```

By default, the script will print the 500 first train images in the dataset, but it is possible to change this by changing the parameters in the `main` function in the script.

## Qualitative performance assessment

To check how the model is performing on real images, check out the `performance assessment` folder. Run the test script by doing:

```
python -m performance_assessment.save_comparison_images <config_file>
```

If you for example want to use the config file `configs/tdt4265.py`, the command becomes:

```
python -m performance_assessment.save_comparison_images configs/tdt4265.py
```

This script comes with several extra flags. If you for example want to check the output on the 500 first train images, you can run:

```
python -m performance_assessment.save_comparison_images configs/tdt4265.py --train -n 1000
```

### Test on video:
You can run your code on video with the following script:
```
python -m performance_assessment.demo_video configs/tdt4265.py input_path output_path
```
Example:
```
python3 -m performance_assessment.demo_video configs/tdt4265.py Video00010_combined.avi output.avi
```
You can download the validation videos from [OneDrive](https://studntnu-my.sharepoint.com/:f:/g/personal/haakohu_ntnu_no/EhTbLF7OIrZHuUAc2FWAxYoBpFJxfuMoLVxyo519fcSTlw?e=ujXUU7).
These are the videos that are used in the current TDT4265 validation dataset.



## Bencharking the data loader
The file `benchmark_data_loading.py` will automatically load your training dataset and benchmark how fast it is.
At the end, it will print out the number of images per second.

```
python benchmark_data_loading.py configs/tdt4265.py
```

## Uploading results to the leaderboard:
Run the file:
```
python save_validation_results.py configs/tdt4265.py results.json
```
Remember to change the configuration file to the correct config.
The script will save a .json file to the second argument (results.json in this case), which you can upload to the leaderboard server.
