# Walkthrough of TDT4265 Annotation system

## Annotation Tutorial
**We will publish a thorough video describing the annotation process soon.**
Watch the tutorial before starting the labeling task.

## Annotation Steps:

1. Open the link to the annotation web server, and log in with your provided username and password. https://tdt4265-annotering.idi.ntnu.no/

2. Choose a task and click open.

3. Assign the task to yourself by selecting your group number from the dropdown menu.
**When you set yourself as the assignee, you're not able to remove yourself. Therefore, you have to finish the annotation task before starting a new one!**

4. Click the Job number to open the task.

5. Remember to save while labeling to avoid losing progress.

6. Once labeling is completed, go to "Info" menu and set job status to validation. Then student assistants will review your labeling and you will be awarded points if the task is completed and labeled correctly.


## Used labels
These are the labels that you are supposed to be annotating: 

* Rider (a person riding a bicycle/motorcycle)
* Person
* Scooter
* Bicycle
* Motorcycle
* Bus
* Truck
* Car (Includes cars, vans, large family cars for 6-8 people etc.)

## Annotation Guidelines

**What to label:**
**ALL objects** of the defined categories, **unless**:
* You are unsure what the object is.
* The object is very small (at your discretion). A general rule of thumb is that whatever a human eye can see, computer must be able to recognize. So if you can make out an object and which class it belongs to in the frame, you must label it.
* Less than 10-20% of the object is visible, such that you cannot be sure what class it is. e.g. if only a tyre is visible it may belong to car or truck so cannot be labelled car, but feet/faces can only belong to a person
* All objects that are visible in **any of the LIDAR channels** (that is, objects that are visible in either intensity, range or ambience). The CVAT server only visualizes the LIDAR intensity, but you can take a look at the other channels by looking at the original videos posted [here](https://studntnu-my.sharepoint.com/:f:/g/personal/haakohu_ntnu_no/EnNwXrHCFbRPn9fYNWXaw7MBUvAD4pz1kVs0HpJWe9PfTA?e=qAITlw). 

**Bounding Box:**
* Bounding box should be tight around the object, except where the bounding box would have to be made excessively large to include a few additional pixels (<5%) e.g. a car aerial.
* Mark the bounding box of the visible area of the object (not the estimated total extent of the object).

## Labeling Tips:
* Follow [Piazza post](https://piazza.com/class/kyipdksfp9q1dn?cid=226) for updated labeling tips and FAQs.
* Use **Track mode** for labeling. It interpolates the bounding box between frames so you can save some time.
* Automatic annotations have been generated using detectron2 but remember to check whether they are correct or not. You will have to either remove, adjust or draw the bounding boxes yourself so **check carefully**.

## CVAT Shortcuts:
* N : Start/Finish drawing bounding box
* Space : Play/Pause video
* D: Go back one frame
* F: Go forward one frame
* C: Go back 10 frames
* V: Go forward 10 frames 
* Ctrl+S : Save progress
* Mouse Wheel: Zoom in/out


## Other resources:
* OPENCV CVAT detailed user guide: [https://openvinotoolkit.github.io/cvat/docs/manual/basics/interface/](https://openvinotoolkit.github.io/cvat/docs/manual/basics/interface/)
* VOC2011 Annotation Guidelines [http://host.robots.ox.ac.uk/pascal/VOC/voc2011/guidelines.html](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/guidelines.html)
