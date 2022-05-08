from .tdt4265_init_weights import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)


"""
Test 1:
"""
#strides = [[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [90, 90]]



"""
test 2:
"""
anchors.strides = [[4, 4], [8, 8], [16, 16], [32, 32], [48,48] ,[64, 64]]


#Test1
anchors.aspect_ratios = [ [0.2, 1.4, 2.5], [0.2, 1.4, 2.5], [0.2, 1.4 , 2.5], [0.2, 1.4, 2.5], [1, 1.5, 2 ], [1, 1.5, 2] ]
#anchors.min_sizes= [[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]]

#test2
anchors.min_sizes= [[8, 8], [16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128]]


#anchors.aspect_ratios = [ [0.3,1,1.5] ,  [0.25,0.9,1.5] ,  [0.25,0.9,1.5] ,  [0.25,0.9,1.5] ,  [0.25,0.9,1.5] ,  [0.25,0.9,1.5]  ]

#test3
#anchors.aspect_ratios = [ [1,3,4] , [1,3,4], [1,3,4], [1,3,4], [1,3,4], [1,3,4] ]

#test4
#anchors.aspect_ratios = [ [0.3,0.7,1,1.5] ,  [0.3,0.7,1,1.5] ,  [0.3,0.7,1,1.5] ,  [0.3,0.7,1,1.5],  [0.3,0.7,1,1.5] ,  [0.3,0.7,1,1.5]  ]

"""
test6
"""
#anchors.strides= [[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [90, 90]]
#anchors.aspect_ratios = [ [2,3,4] , [2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4] ]


""""
Test7
"""
#anchors.strides= [[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [90, 90]]

#Width 1024
#height 128
#width car: 17
# height car: 10
#car min size: [17,10]
#aspect ratio car: 17/5=3.4
#width pedestrian: 5
#height pedestrian: 10
#pedestrian min size: [5,10]
#aspect ratio pedestrian: 5/5=1


#anchors.min_sizes = [ [17, 10], [17, 10], [5, 10], [5, 10], [10, 5], [17, 17] ]

#test5
#anchors.min_sizes = [[5, 10],[10,20],[17,10],[17*2,10*2], [10,5],[128, 128], [128, 400] ]

#anchors.aspect_ratios = [ [3.4, 3.4, 1], [3.4, 3.4, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [3.4, 3.4, 1] ]
