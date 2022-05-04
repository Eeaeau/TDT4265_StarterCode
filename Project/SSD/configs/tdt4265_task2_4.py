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

#test1
#anchors.aspect_ratios = [ [1, 50] , [1, 50], [1, 50], [1, 50], [1, 50], [1, 50]]


#test2
#anchors.aspect_ratios = [ [1, 1] , [1, 1], [6, 1], [6, 1], [6, 1], [6, 1]]
"""
epoch 10:
2022-05-04 22:38:44,139 [INFO ] metrics/mAP: 0.045, metrics/mAP@0.5: 0.119, metrics/mAP@0.75: 0.020, metrics/mAP_small: 0.011, metrics/mAP_medium: 0.059, metrics/mAP_large: 0.134, metrics/average_recall@1: 0.018, metrics/average_recall@10: 0.085, metrics/average_recall@100: 0.102, metrics/average_recall@100_small: 0.058, metrics/average_recall@100_medium: 0.117, metrics/average_recall@100_large: 0.189, metrics/AP_car: 0.177, metrics/AP_bus: 0.000, metrics/AP_person: 0.001, metrics/AP_rider: 0.002,

"""

#test3
#anchors.aspect_ratios = [ [1],[1],[6],[6],[6],[6],[6]]
"""
epoch 10:
2022-05-04 23:01:18,211 [INFO ] metrics/mAP: 0.046, metrics/mAP@0.5: 0.121, metrics/mAP@0.75: 0.020, metrics/mAP_small: 0.011, metrics/mAP_medium: 0.057, metrics/mAP_large: 0.156, metrics/average_recall@1: 0.018, metrics/average_recall@10: 0.075, metrics/average_recall@100: 0.094, metrics/average_recall@100_small: 0.039, metrics/average_recall@100_medium: 0.118, metrics/average_recall@100_large: 0.231, metrics/AP_car: 0.176, metrics/AP_bus: 0.000, metrics/AP_person: 0.006, metrics/AP_rider: 0.000,
"""
#anchors.min_sizes =[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [400, 100], [128, 400]]

#test 4
anchors.aspect_ratios = [ [6],[6],[6],[6],[6],[6],[6]]