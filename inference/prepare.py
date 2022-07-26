# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
"""


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "path_to_test_x": "../../../data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "../../../data/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": "../../../data",
        "filename": "wtbdata_245days.csv",
        "target": "Patv",
        "checkpoints": "checkpoints",
        "input_len": 144,
        "output_len": 288,
        "start_col": 3,
        "day_len": 144,
        "num_workers": 4,
        "patience": 3,
        "capacity": 134,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        "is_debug": True
    }

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
