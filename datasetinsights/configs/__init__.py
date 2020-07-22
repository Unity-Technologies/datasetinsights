"""System configurations
"""
from yacs.config import CfgNode as CN

import datasetinsights.constants as const

system = CN()

system.logdir = const.NULL_STRING
system.metricsdir = const.METRICS_DIR
system.metricsfilename = const.METRICS_FILE_NAME
system.data_root = const.DEFAULT_DATA_ROOT
system.verbose = False
system.dryrun = False
system.no_cuda = False
system.workers = 0
system.val_interval = 1

# USim authorization token
system.auth_token = const.NULL_STRING
