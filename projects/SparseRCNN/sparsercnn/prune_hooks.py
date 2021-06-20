import datetime
import itertools
import logging
import os
import tempfile
import time
from collections import Counter
import torch
from fvcore.common.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from fvcore.common.param_scheduler import ParamScheduler
from fvcore.common.timer import Timer
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.solver import LRMultiplier
from detectron2.utils.events import EventStorage, EventWriter
from detectron2.utils.file_io import PathManager

from detectron2.engine.train_loop import HookBase
from detectron2.engine.defaults import DefaultTrainer
import torch.nn.utils.prune as prune

class UnStructureIMP(HookBase):
    def __init__(self,prune_steps,prune_gamma,verbose=False):
        super(UnStructureIMP, self).__init__()
        self.prune_steps = prune_steps
        self.prune_gamma = prune_gamma
        self.param_remain = 1.0
        self.verbose = verbose

    def after_step(self):
        if self.verbose:
            print('iter:', self.trainer.iter)
            print('steps:',self.prune_steps)

        if self.trainer.iter in self.prune_steps:
            model = self.trainer._trainer.model
            if isinstance(model,DistributedDataParallel):
                backbone = model.module.backbone
            else:
                backbone = model.backbone
            # TODO:switch to global pruning
            for name, module in backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    print('pruned a conv')
                    prune.l1_unstructured(module, name='weight', amount=self.prune_gamma)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=self.prune_gamma)
                    print('pruned a linear')
            self.param_remain *= self.prune_gamma
            if self.verbose:
                print('enter')
                logger = logging.getLogger(__name__)

                logger.info(
                    "Parameter remain in model: %.2f %%" % self.param_remain * 100
                )


