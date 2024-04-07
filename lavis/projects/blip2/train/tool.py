import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.registry import registry
from lavis.common.utils import now
# from lavis.compression import load_pruner
# imports modules for registration
from lavis.runners import *
import lavis
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)



print(lavis)

print(registry.list_lr_schedulers())
print(registry.list_datasets())

print(registry.list_processors())
print(registry.list_pruners())
