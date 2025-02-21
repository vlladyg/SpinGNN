# reference example
from nequip.data import dataset_from_config
from nequip.utils import Config
from nequip.train.trainer import Trainer
from e3nn import o3
import sys
from nequip.utils import finish_all_writes, atomic_write_group, finish_all_writes
from time import perf_counter
#from nequip.utils.misc import get_default_device_name
#from nequip.utils.config import _GLOBAL_ALL_ASKED_FOR_KEYS

from nequip.model import model_from_config
import os

default_config = dict(
    root="./",
    tensorboard=False,
    wandb=False,
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "StressForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    device='cuda:0',
    default_dtype="float64",
    model_dtype="float32",
    allow_tf32=True,
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    gpu_oom_offload=False,
    append=False,
    warn_unused=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy=[("DYNAMIC", 3)],
    # Due to what appear to be ongoing bugs with nvFuser, we default to NNC (fuser1) for now:
    # TODO: still default to NNC on CPU regardless even if change this for GPU
    # TODO: default for ROCm?
    _jit_fuser="fuser1",
)

os.environ['NEQUIP_NUM_TASKS'] = '1'

config = Config.from_file('./config/example_ETN_opt_MEA.yaml', defaults=default_config)


def run_ind(ind):
    
    config['root'] = f'results/MEA_Allegro_{ind}'
    config['seed'] = 1234560 + ind
   
    dataset = dataset_from_config(config, prefix="dataset")
    validation_dataset = None    

    # Trainer
    trainer = Trainer(model=None, **Config.as_dict(config))
    
    # what is this
    # to update wandb data?
    config.update(trainer.params)
    
    # = Train/test split =
    trainer.set_dataset(dataset, validation_dataset)
    
    
    # = Build model =
    final_model = model_from_config(
        config=config, initialize=True, dataset=trainer.dataset_train)


    trainer.model = final_model

    #trainer.train()

    # Init the trainer
    init_trainer(trainer)
    
    # Check if epoch per sweep has a correct value
    assert (config['max_epochs'] % config['epochs_per_sweep'] == 0)
    
    
    num_sweeps = config['max_epochs'] // config['epochs_per_sweep']
    
    cur_sweep = 0
    while cur_sweep < num_sweeps:
        cur_sweep += run_one_sweep_cycle(trainer, config)
    
    for callback in trainer._final_callbacks:
        callback(trainer)

    trainer.final_log()

    trainer.save()
    finish_all_writes()
    

def init_trainer(trainer):
    """Init the trainer"""
    
    if not trainer._initialized:
        trainer.init()

    for callback in trainer._init_callbacks:
        callback(trainer)

    trainer.init_log()
    trainer.wall = perf_counter()
    trainer.previous_cumulative_wall = trainer.cumulative_wall
    
    with atomic_write_group():
        if trainer.iepoch == -1:
            trainer.save()
        if trainer.iepoch in [-1, 0]:
            trainer.save_config()
    
    trainer.init_metrics()

def run_one_sweep_cycle(trainer, config):
    
    cur_sweep = 0 
    # Making all non trainable
    for i in range(config['d']):
        trainer.model.get_submodule('model.model.func.etn.cores')[i].requires_grad_(False)
    
    # Forward sweeps
    for i in range(config['d']):
        trainer.model.get_submodule('model.model.func.etn.cores')[i].requires_grad_(True)
        
        if i != 0:
            trainer.model.get_submodule('model.model.func.etn.cores')[i-1].requires_grad_(False)

        for j in range(config['epochs_per_sweep']):
            trainer.epoch_step()
            trainer.end_of_epoch_save()
        
        cur_sweep += 1
    
    # Backward sweeps   
    for i in range(config['d']-2, -1, -1):
        trainer.model.get_submodule('model.model.func.etn.cores')[i].requires_grad_(True)
        trainer.model.get_submodule('model.model.func.etn.cores')[i+1].requires_grad_(False)
        
        for j in range(config['epochs_per_sweep']):
            trainer.epoch_step()
            trainer.end_of_epoch_save()

        cur_sweep += 1
        
    return cur_sweep 
    
    
if __name__ == """__main__""":
    num_model = int(sys.argv[1])
    
    for ind in range(num_model):
        run_ind(ind)
