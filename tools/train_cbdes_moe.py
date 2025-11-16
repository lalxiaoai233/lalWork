#!/usr/bin/env python3
"""
Training script for CBDES MoE BEVFusion model.

This script extends the original training pipeline to support
CBDES MoE specific loss functions and expert utilization monitoring.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

# Ensure CBDES-specific pipeline components are registered
try:
    from mmdet3d.datasets.pipelines import cbdes_safe  # noqa: F401
except Exception:
    cbdes_safe = None

# Version information (optional)
# try:
#     from mmdet import __version__ as mmdet_version
# except ImportError:
#     mmdet_version = None

# try:
#     from mmdet3d import __version__ as mmdet3d_version  
# except ImportError:
#     mmdet3d_version = None


def main():
    dist.init()

    parser = argparse.ArgumentParser(description='Train CBDES MoE BEVFusion model')
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='resume training from a checkpoint file',
    )
    parser.add_argument(
        '--routing-loss-weight',
        type=float,
        default=0.01,
        help='weight for routing loss (load balancing)',
    )
    parser.add_argument(
        '--expert-utilization-interval',
        type=int,
        default=100,
        help='interval to log expert utilization statistics',
    )
    
    args, opts = parser.parse_known_args()

    # Load config using torchpack (same as original train.py)
    configs.load(args.config, recursive=True)
    
    # Update configs with opts if provided (same as standard train.py)
    configs.update(opts)
    
    # Handle resume_from if provided via command line argument
    if args.resume_from is not None:
        configs.resume_from = args.resume_from

    # Use recursive_eval to evaluate all config expressions
    cfg = Config(recursive_eval(configs), filename=args.config)

    # Override routing loss weight if specified
    if hasattr(args, 'routing_loss_weight'):
        cfg.setdefault('loss_weights', {})
        cfg.loss_weights['routing_loss'] = args.routing_loss_weight

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    
    # Create a tee-like logger that writes to both file and console
    class TeeLogger:
        def __init__(self, log_file):
            self.log_file = open(log_file, 'a')
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            
        def write(self, message):
            self.log_file.write(message)
            self.log_file.flush()
            self.original_stdout.write(message)
            self.original_stdout.flush()
            
        def flush(self):
            self.log_file.flush()
            self.original_stdout.flush()
            
        def close(self):
            if self.log_file:
                self.log_file.close()
    
    # Redirect stdout/stderr to both file and console
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger
    sys.stderr = tee_logger
    
    try:
        logger = get_root_logger(log_file=log_file)
    except:
        logger = get_root_logger()

    # Add warmup learning rate config if not present (参考20251109的配置)
    if cfg.get('lr_config') is None or cfg.lr_config is None:
        logger.info("Adding warmup learning rate config (参考20251109配置)")
        cfg.lr_config = dict(
            policy='CosineAnnealing',
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=0.33333333,
            min_lr_ratio=1.0e-3
        )

    # log some basic info
    try:
        logger.info(f"Config:\n{cfg.pretty_text}")
    except Exception as e:
        # Fallback if pretty_text fails (e.g., yapf formatting error)
        logger.warning(f"Failed to format config with pretty_text: {e}")
        try:
            # Try using text property instead
            logger.info(f"Config:\n{cfg.text}")
        except Exception:
            # Final fallback: read from dumped config file
            config_file = os.path.join(cfg.run_dir, "configs.yaml")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    logger.info(f"Config:\n{f.read()}")
            else:
                logger.info("Config loaded successfully (unable to format for display)")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    cfg.data.samples_per_gpu = 1
    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Add CBDES MoE specific hooks if applicable
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'encoders'):
        if cfg.model.encoders.get('camera') and cfg.model.encoders['camera'].get('backbone', {}).get('type') == 'CBDESMoE':
            logger.info("CBDES MoE model detected, adding expert utilization monitoring")
            
            from mmcv.runner import HOOKS
            from mmcv.runner.hooks import Hook
            
            @HOOKS.register_module()
            class ExpertUtilizationHook(Hook):
                def __init__(self, interval=100):
                    self.interval = interval
                    
                def after_train_iter(self, runner):
                    if runner.iter % self.interval == 0:
                        if hasattr(runner.model, 'get_expert_utilization'):
                            utilization = runner.model.get_expert_utilization()
                            if utilization:
                                runner.logger.info(f"Expert Utilization: {utilization}")
            
            # Add the hook to training config
            if 'custom_hooks' not in cfg:
                cfg.custom_hooks = []
            cfg.custom_hooks.append({
                'type': 'ExpertUtilizationHook',
                'interval': getattr(args, 'expert_utilization_interval', 100)
            })

    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )
    
    # Close log file
    if 'tee_logger' in locals():
        tee_logger.close()


if __name__ == "__main__":
    main()
