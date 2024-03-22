import torch
import torch.nn as nn
import os
from absl import logging
import sys
from pathlib import Path
import json
from tqdm import tqdm
import yaml
from ml_collections import ConfigDict


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem

def get_data_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--data='):
            return Path(argv[i].split('=')[-1]).stem
        

def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit_multi_post_ln_v1':
        from libs.unidiffuser_v1 import UViT
        return UViT(**kwargs)
    elif name == "UViTCondToken":
        from libs.unidiffuser_v1 import UViTCondToken
        return UViTCondToken(**kwargs)
    else:
        raise NotImplementedError(name)

def get_dataset(name, **kwargs):
    if name == "PeopleDataset":
        from libs.extra_cond_data import PeopleDataset
        return PeopleDataset(**kwargs)
    else:
        raise NotImplementedError(name)

def get_feed_model(name, **kwargs):
    if name in ["t2iadp1", "t2iadp1_no_encode"]:
        from libs.uvit_v1_t2iadapter import T2IAdapter
        return T2IAdapter(**kwargs)
    else:
        raise NotImplementedError(name)

def get_loss_fn(name, **kwargs):
    if name in ["t2iadp1_no_encode_addition"]:
        from libs.uvit_v1_t2iadapter import T2i_compute_mask_loss_no_encode_addition_unidiffuser
        return T2i_compute_mask_loss_no_encode_addition_unidiffuser
    else:
        raise NotImplementedError(name)

def get_sample_fn(name, **kwargs):
    if name in ["t2iadp1_no_encode"]:
        from libs.feed_sample import t2iadp_no_encode_sample
        return t2iadp_no_encode_sample
    else:
        raise NotImplementedError(name)

def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)

def load_yaml_config(path):
    with open(path) as f:
        config = ConfigDict(initial_dictionary=yaml.load(f, Loader=yaml.loader.UnsafeLoader))
    return config

def load_json_config(path):
    with open(path) as f:
        config = ConfigDict(initial_dictionary=json.load(f))
    return config

def load_config(path):
    if path.endswith(".yaml") or path.endswith(".yml"):
        return load_yaml_config(path)
    elif path.endswith(".json"):
        return load_json_config(path)
    else:
        raise ValueError("Unknown config file type: {}".format(path))

def load_small_modules(device, base="models"):
    from libs.clip import FrozenCLIPEmbedder
    from libs.caption_decoder import CaptionDecoder
    import libs.autoencoder
    import clip

    caption_decoder = CaptionDecoder(
        device=device,
        pretrained_path=f"{base}/caption_decoder.pth",
        hidden_dim=64,
        tokenizer_path = f"{base}/gpt2"
    )
    autoencoder = libs.autoencoder.get_model(pretrained_path=f'{base}/autoencoder_kl.pth',).to(device)
    clip_text_model = FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14", device=device)
    clip_img_model, clip_img_model_preprocess = clip.load("ViT-B/32", jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)
    return autoencoder, caption_decoder, clip_text_model, clip_img_model


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None, feed_model=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema
        self.feed_model = feed_model

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def resume(self, ckpt_path=None, only_load_model=False):
        if ckpt_path is None:
            return

        logging.info(f'resume from {ckpt_path}, only_load_model={only_load_model}')
        self.step = torch.load(os.path.join(ckpt_path, 'step.pth'))

        if only_load_model:
            for key, val in self.__dict__.items():
                if key == 'nnet_ema' or key == 'nnet':
                    val.load_state_dict(torch.load(os.path.join(ckpt_path, f'{key}.pth'), map_location='cpu'))
        else:
            for key, val in self.__dict__.items():
                if key != 'step' and val is not None:
                    val.load_state_dict(torch.load(os.path.join(ckpt_path, f'{key}.pth'), map_location='cpu'))

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'), False)
    
    feed_model = get_feed_model(**config.feed_model)
    if config.feed_model.name in ["t2iadp1", "t2iadp2", "t2iadp2sep", "t2iadp1_no_encode"]:
        feed_model.apply_to(nnet)
    elif "t2i" in config.feed_model.name:
        raise
    # config trained parameters
    
    
    adapter_param = None
    if config.feed_model.name in ['mask_loss_face0_clip_perceiver']:
        for n, p in feed_model.named_parameters():
            if n.startswith('clip_model.vision_model.post_layernorm'):
                p.requires_grad = False
        adapter_param = feed_model.perceiver.parameters()
    elif config.feed_model.name in ["simple_face0", "face0_mask_loss", "face0", "face0_mask", "shallow_face0"]:
        for n, p in feed_model.named_parameters():
            if n.startswith('resnet.logits'):
                p.requires_grad = False
        adapter_param = feed_model.adapter.parameters()
    elif config.feed_model.name in ["t2iadp1", "t2iadp2", "t2iadp2sep", "t2iadp1_no_encode"]:
        pass
    else:
        raise
    
    if config.train_nnet:
        params += nnet.parameters()
    
    if config.train_feed:# by default adp module is included in feed model
        params += feed_model.parameters()
    elif config.train_adp:
        params += adapter_param

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)
    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, nnet=nnet, feed_model=feed_model)
    train_state.to(device)
    
    # for the case when the lr is manually changed
    lr_scheduler.base_lrs = [config.optimizer.lr]
    optimizer.param_groups[0]['initial_lr'] = config.optimizer.lr
    lr_scheduler._last_lr = lr_scheduler.get_lr()
    optimizer.param_groups[0]['lr'] = lr_scheduler.get_lr()[0]

    return train_state

def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.'):
            hparam_full, val = argv[i].split('=')
            hparam = hparam_full.split('.')[-1]
            if hparam_full.startswith('--config.optimizer.lm'):
                hparam = f'lm_{hparam}'
            if hparam_full.startswith('--config.optimizer.decoder'):
                hparam = f'decoder_{hparam}'
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams

def add_prefix(dct, prefix):
    return {f'{prefix}/{key}': val for key, val in dct.items()}

def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def setup(config):
    import builtins
    import ml_collections
    from torch import multiprocessing as mp
    import accelerate
    import wandb

    mp.set_start_method('spawn')
    assert config.gradient_accumulation_steps == 1, \
        'fix the lr_scheduler bug before using larger gradient_accumulation_steps'
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config.num_processes = accelerator.num_processes

    assert not ('total_batch_size' in config and 'batch_size' in config)
    if 'total_batch_size' not in config:
        config.total_batch_size = config.batch_size * accelerator.num_processes
    if 'batch_size' not in config:
        assert config.total_batch_size % accelerator.num_processes == 0
        config.batch_size = config.total_batch_size // accelerator.num_processes
    
    config = ml_collections.FrozenConfigDict(config)

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=config.project, config=config.to_dict(),
                   name=config.wandb_run_name, job_type='train', mode=config.wandb_mode)
        set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    return accelerator, device



def get_data_generator(loader, enable_tqdm, desc):
    while True:
        for data in tqdm(loader, disable=not enable_tqdm, desc=desc):
            yield data
