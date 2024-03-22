import torch
import utils
from absl import logging
import os
import wandb
import libs.autoencoder
import clip
from libs.clip import FrozenCLIPEmbedder
from libs.caption_decoder import CaptionDecoder
from torch.utils.data import DataLoader
from libs.diffusion_schedule import stable_diffusion_beta_schedule, Schedule
import json
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP




def train(config):
    
    """
    prepare models
    准备各类需要的模型
    """
    accelerator, device = utils.setup(config)
    
    # frozen caption decoder, auto encoder and text encoder
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)
    clip_img_model, clip_img_model_preprocess = clip.load(config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)
    
    train_state = utils.initialize_train_state(config, device)
    
    loss_fn = utils.get_loss_fn(name=config.loss)

    """
    处理数据部分
    """
    # process data
    train_dataset = utils.get_dataset(**config.dataset)
    train_dataset_loader = DataLoader(train_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      pin_memory=True,
                                      drop_last=True
                                      )
    
    nnet, optimizer, feed_model, train_dataset_loader, lr_scheduler = accelerator.prepare(train_state.nnet, 
                                                                                          train_state.optimizer,
                                                                                          train_state.feed_model,
                                                                                          train_dataset_loader,
                                                                                          train_state.lr_scheduler)
    train_data_generator = utils.get_data_generator(train_dataset_loader, enable_tqdm=accelerator.is_main_process, desc='train')
    

    if accelerator.is_main_process:
        logging.info("saving meta data")
        os.makedirs(config.meta_dir, exist_ok=True)
        with open(os.path.join(config.meta_dir, "config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=4)
    
    _betas = stable_diffusion_beta_schedule()
    schedule = Schedule(_betas)
    logging.info(f'use {schedule}')

    total_step = 0
    global_step = 0
    def train_step():
        nonlocal global_step, total_step
        nnet.train()
        feed_model.train()
        iter_dict = next(train_data_generator)
        loss, loss_img, loss_clip_img = loss_fn(
            feed_model=feed_model,
            nnet=nnet,
            clip_text_model=clip_text_model,
            caption_decoder=caption_decoder,
            autoencoder=autoencoder,
            clip_img_model=clip_img_model,
            schedule=schedule,
            device=device,
            iter_dict=iter_dict,
            config=config)


        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        global_step += 1
        optimizer.zero_grad()
        
        metrics = {}
        metrics['loss'] = accelerator.gather(loss.detach().mean()).mean().item()
        metrics['loss_img'] = accelerator.gather(loss_img.detach().mean()).mean().item()
        metrics['loss_clip_img'] = accelerator.gather(loss_clip_img.detach().mean()).mean().item()
        metrics['scale'] = accelerator.scaler.get_scale()
        metrics['lr'] = optimizer.param_groups[0]['lr']

        return metrics


    def loop():
        nonlocal global_step, total_step
        log_step = 0
        eval_step = 0
        save_step = config.save_interval
        while True:
            nnet.eval()
            metrics = train_step()
            
            total_step = global_step * config.total_batch_size
            if total_step >= eval_step:
                eval_step += config.eval_interval
                accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                nnet.eval()
                feed_model.eval()
                if total_step >= log_step:
                    logging.info(utils.dct2str(dict(step=total_step, **metrics)))
                    wandb.log(utils.add_prefix(metrics, 'train'), step=total_step)
                    log_step += config.log_interval


                if total_step >= save_step:
                    logging.info(f'Save and eval checkpoint {total_step}...')
                    eval_feed_model = feed_model.module if isinstance(feed_model, DDP) else feed_model
                    eval_nnet = nnet.module if isinstance(nnet, DDP) else nnet
                    if config.train_feed or config.train_adp:
                        torch.save(eval_feed_model.state_dict(), os.path.join(config.ckpt_root, f'{total_step:06}.pt'))
                    if config.train_nnet:
                        torch.save(eval_nnet.state_dict(), os.path.join(config.ckpt_root, f'{total_step:06}_nnet.pt'))
                    save_step += config.save_interval

            accelerator.wait_for_everyone()
            
            if total_step  >= config.max_step:
                break

    loop()






def save_source_files(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    file_list = [
        f"{__file__}",
        "utils.py"
    ]

    for i in os.listdir("libs"):
        if i.endswith(".py"):
            file_list.append(os.path.join("libs", i))
    import shutil
    for f in file_list:
        shutil.copyfile(f, os.path.join(target_dir, os.path.split(f)[-1]))


from absl import flags
from absl import app
from ml_collections import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("workdir", "workdir", "Work unit directory.")
flags.DEFINE_string("resume_ckpt_path", None, "The path containing the train state to resume.")
flags.DEFINE_string("logdir", "logs", "base log dir")
flags.DEFINE_string("wandb_run_prefix", None, "prefix of wandb run")
flags.DEFINE_string("wandb_mode", "offline", "offline / online")
flags.DEFINE_string("nnet_path", "models/uvit_v1.pth", "pretrain resume of unidiffuser")
flags.mark_flags_as_required(["config"])


def main(argv):
    config = FLAGS.config
    config.log_dir = FLAGS.logdir
    config.config_name = utils.get_config_name()
    config.data_name = config.dataset.name
    config.hparams = utils.get_hparams()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M") # avoid process dir differs from different process, end with minute but not second
    folder_name = f"{config.config_name}-{config.data_name}-{config.hparams}-{now}"
    config.workdir = os.path.join(config.log_dir, folder_name)
    config.hparams = utils.get_hparams()
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.meta_dir = os.path.join(config.workdir, "meta")
    config.resume_ckpt_path = FLAGS.resume_ckpt_path
    config.nnet_path = FLAGS.nnet_path
    os.makedirs(config.workdir, exist_ok=True)
    save_source_files(config.meta_dir)
    
    # wandb name and mode
    if FLAGS.wandb_run_prefix is not None:
        config.wandb_run_name = f"{FLAGS.wandb_run_prefix}-{config.wandb_run_name}"
    else:
        config.wandb_run_name = folder_name
    config.wandb_mode = FLAGS.wandb_mode
    
    train(config)


if __name__ == "__main__":
    app.run(main)
