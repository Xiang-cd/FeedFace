import torch
import random
from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import einops
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(prompt, config, clip_text_model, device):
    img_contexts = torch.randn(config.n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])
    clip_imgs = torch.randn(config.n_samples, 1, config.clip_img_dim)

    prompts = [ prompt ] * config.n_samples
    contexts = clip_text_model.encode(prompts)

    return contexts, img_contexts, clip_imgs


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def set_seed(seed: int):
    print("setting seed to {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split(x, config):
    C, H, W = config.z_shape
    z_dim = C * H * W
    z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
    z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
    return z, clip_img


def combine(z, clip_img):
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
    return torch.concat([z, clip_img], dim=-1)

@torch.cuda.amp.autocast()
def decode(_batch, autoencoder):
    return autoencoder.decode(_batch)


def sample_fn(text, config, device, _n_samples, _betas, t2i_nnet):
    _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
    _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
    _x_init = combine(_z_init, _clip_img_init)
    N = len(_betas)

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

    def model_fn(x, t_continuous):
        t = t_continuous * N
        return t2i_nnet(x, t, text)

    dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
    with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
        x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)

    _z, _clip_img = split(x, config)
    return _z, _clip_img





def t2iadp_no_encode_sample(
    prompt,
    detect_path_ls,
    config, nnet, clip_text_model, feed_model, autoencoder, caption_decoder, 
    device, return_map=False,
    detect_mask_path=None,
    cond_multiplier=1.,
    uncond_multiplier=0.,
    cfg = 5.,
    mask_path=None, **kwargs):
    assert config.feed_model.name in ["t2iadp1_no_encode"]
    print(f"t2iadp sampling ...cfg={cfg}, cm:{cond_multiplier} ucm:{uncond_multiplier}")
    
    set_seed(config.seed)
    res = config.detect_resolution
    detect_transform = transforms.Compose([transforms.Resize(res),
                                            transforms.CenterCrop(res),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])
    detect_path = detect_path_ls[0]
    if type(detect_path) is str:
        print(f"sample with cond:{detect_path}")
        detect = Image.open(detect_path).convert("RGB")
        detect = detect_transform(detect).unsqueeze(0).to(device)
    elif type(detect_path) is np.ndarray:
        detect = detect_transform(Image.fromarray(detect_path).convert("RGB")).unsqueeze(0).to(device)
    
        
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    empty_context = clip_text_model.encode([''])[0]
    contexts, img_contexts, clip_imgs = prepare_contexts(prompt, config, clip_text_model, device)
    contexts_low_dim = caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet
    _n_samples = contexts_low_dim.size(0)

    feed_model(detect.repeat(_n_samples, 1, 1, 1))
    
    attention_maps = []

    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        z, clip_img = split(x, config)
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        feed_model.set_multiplier(cond_multiplier)
        dict_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                        data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type, return_map=return_map)
        feed_model.set_multiplier(uncond_multiplier)
        z_out, clip_img_out, text_out = dict_out["img_out"], dict_out["clip_img_out"], dict_out["text_out"]
        if return_map:
            attention_maps.append([i.detach().cpu() for i in dict_out["attention_maps"]])
        x_out = combine(z_out, clip_img_out)

        if config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            _empty_context = caption_decoder.encode_prefix(_empty_context)
            dict_out = nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            z_out_uncond, clip_img_out_uncond, text_out_uncond =  dict_out["img_out"], dict_out["clip_img_out"], dict_out["text_out"]
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif config.sample.t2i_cfg_mode == 'true_uncond':
            text_N = torch.randn_like(text)  # 3 other possible choices
            dict_out = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            z_out_uncond, clip_img_out_uncond, text_out_uncond =  dict_out["img_out"], dict_out["clip_img_out"], dict_out["text_out"]
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + cfg * (x_out - x_out_uncond)


    samples = []  
    for i in range(config.n_iter):
        _z, _clip_img = sample_fn(text=contexts_low_dim,
                            config=config,
                            device=device,
                            _n_samples=_n_samples,
                            _betas=_betas,
                            t2i_nnet=t2i_nnet)  # conditioned on the text embedding
        new_samples = unpreprocess(decode(_z, autoencoder))
        for sample in new_samples:
            samples.append(transforms.ToPILImage()(sample))
    return {"samples": samples,
            "attention_maps": attention_maps
            }


@torch.no_grad()
def t2iadp_no_encode_sample_feature_mean(
    prompt, image2:str,
    detect_path_ls,
    config, nnet, clip_text_model, feed_model, autoencoder, caption_decoder, 
    device, return_map=False,
    detect_mask_path=None,
    cond_multiplier=1.,
    uncond_multiplier=0.,
    cfg = 5.,
    mask_path=None, **kwargs):
    """
    condition on a batch of ref images, first use preprocess module to extract feature, then use torch.mean to get
    final feature token
    """
    assert config.feed_model.name in ["t2iadp1_no_encode"]
    
    print(f"t2iadp sampling ...cfg={cfg}, cm:{cond_multiplier} ucm:{uncond_multiplier}")
    set_seed(config.seed)
    res = config.detect_resolution
    detect_transform = transforms.Compose([transforms.Resize(res),
                                            transforms.CenterCrop(res),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])
    detect_ls = []
    for detect_path in detect_path_ls:
        if type(detect_path) is str:
            print(f"sample with cond:{detect_path}")
            detect = Image.open(detect_path).convert("RGB")
            detect = detect_transform(detect).unsqueeze(0).to(device)
        elif type(detect_path) is np.ndarray:
            detect = detect_transform(Image.fromarray(detect_path).convert("RGB")).unsqueeze(0).to(device)
        detect_ls.append(detect)
    detect_ls = torch.vstack(detect_ls)
    
    extracted = torch.mean(feed_model.preprocess_module(detect_ls.to(device)), dim=0, keepdim=True)
    
    # hack feed model
    def feed_forward(self, x, **kwargs):
        """
        as the feature is exctract outside the feed model, so we just set contidion
        """
        self.clear_feature()
        for b in self.blocks:
            b.preprocessed = x
    origin_feed_forward = feed_model.forward
    feed_model.forward = feed_forward.__get__(feed_model, type(feed_model))
    
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    empty_context = clip_text_model.encode([''])[0]
    contexts, img_contexts, clip_imgs = prepare_contexts(prompt, config, clip_text_model, device)
    contexts_low_dim = caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet
    _n_samples = contexts_low_dim.size(0)

    feed_model(extracted.repeat(_n_samples, 1, 1))
    
    attention_maps = []

    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        z, clip_img = split(x, config)
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        feed_model.set_multiplier(cond_multiplier)
        dict_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                        data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type, return_map=return_map)
        feed_model.set_multiplier(uncond_multiplier)
        z_out, clip_img_out, text_out = dict_out["img_out"], dict_out["clip_img_out"], dict_out["text_out"]
        if return_map:
            attention_maps.append([i.detach().cpu() for i in dict_out["attention_maps"]])
        x_out = combine(z_out, clip_img_out)

        if config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            _empty_context = caption_decoder.encode_prefix(_empty_context)
            dict_out = nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            z_out_uncond, clip_img_out_uncond, text_out_uncond =  dict_out["img_out"], dict_out["clip_img_out"], dict_out["text_out"]
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif config.sample.t2i_cfg_mode == 'true_uncond':
            text_N = torch.randn_like(text)  # 3 other possible choices
            dict_out = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            z_out_uncond, clip_img_out_uncond, text_out_uncond =  dict_out["img_out"], dict_out["clip_img_out"], dict_out["text_out"]
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + cfg * (x_out - x_out_uncond)

    samples = []  
    for i in range(config.n_iter):
        _z, _clip_img = sample_fn(text=contexts_low_dim,
                                  config=config,
                                  device=device,
                                  _n_samples=_n_samples,
                                  _betas=_betas,
                                  t2i_nnet=t2i_nnet)  # conditioned on the text embedding
        new_samples = unpreprocess(decode(_z, autoencoder))
        for sample in new_samples:
            samples.append(transforms.ToPILImage()(sample))
    
    feed_model.forward = origin_feed_forward.__get__(feed_model, type(feed_model))
    return {"samples": samples,
            "attention_maps": attention_maps
            }

@torch.no_grad()
def t2iadp_no_encode_sample_feature_cat(
    prompt, image2:str,
    detect_path_ls,
    config, nnet, clip_text_model, feed_model, autoencoder, caption_decoder, 
    device, return_map=False,
    detect_mask_path=None,
    cond_multiplier=1.,
    uncond_multiplier=0.,
    cfg = 5.,
    mask_path=None, **kwargs):
    """
    condition on a batch of ref images, first use preprocess module to extract feature, then use torch.mean to get
    final feature token
    """
    assert config.feed_model.name in ["t2iadp1_no_encode"]
    
    print(f"t2iadp cat sampling ...cfg={cfg}, cm:{cond_multiplier} ucm:{uncond_multiplier}")
    set_seed(config.seed)
    res = config.detect_resolution
    detect_transform = transforms.Compose([transforms.Resize(res),
                                            transforms.CenterCrop(res),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])
    detect_ls = []
    for detect_path in detect_path_ls:
        if type(detect_path) is str:
            print(f"sample with cond:{detect_path}")
            detect = Image.open(detect_path).convert("RGB")
            detect = detect_transform(detect).unsqueeze(0).to(device)
        elif type(detect_path) is np.ndarray:
            detect = detect_transform(Image.fromarray(detect_path).convert("RGB")).unsqueeze(0).to(device)
        detect_ls.append(detect)
    detect_ls = torch.vstack(detect_ls)
    
    extracted = feed_model.preprocess_module(detect_ls.to(device))
    print(extracted.shape)
    extracted = torch.reshape(extracted[:5], (1, -1, extracted.shape[-1])) # only take first 5 images

    
    # hack feed model
    def feed_forward(self, x, **kwargs):
        """
        as the feature is exctract outside the feed model, so we just set contidion
        """
        self.clear_feature()
        for b in self.blocks:
            b.preprocessed = x
    origin_feed_forward = feed_model.forward
    feed_model.forward = feed_forward.__get__(feed_model, type(feed_model))
    
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    empty_context = clip_text_model.encode([''])[0]
    contexts, img_contexts, clip_imgs = prepare_contexts(prompt, config, clip_text_model, device)
    contexts_low_dim = caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet
    _n_samples = contexts_low_dim.size(0)

    feed_model(extracted.repeat(_n_samples, 1, 1))
    
    attention_maps = []

    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        z, clip_img = split(x, config)
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        feed_model.set_multiplier(cond_multiplier)
        dict_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                        data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type, return_map=return_map)
        feed_model.set_multiplier(uncond_multiplier)
        z_out, clip_img_out, text_out = dict_out["img_out"], dict_out["clip_img_out"], dict_out["text_out"]
        if return_map:
            attention_maps.append([i.detach().cpu() for i in dict_out["attention_maps"]])
        x_out = combine(z_out, clip_img_out)

        if config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            _empty_context = caption_decoder.encode_prefix(_empty_context)
            dict_out = nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            z_out_uncond, clip_img_out_uncond, text_out_uncond =  dict_out["img_out"], dict_out["clip_img_out"], dict_out["text_out"]
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif config.sample.t2i_cfg_mode == 'true_uncond':
            text_N = torch.randn_like(text)  # 3 other possible choices
            dict_out = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            z_out_uncond, clip_img_out_uncond, text_out_uncond =  dict_out["img_out"], dict_out["clip_img_out"], dict_out["text_out"]
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + cfg * (x_out - x_out_uncond)

    

    samples = []  
    for i in range(config.n_iter):
        _z, _clip_img = sample_fn(text=contexts_low_dim,
                                  config=config,
                                  device=device,
                                  _n_samples=_n_samples,
                                  _betas=_betas,
                                  t2i_nnet=t2i_nnet)  # conditioned on the text embedding
        new_samples = unpreprocess(decode(_z, autoencoder))
        for sample in new_samples:
            samples.append(transforms.ToPILImage()(sample))
    
    feed_model.forward = origin_feed_forward.__get__(feed_model, type(feed_model))
    return {"samples": samples,
            "attention_maps": attention_maps
            }
    
  