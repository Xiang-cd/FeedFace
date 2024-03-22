import ml_collections
"""
1. using zero linear
2. less block
3. adding block at outblock position
"""

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.project = "feed"
    config.wandb_run_name = "t2iadp_face0"
    config.wandb_mode = "offline"

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64  # reduce dimension
    config.data_type = 1
    config.gradient_accumulation_steps = 1
    config.log_interval = 5000
    config.eval_interval = 100000
    config.save_interval = 400000
    config.max_step = 5000000
    
    config.train_nnet = False
    config.train_feed = True
    config.train_adp = False
    config.detect_resolution = 224
    config.feed_model = d(
        name = "t2iadp1_no_encode",
        block_names=[
        "in_blocks.0",
        "in_blocks.2",
        "in_blocks.4",
        "in_blocks.6",
        "in_blocks.8",
        "in_blocks.10",
        "in_blocks.12",
        "out_blocks.0",
        "out_blocks.2",
        "out_blocks.4",
        "out_blocks.6",
        "out_blocks.8",
        "out_blocks.10",
        "out_blocks.12",
        ],
        type_ = 2,
        block_args=d(
            num_block=1,
            name="SDBT",
            block_para=d(
                dim=1536,
                n_heads=64,
                d_head=24,
                zero_linear=True
            )
        ),
        multiplier = 1.,
        verbose=True
    )
    
    config.loss = "t2iadp1_no_encode_addition"
    config.mask_p = 0.9 # if p < mask_p, then use mask loss
    config.cfg_p = 0.1 # if p < cfg_p, then train uncond
    
    

    config.dataset = d(
        name="PeopleDataset",
        jsonl_files=[
            "configs/FFHQ_in_the_wild-llava_1_5_13b.jsonl"
        ],
        repeat=10,
        detect_resolution=config.get_ref('detect_resolution')
    )

    # eval set
    config.eval_list = [
    ]
    
    config.num_workers = 20
    config.batch_size = 16
    config.resolution = 512
    
    config.clip_img_model = "ViT-B/32"
    config.clip_text_model = "openai/clip-vit-large-patch14"
    
    config.only_load_model = True
    

    config.optimizer = d(
        name='adamw',
        lr=2e-5,
        weight_decay=0.03,
        betas=(0.9, 0.9),
        amsgrad=False
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=100
    )

    config.autoencoder = d(
        pretrained_path='models/autoencoder_kl.pth',
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.get_ref('text_dim'),
        tokenizer_path = "models/gpt2"
    )

    config.nnet = d(
        name='UViTCondToken',
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        text_dim=config.get_ref('text_dim'),
        num_text_tokens=77,
        clip_img_dim=config.get_ref('clip_img_dim'),
        use_checkpoint=True
    )


    # sample
    config.mode = "t2i"
    config.n_samples = 3
    config.n_iter = 1
    config.nrow = 4
    config.sample = d(
        sample_steps=30,
        scale=7.,
        t2i_cfg_mode='true_uncond'
    )

    return config
