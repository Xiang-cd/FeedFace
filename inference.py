import utils
import torch
from insightface.app import FaceAnalysis
from utils import load_small_modules
from utils import load_config
import utils
from absl import logging
import torch
import os
import json

def prepare_context(args):
    """
    prepare context for later use
    """
    device = "cuda:0"

    autoencoder, caption_decoder, clip_text_model, clip_img_model = load_small_modules(device)

    ## load nnet and feed
    config_path = args.config
    feed_resume_path = args.ckpt_path


    config = load_config(config_path)
    nnet = utils.get_nnet(**config.nnet)
    print(nnet.load_state_dict(torch.load(config.nnet_path, map_location="cpu"), False))
    feed_model = utils.get_feed_model(**config.feed_model)
    print(type(feed_model))
    print(feed_model.load_state_dict(torch.load(feed_resume_path)))
    feed_model.eval()
    feed_model.apply_to(nnet)

    nnet.eval()
    nnet.to(device)
    feed_model.to(device)

    face_app = FaceAnalysis("buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(512, 512))
    sfn = "cat"
    print("sample_fn:", sfn)
    return {
        "device": device,
        "config": config,
        "caption_decoder": caption_decoder,
        "nnet": nnet,
        "feed_model": feed_model,
        "face_app": face_app,
        "autoencoder": autoencoder,
        "clip_text_model": clip_text_model,
        "clip_img_model": clip_img_model,
        "sample_fn": sfn,
    }


def get_face(face_app, image_path):
    import cv2

    img = cv2.imread(image_path)
    faces = face_app.get(img, max_num=1)

    bbox = faces[0].bbox.astype(int)
    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
    # get square bbox
    center_x, center_y = x + w / 2, y + h / 2
    image_size = int(max(h, w) * 1.1)
    x = int(center_x - image_size / 2)
    y = int(center_y - image_size / 2)
    bbox = [x, y, x + image_size, y + image_size]
    detected_image = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    return detected_image


def process_one_json(json_data, context={}):
    """
    given a json object, process the task the json describes
    """


    # 初始化训练步数

    config = context["config"]
    device = context["device"]
    caption_decoder = context["caption_decoder"]
    nnet = context["nnet"]
    autoencoder = context["autoencoder"]
    clip_text_model = context["clip_text_model"]
    face_app = context["face_app"]
    feed_model = context["feed_model"]
    assert context["sample_fn"] in ["first", "cat", "mean"]

    from libs.feed_sample import (
        t2iadp_no_encode_sample,
        t2iadp_no_encode_sample_feature_mean,
        t2iadp_no_encode_sample_feature_cat,
    )

    if context["sample_fn"] == "first":
        sample_fn = t2iadp_no_encode_sample
    elif context["sample_fn"] == "cat":
        sample_fn = t2iadp_no_encode_sample_feature_cat
    elif context["sample_fn"] == "mean":
        sample_fn = t2iadp_no_encode_sample_feature_mean

    ref_paths = [i["path"] for i in json_data["source_group"]]
    detect_imgs = [get_face(face_app, ref_path) for ref_path in ref_paths]
    detect_imgs = [img for img in detect_imgs if img is not None]
    if len(detect_imgs) == 0:
        logging.info("No face detected, skip")
        return {"id": json_data["id"], "images": []}

    images = []
    for caption in json_data["caption_list"]:
        config.prompt = caption
        config.sample.sample_steps = 40
        samples = sample_fn(
            caption,
            detect_path_ls=detect_imgs,
            config=config,
            nnet=nnet,
            clip_text_model=clip_text_model,
            feed_model=feed_model,
            autoencoder=autoencoder,
            caption_decoder=caption_decoder,
            device=device,
            cond_multiplier=0.8,
            image2=None,
        )["samples"]
        images.append({"prompt": caption, "samples": samples})

    return {"id": json_data["id"], "images": images}


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--ckpt_path", type=str, default="models/feed-4800000.pt")
    parser.add_argument("-o", "--output", type=str, default="outputs")
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("-j", "--json", type=str, default=None, help="run a batch of sample task using a json file")
    parser.add_argument("-i", "--image", type=str, default=None, help="run a single sample task using a image file")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="prompt for the single sample task")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    context = prepare_context(args)
    context["config"].n_samples = args.n_samples
    context["config"].n_iter = args.n_iter
    if args.json is not None:
        one_data = json.load(open(args.json))
    elif args.image is not None and args.prompt is not None:
        one_data = {
            "id": "test",
            "caption_list": [args.prompt],
            "source_group": [{"path": args.image}],
        }
    else:
        raise ValueError("Please provide json or image and prompt")
    

    return_data = process_one_json(one_data, context=context)

    os.makedirs(args.output, exist_ok=True)
    for item in return_data["images"]:
        for idx, img in enumerate(item["samples"]):
            img.save(f"{args.output}/{return_data['id']}_{item['prompt']}_{idx}.jpg")
