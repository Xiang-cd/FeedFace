import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Union
from transformers import CLIPProcessor
import torchvision.transforms.functional as F
import random



class PeopleDataMaker(object):
    """
    只适用于/mnt/vepfs/xiang-cd/Data/face_finalLabel以及类似数据形式的数据的数据集制作
    路径和prompt位于`/mnt/vepfs/xiang-cd/Data/face_finalLabel/people.json`
    group信息则在文件夹级别体现
    
    当前使用的数据来源：
    ```
    PeopleDataMaker("/mnt/vepfs/xiang-cd/Data/face_finalLabel/people.jsonl","/mnt/vepfs/xiang-cd/Data/face_finalLabel/cond_data_test").make()
    ```
    """
    
    def __init__(self, data_jsonl, output_dir):
        self.data_jsonl = data_jsonl
        self.output_dir = output_dir
        self.group_jsons_index = 0
        self.group_json_dirs = {}
        
        assert os.path.exists(self.data_jsonl)
        assert os.path.exists(self.output_dir) and os.path.isdir(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, 'group_jsons'), exist_ok=True)
        self.group_json_base = os.path.join(self.output_dir, 'group_jsons')
    
    
    def make(self):
        """
        生成数据集
        """
        with open(self.data_jsonl) as f, open(os.path.join(self.output_dir, "data.jsonl"), 'w', encoding="utf8") as f_out:
            for line in f:
                data = json.loads(line)
                rt = self.make_single(data)
                if rt is None:
                    continue
                f_out.write(json.dumps(rt) + '\n')
    
    def make_single(self, data:Dict):
        path = Path(data['path'])
        caption = data['caption']
        group_dir = path.parent
        output_path = self.dir_to_group_json(group_dir)
        
        class_list = ["woman", "people", "man", "boy", "girl", "person"]
        class_word = None
        for cls in class_list:
            if cls in caption:
                class_word = cls
                break
        if class_word is None:
            return None
        return {
            "image1": str(path),
            "caption": caption,
            "group_json": output_path,
            "mask": "",
            "class_word": class_word,
            "type": "object"
        }
    
    def dir_to_group_json(self, dir_path):
        """
        给定包括同一物体多个照片的文件夹路径, 制作group_json, save到output_dir
        注意元素样例: {"path": "image_path", "mask": ""}
        """
        
        if dir_path in self.group_json_dirs.keys():
            return self.group_json_dirs[dir_path]
        
        output_path = os.path.join(self.group_json_base, f"{self.group_jsons_index:03}.json")
        self.group_jsons_index += 1
        self.group_json_dirs[dir_path] = output_path
        
        image_ls = os.listdir(dir_path)
        image_ls = [{"path":os.path.join(dir_path, img), "mask":""} for img in image_ls]
        json.dump(image_ls, open(output_path, 'w', encoding='utf8'),ensure_ascii=False)
        return output_path

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
 
        
class PeopleDataset(Dataset):
    """
    input_jsonl: {
        'path': full image path,
        'caption': image caption,
        'width': W,
        'height': H,
        'bbox': face bounding box,
        'mask_path': face parsing mask path
    }
    
    face_parsing_mask_pixel_map = {
        'background': 0,
        'face': 23,
        'rb': 46,
        'lb': 69,
        're': 92,
        'le': 115,
        'nose': 139,
        'ulip': 162,
        'imouth': 185,
        'llip': 208,
        'hair': 231,
    }

    只包含image1, caption, 
    返回的数据形式为:
        {
            "image1": torch.Tensor, # resize for diffusion
            "img4clip": torch.Tensor,
            "caption": str,
            "detect": torch.Tensor, # detected image resize to 244 * 244
            "detect_mask": torch.Tensor, # detected mask resize to 244 * 244
            "mask": torch.Tensor, # same size as image1, with mask on it
            "type": str
            "data_type": int, for unidiffuser
        }
    """
    def __init__(self,
                 jsonl_files,
                 repeat=1,
                 train_resolution=512,
                 detect_resolution=224,
                 p_exclude_hair=0,
                 p_flip_reference=0.5,
                 detect_transform=None):
        super().__init__()
        self.metadatas = self.init_metadatas(jsonl_files)
        self.repeat = repeat
        self.transform_clip = _transform(224)
        resolution = train_resolution
        mask_resolution = train_resolution // 8
        self.p_exclude_hair = p_exclude_hair
        self.p_flip_reference = p_flip_reference
        self.transform = transforms.Compose([transforms.Resize(resolution),
                                             transforms.CenterCrop(resolution),
                                             transforms.ToTensor(),
                                             transforms.Normalize(0.5, 0.5)])
        self.transform_mask = transforms.Compose([transforms.Resize(mask_resolution),
                                                  transforms.CenterCrop(mask_resolution),
                                                  transforms.ToTensor()])
        
        self.detect_transform = detect_transform if detect_transform is not None \
                                                 else transforms.Compose([transforms.Resize(detect_resolution),
                                                                          transforms.CenterCrop(detect_resolution),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(0.5, 0.5)])
        self.detect_mask_transform = transforms.Compose([transforms.Resize(detect_resolution),
                                                         transforms.CenterCrop(detect_resolution),
                                                         transforms.ToTensor()])

    def init_metadatas(self, jsonl_files):
        metadatas = []
        for jsonl_file in jsonl_files:
            with open(jsonl_file) as f:
                for line in f:
                    data = json.loads(line)
                    metadatas.append(data)
        return metadatas


    def get_single(self, metadata):
        caption = metadata['caption']
        image1_pil = Image.open(metadata['path']).convert('RGB') ## full img
        image1 = self.transform(image1_pil)
        
        mask_pil = Image.open(metadata['mask_path']).convert("L")
        mask_arr = np.array(mask_pil)
        if random.random() < self.p_exclude_hair:
            mask_arr[mask_arr == 231] = 0
        mask_arr[mask_arr > 0] = 255
        mask_pil = Image.fromarray(mask_arr)
        mask = self.transform_mask(mask_pil)

        detect_pil = image1_pil.crop(metadata['square_bbox'])
        detect_mask_pil = mask_pil.crop(metadata['square_bbox'])
        if random.random() < self.p_flip_reference:
            detect_mask_pil =  F.hflip(detect_mask_pil)
            detect_pil = F.hflip(detect_pil)
        detect = self.detect_transform(detect_pil)
        if not isinstance(self.detect_transform, transforms.Compose):
            detect = torch.tensor(detect["pixel_values"][0])
        detect_mask = self.detect_mask_transform(detect_mask_pil)
        

        return dict(image1=image1,
                    caption=caption,
                    img4clip=self.transform_clip(image1_pil),
                    mask=mask,
                    detect=detect,
                    detect_mask=detect_mask,
                    type="object",
                    data_type=0)

    def __len__(self):
        return len(self.metadatas) * self.repeat

    def __getitem__(self, idx):
        return self.get_single(self.metadatas[idx % len(self.metadatas)])