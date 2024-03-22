import json
import torch
from accelerate import Accelerator
import facer
import numpy as np  
from PIL import Image, ImageOps
import cv2
import os
import tqdm
from pathlib import Path
import glob

# Usage:
# CUDA_VISIBLE_DEVICES=3,1,2,4,5,6,7 accelerate launch make_data.py 

def calculate_iou(box1, box2):  
    # box1, box2 are lists/tuples of (x1, y1, x2, y2)  
    x1_tl, y1_tl, x1_br, y1_br = box1  
    x2_tl, y2_tl, x2_br, y2_br = box2  
  
    # calculate intersection box  
    x1 = max(x1_tl, x2_tl)  
    y1 = max(y1_tl, y2_tl)  
    x2 = min(x1_br, x2_br)  
    y2 = min(y1_br, y2_br)  
  
    # check if boxes intersect  
    if x2 <= x1 or y2 <= y1:  
        return 0.0  
  
    # calculate intersection area  
    intersection_area = (x2 - x1) * (y2 - y1)  
  
    # calculate union area  
    union_area = (x1_br - x1_tl) * (y1_br - y1_tl) + (x2_br - x2_tl) * (y2_br - y2_tl) - intersection_area  
  
    # calculate IoU  
    iou = intersection_area / union_area  
    return iou

def read_hwc(path: str, resize_s=1) -> torch.Tensor:
    """Read an image from a given path.

    Args:
        path (str): The given path.
    """
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    
    if resize_s != 1:
        width, height = image.size
        new_size = int(width*resize_s), int(height*resize_s)
        image = image.resize(new_size)
    
    np_image = np.array(image.convert('RGB'))
    return torch.from_numpy(np_image)

class FacerModel():
    def __init__(self, device, out_folder, mask_exclude_hair=False) -> None:
        self.device = device
        self.mask_exclude_hair = mask_exclude_hair
        self.out_folder = out_folder
        
        self.iou_thresh = 0.7
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        self.face_parser = facer.face_parser('farl/lapa/448', device=device)
        
    def seg_logits_2_seg_masks(self, faces):
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        n_classes = seg_probs.size(1)
        vis_seg_probs = seg_probs.argmax(dim=1).float()
        ## Remove Hair
        if self.mask_exclude_hair:
            vis_seg_probs[vis_seg_probs==10.0] = 0
        
        vis_seg_probs = vis_seg_probs/n_classes*255
        seg_masks = vis_seg_probs.cpu().numpy().astype(np.uint8)
        return seg_masks
        
    def seg_mask_get_bbox(self, seg_mask):
        ## cal bounding box for face segmentation
        contours, hierarchy = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        if len(contours) == 0:
            return [0,0,0,0]
        x, y, w, h = cv2.boundingRect(contours[0])  
        x1, y1, x2, y2 = x, y, x+w, y+h
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        return bbox
    
    def process_and_remove_duplicate(self, faces):
        ## for each faces, get segmentation mask
        ## for each segmentation mask, get bbox
        ## remove overlap bbox, iou > 90
        seg_masks = self.seg_logits_2_seg_masks(faces)
        masks_bbox = [self.seg_mask_get_bbox(s) for s in seg_masks]
        
        keep_index = []
        for i in range(len(masks_bbox)):
            ## compare to all keep index
            keep = True
            if masks_bbox[i][2] == 0 and masks_bbox[i][3] == 0:
                keep = False
            else:
                for j in keep_index:
                    iou = calculate_iou(masks_bbox[i], masks_bbox[j])
                    if iou > self.iou_thresh: 
                        keep = False
                        break
            if keep:
                keep_index.append(i)            
                
        ## take only keep_index
        masks_bbox = [masks_bbox[i] for i in keep_index]
        seg_masks = seg_masks[keep_index]
        for k in faces.keys():
            if k == "seg":
                faces['seg']['logits'] = faces['seg']['logits'][keep_index]
            else:
                faces[k] = faces[k][keep_index]
            
        return faces, seg_masks, masks_bbox
        
        
    def infer(self, img_path, resize_s=1):
        filename = Path(img_path).stem

        image = facer.hwc2bchw(read_hwc(img_path, resize_s=resize_s)).to(self.device)
        h,w = image.size()[-2:]
        with torch.inference_mode():
            faces = self.face_detector(image)
            if len(faces) == 0: ## not detect any face, skip
                ## try again with half scale
                resize_s = resize_s / 2
                image = facer.hwc2bchw(read_hwc(img_path, resize_s=resize_s)).to(self.device)
                
                faces = self.face_detector(image)
                
            if len(faces) == 0:
                ## length will be 4 if detected some face , 0 otherwise
                raise Exception("No Face detected")
            
            faces = self.face_parser(image, faces)
            
            faces, seg_masks, masks_bbox = self.process_and_remove_duplicate(faces)
            
            ## separate num face > 1
            face_count = len(faces['rects'])
            for i in range(face_count):
                if face_count == 1:
                    # folder = f"{self.out_folder}/single/{subfolder}"
                    folder = f"{self.out_folder}/single"
                    os.makedirs(folder, exist_ok=True)
                    mask_path = f"{folder}/{filename}.png"
                    meta_path = f"{folder}/{filename}.json"
                else:
                    # folder = f"{self.out_folder}/multi/{subfolder}"
                    folder = f"{self.out_folder}/multi"
                    os.makedirs(folder, exist_ok=True)
                    mask_path = f"{folder}/{filename}_{i}.png"
                    meta_path = f"{folder}/{filename}_{i}.json"
                
                ## save seg mask
                seg_data = seg_masks[i]
                seg_image = Image.fromarray(seg_data)
                img_size = seg_image.size
                if resize_s != 1:
                    seg_image = seg_image.resize((w,h), Image.NEAREST)
                seg_image.save(mask_path)
                bbox = masks_bbox[i]

                ## read prediction 
                rect = faces['rects'][i].cpu().numpy().tolist()
                point = faces['points'][i].cpu().numpy().tolist()
                scores = faces['scores'][i].cpu().numpy().tolist()
                
                ## save result
                out_dict = {
                    'img_size': img_size,
                    'rect': rect,
                    'bbox': bbox,
                    'point': point,
                    'scores': scores,
                    'resize_s': resize_s 
                }
                json.dump(out_dict, open(meta_path, 'w+'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", type=str, default="/mnt/vepfs/xcd-share/dataset/FFHQ_raw_1024/faces")
    parser.add_argument("--data_folder", type=str, default="/mnt/vepfs/xcd-share/dataset/FFHQ_raw_1024/eval_data")
    args = parser.parse_args()
    ## setup output folder
    out_folder = args.out_folder
    error_folder = f"{out_folder}/error"
    os.makedirs(f"{out_folder}/single", exist_ok=True)
    os.makedirs(f"{out_folder}/multi", exist_ok=True)
    os.makedirs(error_folder, exist_ok=True)
    os.makedirs(error_folder, exist_ok=True)
    
    data = glob.glob(f'{args.data_folder}/*.png') + glob.glob(f'{args.data_folder}/*.jpg')
    print(len(data))
    
    ## multi gpu setup           
    accelerator = Accelerator()
    
    ## model setup
    fm = FacerModel(accelerator.device, out_folder)
    
    # multi gpu infer
    with accelerator.split_between_processes(data,) as batch:
        for d in tqdm.tqdm(batch, disable=accelerator.process_index!=0):
            img_path = d
            
            try:
                fm.infer(img_path)
            except Exception as e:
                if str(e) != "No Face detected":
                    print("Error: ", img_path, e)
                    with open(f"{error_folder}/{accelerator.process_index}_other.log", 'a+') as f:
                        f.write(f"{e}: {img_path}\n")
                else:
                    with open(f"{error_folder}/{accelerator.process_index}.log", 'a+') as f:
                        f.write(f"{e}: {img_path}\n")
