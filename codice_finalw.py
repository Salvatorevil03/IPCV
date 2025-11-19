import torch
import torch.utils.data
import cv2
import os
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms.functional as F  
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm
import math
import copy
import numpy as np # linear algebra
from torchvision.utils import draw_bounding_boxes
import albumentations as A  # our data augmentation library
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes

warnings.filterwarnings("ignore")


"""
Classe Dataset
"""
class MyReducedCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file) # Carica le annotazioni
        all_ids = list(sorted(self.coco.imgs.keys())) ##mappo gli id in una lista ordinata
        self.ids=[]

        for img_id in tqdm(all_ids):
            path = self.coco.loadImgs(img_id)[0]['file_name']
            full_path = os.path.join(self.root, path)
            
            if os.path.exists(full_path):
                self.ids.append(img_id)

    def __getitem__(self, index):
        # 1. Carica ID e path immagine
        coco = self.coco
        img_id = self.ids[index] #mi prendo id effettivo immagine
        ann_ids = coco.getAnnIds(imgIds=img_id) #mi prendo le annotations
        coco_annotation = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        
        img = Image.open(os.path.join(self.root, path)).convert("RGB") ##pytorch vuole solo img in rgb

        # 3. Estrai le Box e le Label
        #nbisogna convertire i bounding box
        num_objs = len(coco_annotation)
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2] # COCO è x,y,w,h -> convertiamo in x,y,x,y
            ymax = ymin + coco_annotation[i]['bbox'][3]
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
            areas.append(coco_annotation[i]['area'])
            iscrowd.append(coco_annotation[i]['iscrowd'])

        # Conversione in Tensori
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Gestione casi senza box (immagini vuote)
        if num_objs == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
    
        return img, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    #((img1,target1),(img2,target2)) ---- ((img1,img2),(target1,target2))
    return tuple(zip(*batch))

"""
Creiamo le istanze della classe dataset
e le diamo in input alla classe dataload
"""
IMG_DIR_TRAIN = '/kaggle/input/coco-reduced/kaggle/working/train'
ANN_FILE_TRAIN = '/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_train2017.json'

IMG_DIR_VAL = '/kaggle/input/coco-reduced/kaggle/working/val'
ANN_FILE_VAL = '/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_val2017.json'

dataset_train = MyReducedCocoDataset(root=IMG_DIR_TRAIN, annotation_file=ANN_FILE_TRAIN)
data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=4, 
        shuffle=True,
        num_workers=2, 
        collate_fn=collate_fn
)

dataset_val = MyReducedCocoDataset(root=IMG_DIR_VAL, annotation_file=ANN_FILE_VAL)
data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_fn
    )

"""
Importiamo il modello Faster RCNN ci torchvision
"""
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.NONE)

"""
Attivare la GPU
"""
device = torch.device("cuda")
model = model.to(device)

"""
Prendiamo la lista di tutti i parametri del modello e creiamo l'ottimizzatore

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)


Definiamo la funzionare per effetturare il train su tot epoche specificate
DA CAMBIARE

def train(model, optimizer, loader, device, num_epoch):
    model.to(device)
    model.train()
    
    for epoch in num_epoch:
        all_losses = []
        all_losses_dict = []
        
        for images, targets in tqdm(loader):
            images = list(image.to(device) for image in images)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
            loss_value = losses.item()
            
            all_losses.append(loss_value)
            all_losses_dict.append(loss_dict_append)
            
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
                print(loss_dict)
                sys.exit(1)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            
        all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
        print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
            all_losses_dict['loss_classifier'].mean(),
            all_losses_dict['loss_box_reg'].mean(),
            all_losses_dict['loss_rpn_box_reg'].mean(),
            all_losses_dict['loss_objectness'].mean()
        ))


Test del train

train(model, optimizer, data_loader_train, device, 2)
"""
