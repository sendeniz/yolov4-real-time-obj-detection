import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.nn.modules.upsampling import Upsample
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.yolov4 import YoloV4_EfficentNet
from loss.yolov4loss import YoloV4Loss
from utils.dataset import CoCoDataset
from utils.utils import (
    mean_average_precision,
    get_bouding_boxes,
    class_accuracy,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# params
img_size = 608 #416
conf_tresh = 0.8 #0.4 #0.6
map_iou_thresh = 0.5  #0.5 #0.5 
nms_iou_thresh = 0.65 #0.45 #0.65
nclasses = 80
lr = 3e-4
nepochs = 300
weight_decay = 0.0
pin_memory = True
nworkers = 4
batch_size = 2
momentum = 0.949

anchors = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

S = [13, 26, 52]

def train_yolov4(train_loader, model, optimizer, loss_fn, scaled_anchors):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.permute(0, 3, 1, 2)
        x = x.to(torch.float32).to(device)
        y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))
        # x shape :-: (batchsize, channels, height, width)
        preds = model(x)
        loss_val = (loss_fn(preds[0], y0, scaled_anchors[0]) + loss_fn(preds[1], y1, scaled_anchors[1]) + loss_fn(preds[2], y2, scaled_anchors[2]))
        optimizer.zero_grad()
        class_acc, noobj_acc, obj_acc = class_accuracy(preds, y, conf_tresh)
        loss_val.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    return (float(loss_val.item()), class_acc, noobj_acc, obj_acc)

def test_yolov4(test_loader, model, loss_fn, scaled_anchors):
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.permute(0, 3, 1, 2)
            x = x.to(torch.float32).to(device)
            y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))
            preds = model(x)
            loss_val = (loss_fn(preds[0], y0, scaled_anchors[0]) + loss_fn(preds[1], y1, scaled_anchors[1]) + loss_fn(preds[2], y2, scaled_anchors[2]))
            class_acc, noobj_acc, obj_acc = class_accuracy(preds, y, conf_tresh)
            
    return (float(loss_val.item()), class_acc, noobj_acc, obj_acc)

def main():
    model = YoloV4_EfficentNet(nclasses=nclasses).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.949, 0.999))
    loss_fn = YoloV4Loss()

    train_dataset = CoCoDataset("data/coco/train_2examples.csv", "data/coco/images/", "data/coco/labels/",
                          S=[13, 26, 52], anchors=anchors, transform=True)
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, num_workers = nworkers,
        pin_memory= True, shuffle = True, drop_last=False)

    #test_dataset = CoCoDataset("data/coco/test_2examples.csv", "data/coco/images/", "data/coco/labels/",
    #                      S=[13, 26, 52], anchors=anchors, transform=None)
    
    #test_loader = DataLoader(dataset = test_dataset,batch_size = batch_size, num_workers = nworkers,
    #    pin_memory= True, shuffle = True, drop_last=False)
    
       
    scaled_anchors = (torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(device)
    #print("Scaled Anchor Shape:", scaled_anchors.shape)
    for epoch in range(nepochs ):
        train_loss_val, train_class_acc, train_noobj_acc, train_obj_acc = train_yolov4(train_loader, model, optimizer, loss_fn, scaled_anchors)
        train_pred_boxes, train_true_boxes = get_bouding_boxes(train_loader, model, iou_threshold = nms_iou_thresh, anchors=anchors,
                                                   threshold=conf_tresh)  
        #test_loss_val, test_class_acc, test_noobj_acc, test_obj_acc = test_yolov4(test_loader, model, loss_fn, scaled_anchors)
        
        #test_pred_boxes, test_true_boxes = get_bouding_boxes(test_loader, model, iou_threshold = nms_iou_thresh, anchors=anchors,
        #                                           threshold=conf_tresh)
        print(f"Train number bounding boxes:  predictions:{len(train_pred_boxes)}  true:{len(train_true_boxes)}")
        #print(f"Test number bounding boxes:  predictions:{len(test_pred_boxes)}  true:{len(test_true_boxes)}")

        train_map_val = mean_average_precision(train_pred_boxes, train_true_boxes, map_iou_thresh, boxformat = "midpoints", nclasses=nclasses)
        #test_map_val = mean_average_precision(test_pred_boxes, test_true_boxes, map_iou_thresh, boxformat = "midpoints", nclasses=nclasses)

        print(f"Epoch:{epoch}  Train[Loss:{train_loss_val} mAP:{train_map_val} class acc:{train_class_acc}, noobj acc:{train_noobj_acc}   obj acc:{train_obj_acc}]")
        #print(f"Epoch:{epoch}  Test[Loss:{test_loss_val} mAP:{test_map_val} class acc:{test_class_acc}, noobj acc:{test_noobj_acc}   obj acc:{test_obj_acc}]")
  


"""
if __name__ == "__main__":
    main()
"""