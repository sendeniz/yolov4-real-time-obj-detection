import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from cells.rnncells import RnnCell, GruCell, LstmCell, UrLstmCell
from models.rnn import SimpleRNN, GruRNN, LstmRNN, UrLstmRNN, HippoRNN
from models.hippo import Hippo
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import top1accuracy, top5accuracy, strip_square_brackets, cells_to_bboxes
from numpy import genfromtxt
import numpy as np
from models.yolov4 import YoloV4_EfficentNet
from loss.yolov4loss import YoloV4Loss, YoloV4Loss2
from utils.dataset import CoCoDataset
from utils.utils import mean_average_precision, get_bouding_boxes, class_accuracy
from utils.utils import linearly_increasing_lr

from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device Name: {torch.cuda.get_device_name(device)}" if device.type == 'cuda' else "CPU")
#torch.autograd.set_detect_anomaly(True)

data_dir =  'data/'

model_names = ['rnn', 
                'gru',
                'lstm',
                'urlstm',
                'hippo',
                'gatedhippo',
                'yolov4_608',
                'yolov4_416']

trains_normal = ['Simple Rnn', 
                'Gru Rnn',
                'Lstm Rnn',
                'Ur Lstm Rnn', 
                'Hippo Rnn']

trains_hippo = ['Simple Hippo']

trains_yolo = ['YoloV4 608',
              'YoloV4 416']

dataset_names = ['mnist',
                 'mscoco']

floatTypes = [
  "weight_decay", 
  "lr",
  "warmup_lr",
  "momentum",
  "min_lr",
  "conf_thresh",
  "map_iou_thresh", 
  "nms_iou_thresh", 
]

intTypes = [
  "sequence_length", 
  "input_size",
  "hidden_size",
  "output_size",
  "nclasses",
  "batch_size",
  "nepochs",
  "nruns",
  "nworkers",
  "warmup",
  "N",
  "image_size",
]

boolTypes = [
  "save_model",
  "continue_training",
  "cosine_anneal",
  "exponential_decay",
  "one_cycle",
]

def initialize_with_args(_arguments):
  arguments = {
      "model_name": "rnn",
      "dataset_name": "mnist",
      "save_model": True,
      "continue_training": False,
      "weight_decay": 0.0005,
      "sequence_length": 784,
      "input_size": 1,
      "hidden_size": 512,
      "output_size": 10,
      "nclasses": 80,
      "batch_size": 64,
      "nepochs": 50,
      "nruns": 1,
      "warmup": 0,
      "nworkers": 2,
      "N": 512,
      "image_size": 416,
      "conf_thresh": 0.6 #0.5, # ignore threshold in cfg original value 0.7
      "map_iou_thresh": 0.5, #0.213, # iou threshold in cfg
      "nms_iou_thresh": 0.5,  # beta nms in cfg original value 0.6
      "init_rnn": False,
      "init_grurnn": False,
      "init_lstmrnn": False,
      "init_urlstmrnn": False,
      "init_hippo": False,
      "init_hippornn": False,
      "init_yolov4": False,
      "mnist": False,
      "mscoco": False,
      "current_model": "",
      "lr": 1e-4, # inital learning rate
      "warmup_lr": 1e-3, # learning rate used for linear warmup: cosine_anneal and exponential decay
      "momentum": 0.9, # default val in adam pytorch
      "min_lr": 1e-5,  # for cosine anneal
      "cosine_anneal": False, # performs internal warmup so no linear warmup needed
      "exponential_decay": False,
      "one_cycle": False,
      "path_cpt_file": ""
  }
  arguments = {**arguments, **_arguments}

  for key, value in arguments.items():
    if key in floatTypes:
        arguments[key] = float(value)
    if key in intTypes:
        arguments[key] = int(value)
    if key in boolTypes:
        if value == "False":
            arguments[key] = False
        if value == "True":
            arguments[key] = True
         
  if arguments["model_name"] not in model_names:
    print(f"model name {arguments['model_name']} was not found, use simple, gru, lstm, hippo, gatedhippo or yolov4_416 or yolov4_608.")
    return
  if arguments["dataset_name"] not in dataset_names:
    print(f"dataset name {arguments['dataset_name']} was not found")
    return
  if arguments["model_name"] == "rnn":
    arguments["init_rnn"] = True
    arguments["current_model"] = model_names[0]
    arguments["path_cpt_file"] = 'cpts/{}_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Simple Rnn'
  elif arguments["model_name"] == "gru":
    arguments["init_grurnn"] = True
    arguments["current_model"] = model_names[1]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Gru Rnn'
  elif arguments["model_name"] == "lstm":
    arguments["init_lstmrnn"] = True
    arguments["current_model"] = model_names[2]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Lstm Rnn'
  elif arguments["model_name"] == "urlstm":
    arguments["init_urlstmrnn"] = True
    arguments["current_model"] = model_names[3]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Ur Lstm Rnn'
  elif arguments["model_name"] == "hippo":
    arguments["init_hippo"] = True
    arguments["current_model"] = model_names[4]
    arguments["path_cpt_file"] = 'cpts/{}_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Simple Hippo'
  elif arguments["model_name"] == "gatedhippo":
    arguments["init_hippornn"] = True
    arguments["current_model"] = model_names[5]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Hippo Rnn'
  # for object detect
  elif arguments["model_name"] == "yolov4_608":
    arguments["init_yolov4"] = True
    arguments["current_model"] = model_names[6]
    arguments["path_cpt_file"] = 'cpts/{}_mscoco.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'YoloV4 608'
    # yolo anchors rescaled between 0,1
    # yolo scales and anchors for image size 608, 608
    S = [19, 38, 76]
    anchors = [
       [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
       [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
       [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
       ]
 
  elif arguments["model_name"] == "yolov4_416":
    arguments["init_yolov4"] = True
    arguments["current_model"] = model_names[7]
    arguments["path_cpt_file"] = 'cpts/{}_mscoco.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'YoloV4 416'
    # yolo scales and anchors for image size 416, 416
    S = [13, 26, 52]
    anchors = [
       [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
       [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
       [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
       ]

  if arguments["dataset_name"] == "mnist":
      arguments["mnist"] = True
  if arguments["dataset_name"] == "mscoco":
      arguments["mscoco"] = True

  main(arguments)

def train(arguments, train_loader, model, optimizer, loss_f):
    '''
    Performs the training loop. 
    Input: train loader (torch loader)
           model (torch model)
           optimizer (torch optimizer)
           loss function (torch loss).
    Output: No output.
    '''
    model.train()
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = [] 
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # turn [64, 28, 28] to [64, 784, 1]
        x = x.view(x.shape[0], -1, arguments['input_size'])
        # hippo rnn returns a tensor of shape (batchsize, 1, output_size)
        # the extra dimension at the first index is removed with if statement below.
        out = model(x)
        if out.shape == torch.Size([arguments['batch_size'], 1, arguments['output_size']]):
            out = out.squeeze(1)
        loss_val = loss_f(out, y)
        class_prob = F.softmax(out, dim = 1)
        # preds = class_prob.argmax(dim=1)
        loss_lst.append(float(loss_val.item()))
        top1_acc_val = top1accuracy(class_prob, y)
        top5_acc_val = top5accuracy(class_prob, y)
        top1_acc_lst.append(float(top1_acc_val))
        top5_acc_lst.append(float(top5_acc_val))

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
    # compute average loss
    loss_val = round(sum(loss_lst) / len(loss_lst), 4)
    top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst),  4)
    top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
   
    return (loss_val, top1_acc, top5_acc) 

def evaluate(arguments, data_loader, model, loss_f):
    '''
    Input: test or train loader (torch loader) 
           model (torch model)
           loss function (torch loss)
    Output: loss (torch float)
            accuracy (torch float)
    '''
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1, arguments['input_size'])
            out = model(x)
            if out.shape == torch.Size([arguments['batch_size'], 1, arguments['output_size']]):
                out =  out.squeeze(1)
            loss_val = loss_f(out, y)
            class_prob = F.softmax(out, dim = 1)
            # pred = torch.argmax(class_prob, dim = 1)
            loss_lst.append(float(loss_val.item()))
            top1_acc_val = top1accuracy(class_prob, y)
            top5_acc_val = top5accuracy(class_prob, y)
            top1_acc_lst.append(float(top1_acc_val))
            top5_acc_lst.append(float(top5_acc_val))

        # compute average loss
        loss_val = round(sum(loss_lst) / len(loss_lst), 4)
        top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst),  4)
        top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
        return (loss_val, top1_acc, top5_acc)


def trainhippo(arguments, train_loader, model, optimizer, loss_f):
    """
    Input: train loader (torch loader), model (torch model), optimizer (torch optimizer)
          loss function (torch custom yolov1 loss).
    Output: loss (torch float).
    """
    model.train()
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = [] 
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # turn [64, 28, 28] to [64, 784]
        x = x.view(x.shape[0], -1).to(device)
        # turn [64, 784] to [784, 64, 1, 1]
        x = x.T.unsqueeze(-1).unsqueeze(-1).to(device)
        out, rec = model(x)
        del x, rec
        loss_val = loss_f(out, y)
        class_prob = F.softmax(out, dim = 1)
        del out
        # preds = class_prob.argmax(dim=1)
        loss_lst.append(float(loss_val.item()))
        top1_acc_val = top1accuracy(class_prob, y)
        top5_acc_val = top5accuracy(class_prob, y)
        top1_acc_lst.append(float(top1_acc_val))
        top5_acc_lst.append(float(top5_acc_val))
        del y, class_prob
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
    # compute average loss
    loss_val = round(sum(loss_lst) / len(loss_lst), 4)
    top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst),  4)
    top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
   
    return (loss_val, top1_acc, top5_acc) 

def evaluatehippo(arguments, data_loader, model, loss_f):
    """
    Input: train loader (torch loader), model (torch model), optimizer (torch optimizer)
          loss function (torch custom yolov1 loss).
    Output: loss (torch float).
    """
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = []
    model.eval()
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        x = x.view(x.shape[0], -1).to(device)
        x = x.T.unsqueeze(-1).unsqueeze(-1).to(device)
        out, rec = model(x)
        del x, rec
        class_prob = F.softmax(out, dim = 1)
        # pred = torch.argmax(class_prob, dim = 1)
        loss_val = loss_f(out, y)
        del out

        loss_lst.append(float(loss_val.item()))
        top1_acc_val = top1accuracy(class_prob, y)
        top5_acc_val = top5accuracy(class_prob, y)
        del y, class_prob
        top1_acc_lst.append(float(top1_acc_val))
        top5_acc_lst.append(float(top5_acc_val))

    # compute average loss
    loss_val = round(sum(loss_lst) / len(loss_lst), 4)
    top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst),  4)
    top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
    return (loss_val, top1_acc, top5_acc)

def trainyolov4(arguments, train_loader, model, optimizer, scheduler, loss_f, scaled_anchors, scaler, mode = 'iou'):
    model.train()
    # accumulate gradients over n accum_iters using multple batches
    # and update once minibatches are as large as a batchsize of 128
    #accum_iter = 128 / arguments["batch_size"]
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.permute(0, 3, 1, 2)
        x = x.to(device)
        #if torch.isnan(x).any():
        #    print("Input tensor x contains NaN values.")
        y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))
        #if torch.isnan(y0).any() or torch.isnan(y1).any() or torch.isnan(y2).any():
        #    print("Labels y contain NaN values.")
        # x shape :-: (batchsize, channels, height, width)
        with autocast():
            preds = model(x)
            #if torch.isnan(preds[0]).any() or torch.isnan(preds[1]).any() or torch.isnan(preds[2]).any():
            #    print("Preds tensor includes NaN values.")
            loss_val = (
                loss_f(preds[0], y0, scaled_anchors[0], mode = mode) 
                + loss_f(preds[1], y1, scaled_anchors[1], mode = mode) 
                + loss_f(preds[2], y2, scaled_anchors[2], mode = mode))
        class_acc, noobj_acc, obj_acc = class_accuracy(preds, y, arguments["conf_thresh"])
        #optimizer.zero_grad()
        #loss_val.backward()
        #optimizer.step()
        optimizer.zero_grad()
        scaler.scale(loss_val).backward()
        scaler.step(optimizer)
        scaler.update()
        if arguments["one_cycle"] == True:
            scheduler.step()
        #optimizer.zero_grad()
        #if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
        #        optimizer.step()
        #        optimizer.zero_grad()

    return (float(loss_val.item()), float(class_acc),float(noobj_acc), float(obj_acc))


def testyolov4(arguments, test_loader, model, loss_f, scaled_anchors, mode = 'iou'):
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.permute(0, 3, 1, 2)
            x = x.to(device)
            y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))
            with autocast():
                preds = model(x)
                loss_val = (
                    loss_f(preds[0], y0, scaled_anchors[0], mode = mode) 
                    + loss_f(preds[1], y1, scaled_anchors[1], mode = mode) 
                    + loss_f(preds[2], y2, scaled_anchors[2], mode = mode))
            class_acc, noobj_acc, obj_acc = class_accuracy(preds, y, arguments["conf_thresh"])
    return (float(loss_val), float(class_acc), float(noobj_acc), float(obj_acc))

def main(arguments):
    if arguments["image_size"] == 608:
       anchors = [
       [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
       [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
       [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)], 
       ]
       S = [19, 38, 76]

    if arguments["image_size"] == 416:
       anchors = [
       [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
       [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
       [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
       ]
       S = [13, 26, 52]

    last_run = 0
    last_epoch = 0
    train_loss_lst = []
    test_loss_lst = []
    train_top1acc_lst = []
    test_top1acc_lst = []
    train_top5acc_lst = []
    test_top5acc_lst = []
    train_map_lst = []
    test_map_lst = []
    test_yolo_noobj_acc_lst = []
    test_yolo_obj_acc_lst = []
    test_yolo_class_acc_lst = []
    train_yolo_noobj_acc_lst = []
    train_yolo_obj_acc_lst = []
    train_yolo_class_acc_lst = []

    test_yolo_recall_lst = []
    test_yolo_precision_lst = []
    continue_training = arguments["continue_training"]
    # if we continue training extract last epoch and last run from checkpoint
    if continue_training == True:

        checkpoint = torch.load(arguments["path_cpt_file"], map_location = device)
        last_epoch = checkpoint['epoch']
        last_run = checkpoint['run']
        print(f"Continue training from run: {last_run + 1} and epoch: {last_epoch + 1}.")
        
        if arguments['model_name'] in trains_normal:
            strip_square_brackets(f"results/{arguments['current_model']}rnn_train_loss_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}rnn_train_top1acc_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}rnn_train_top5acc_run{last_run + 1}.txt")
                
            strip_square_brackets(f"results/{arguments['current_model']}rnn_test_loss_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}rnn_test_top1acc_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}rnn_test_top5acc_run{last_run + 1}.txt")

            train_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_train_loss_run{last_run + 1}.txt", delimiter=','))
            train_top1acc_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_train_top1acc_run{last_run + 1}.txt", delimiter=','))
            train_top5acc_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_train_top5acc_run{last_run + 1}.txt", delimiter=','))

            test_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_test_loss_run{last_run + 1}.txt", delimiter=','))
            test_top1acc_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_test_top1acc_run{last_run + 1}.txt", delimiter=','))
            test_top5acc_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_test_top5acc_run{last_run + 1}.txt", delimiter=','))
        
        if arguments['model_name'] in trains_hippo:

            strip_square_brackets(f"results/{arguments['current_model']}_train_loss_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}_train_top1acc_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}_train_top5acc_run{last_run + 1}.txt")
                
            strip_square_brackets(f"results/{arguments['current_model']}_test_loss_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}_test_top1acc_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}_test_top5acc_run{last_run + 1}.txt")

            train_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}train_loss_run{last_run + 1}.txt", delimiter=','))
            train_top1acc_lst = list(genfromtxt(f"results/{arguments['current_model']}train_top1acc_run{last_run + 1}.txt", delimiter=','))
            train_top5acc_lst = list(genfromtxt(f"results/{arguments['current_model']}train_top5acc_run{last_run + 1}.txt", delimiter=','))

            test_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}test_loss_run{last_run + 1}.txt", delimiter=','))
            test_top1acc_lst = list(genfromtxt(f"results/{arguments['current_model']}test_top1acc_run{last_run + 1}.txt", delimiter=','))
            test_top5acc_lst = list(genfromtxt(f"results/{arguments['current_model']}test_top5acc_run{last_run + 1}.txt", delimiter=','))

        if arguments['model_name'] in trains_yolo:
            strip_square_brackets(f"results/{arguments['current_model']}_train_loss_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}_test_loss_run{last_run + 1}.txt")

            strip_square_brackets(f"results/{arguments['current_model']}_train_map_run{last_run + 1}.txt")
            strip_square_brackets(f"results/{arguments['current_model']}_test_map_run{last_run + 1}.txt")

            train_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}_train_loss_run{last_run + 1}.txt", delimiter=','))
            #test_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}_test_loss_run{last_run + 1}.txt", delimiter=','))

            train_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}_train_map_run{last_run + 1}.txt", delimiter=','))
            #test_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}_test_map_run{last_run + 1}.txt", delimiter=','))

    for run in range(last_run, arguments["nruns"]):
        # within the run loop if we continue training we initalise model and 
        # optimizer with parameters from the checkpoint
        if continue_training == True:

            if arguments["init_rnn"] == True:
                model = SimpleRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"], activation = 'relu').to(device)

            if arguments["init_grurnn"] == True:
                model = GruRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"]).to(device)

            if arguments["init_lstmrnn"] == True:
                model = LstmRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"]).to(device)
            
            if arguments["init_urlstmrnn"] == True:
                model = UrLstmRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"]).to(device)
            
            if arguments["init_hippo"] == True:
                model = Hippo(N = arguments['N'], maxlength = arguments['sequence_length'], output_size = arguments["output_size"]).to(device)

            if arguments["init_hippornn"] == True:
                model = HippoRNN(N = arguments['N'], maxlength = arguments['sequence_length'], output_size = arguments["output_size"]).to(device)

            if arguments["init_yolov4"] == True:
                model = YoloV4_EfficentNet(nclasses = arguments['nclasses']).to(device)

            optimizer = optim.Adam(model.parameters(), lr = arguments["lr"], weight_decay = arguments["weight_decay"], betas = (arguments["momentum"], 0.999))
            checkpoint = torch.load(arguments["path_cpt_file"], map_location = device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if arguments['model_name'] in trains_normal:
                print(f'Run {run + 1}/{arguments["nruns"]}: {arguments["model_name"]} from a previous checkpoint initalised with and {arguments["hidden_size"]} number of hidden neurons.')
                print(model)
            if arguments['model_name'] in trains_hippo:
                print(f'Run {run + 1}/{arguments["nruns"]}: {arguments["model_name"]} from a previous checkpoint initalised with {arguments["N"]} number of coefficents.')
                print(model)
            
            if arguments['model_name'] in trains_yolo:
                print(f'Run {run + 1}/{arguments["nruns"]}: {arguments["model_name"]} from a previous checkpoint at epoch {last_epoch}/{arguments["nepochs"]} initalised.')
                #print(model)
        
        elif continue_training == False:
            
            if arguments["init_rnn"] == True: 
                model = SimpleRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"], activation = 'relu').to(device)

            if arguments["init_grurnn"] == True:
                model = GruRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"]).to(device)

            if arguments["init_lstmrnn"] == True:
                model = LstmRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"]).to(device)

            if arguments["init_urlstmrnn"] == True:
                model = UrLstmRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"]).to(device)

            if arguments["init_hippo"] == True:
                model = Hippo(N = arguments['N'], maxlength = arguments['sequence_length'], output_size = arguments["output_size"]).to(device)

            if arguments["init_hippornn"] == True:
                model = HippoRNN(input_size = arguments['input_size'], hidden_size = arguments['hidden_size'], output_size = arguments["output_size"]).to(device)

            if arguments["init_yolov4"] == True:
                model = YoloV4_EfficentNet(nclasses = arguments['nclasses']).to(device)

            if arguments['model_name'] in trains_normal:
                print(model)
                print(f'Run {run + 1}/{arguments["nruns"]}: {arguments["model_name"]} initalised with {arguments["hidden_size"]} number of hidden neurons.')
                print(f'Learning rate scheduler: {"Cosine Annealing" if arguments["cosine_anneal"] else "Exponential Decay" if arguments["exponential_decay"] else "Constant"}')
                print(f'Training parameters: \n {arguments["nepochs"]} : epochs. \n {arguments["input_size"]} : input size. \n {arguments["hidden_size"]} : hidden size. \n {arguments["output_size"]} : output size. \n {arguments["lr"]} : learning rate. \n {arguments["weight_decay"]} : weight decay. \n {arguments["momentum"]} : momentum.')
            
            if arguments['model_name'] in trains_hippo:
                print(model)
                print(f'Run {run + 1}/{arguments["nruns"]}: {arguments["model_name"]} initalised with {arguments["N"]} number of coefficents.' )
                print(f'Learning rate scheduler: {"Cosine Annealing" if arguments["cosine_anneal"] else "Exponential Decay" if arguments["exponential_decay"] else "Constant"}')
                print(f'Training parameters: \n {arguments["nepochs"]} : epochs. \n {arguments["input_size"]} : input size. \n {arguments["N"]} : coefficents. \n {arguments["output_size"]} : output size. \n {arguments["lr"]} : learning rate. \n {arguments["weight_decay"]} : weight decay. \n {arguments["momentum"]} : momentum.')
            
            if arguments['model_name'] in trains_yolo:
                print(f'Run {run + 1}/{arguments["nruns"]}: {arguments["model_name"]} initalised with an EfficientNetV2 Backbone.')
                print(f'Learning rate scheduler: {"Cosine Annealing" if arguments["cosine_anneal"] else "Exponential Decay" if arguments["exponential_decay"] else "One Cycle" if arguments["one_cycle"] else "Constant"}')
                print(f'Training parameters: \n {arguments["nepochs"]} : epochs.  \n {arguments["batch_size"]} : batch size. \n {arguments["image_size"]} : image size. \n {arguments["lr"]} : learning rate. \n {arguments["weight_decay"]} : weight decay. \n {arguments["momentum"]} : momentum.')
                print(f' {arguments["conf_thresh"]} : confidence threshold.  \n {arguments["map_iou_thresh"]} : MAP IoU threshold. \n {arguments["nms_iou_thresh"]} : NMS IoU threshold.')
            optimizer = optim.Adam(model.parameters(), lr = arguments["lr"], weight_decay = arguments["weight_decay"],  betas = (arguments["momentum"], 0.999))

            # ensure that lists are empty when a new model is initalised so
            # that lists from a previous run do not interfere with storage
            train_loss_lst = []
            test_loss_lst = []
            train_top1acc_lst = []
            test_top1acc_lst = []
            train_top5acc_lst = []
            test_top5acc_lst = []
            train_map_lst = []
            test_map_lst = []
            test_yolo_noobj_acc_lst = []
            test_yolo_obj_acc_lst = []
            test_yolo_class_acc_lst = []
            train_yolo_noobj_acc_lst = []
            train_yolo_obj_acc_lst = []
            train_yolo_class_acc_lst = []
        
        if arguments["mnist"] == True:
            train_dataset = torchvision.datasets.MNIST(root = data_dir,
                                                train = True, 
                                                transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
                                                download = True)

            test_dataset = torchvision.datasets.MNIST(root =  data_dir,
                                                train = False, 
                                                transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]))
            loss_f = nn.CrossEntropyLoss()

        if arguments["mscoco"] == True:
            train_dataset = CoCoDataset("data/coco/train.csv", "data/coco/images/", "data/coco/labels/",
                          S = S, anchors = anchors, image_size = arguments["image_size"], mode = 'train')

            test_dataset = test_dataset = CoCoDataset("data/coco/test.csv", "data/coco/images/", "data/coco/labels/",
                          S = S, anchors = anchors, image_size = arguments["image_size"], mode = 'test')
            scaled_anchors = (torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(device)
            loss_f = YoloV4Loss2()


        # we drop the last batch to ensure each batch has the same size
        train_loader = DataLoader(dataset = train_dataset, num_workers = arguments["nworkers"],
                                            batch_size = arguments["batch_size"],
                                            shuffle = True, drop_last = False)
        
        test_loader = DataLoader(dataset = test_dataset, num_workers = arguments["nworkers"],
                                            batch_size = arguments["batch_size"],
                                            shuffle = False, drop_last = False)

        if arguments["cosine_anneal"] == True:
           scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = arguments["nepochs"] - arguments["warmup"], eta_min = arguments["min_lr"])

        if arguments["exponential_decay"] == True:
           scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        
        if arguments["one_cycle"] == True:
           scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = arguments["lr"], epochs = arguments["nepochs"], steps_per_epoch = len(train_loader), anneal_strategy = "cos")
        
        for epoch in range(last_epoch, arguments["nepochs"]):
            # we dont compute train map to save compute but compute test_map every 100 Epochs
            # None is changed to the computed map and if not we print an empty string
            train_map_val = None
            train_recall_val = None
            train_precision_val = None
            test_map_val = None
            test_recall_val = None
            test_precision_val = None
            # adjust learning rate for warmup 
            # 1. linear increase from inital learning rate to warmup learning rate over specfied epochs
            if (epoch + last_epoch) > 0 and (epoch + last_epoch) <= arguments["warmup"] and arguments["warmup"] != 0:
                optimizer.param_groups[0]['lr'] =  linearly_increasing_lr(arguments["lr"], arguments["warmup_lr"], epoch, arguments["warmup"])
            # 2. decrease from warmup learning rate to learning rate in specified in the lr scheduler
            # calling scheduler step before optimizer step will trigger a warning
            # however since we adjust learning rate based on epoch and then want
            # to train this warning can be ignored.
            elif (epoch + last_epoch) > arguments["warmup"] and (arguments["cosine_anneal"] == True or arguments["exponential_decay"] == True):
                scheduler.step()
            
            # print("Learning rate:", optimizer.param_groups[0]['lr'])
            
            if arguments['model_name'] in trains_normal:
                # train
                train_loss_value, train_top1acc_value, train_top5acc_value = train(arguments, train_loader, model, optimizer, loss_f)
                train_loss_lst.append(train_loss_value)
                train_top1acc_lst.append(train_top1acc_value)
                train_top5acc_lst.append(train_top5acc_value)

                # test 
                test_loss_value, test_top1acc_value, test_top5acc_value = evaluate(arguments, test_loader, model, loss_f)
                test_loss_lst.append(test_loss_value)
                test_top1acc_lst.append(test_top1acc_value)
                test_top5acc_lst.append(test_top5acc_value)
            

            if arguments['model_name'] in trains_hippo:
                # train
                train_loss_value, train_top1acc_value, train_top5acc_value = trainhippo(arguments, train_loader, model, optimizer, loss_f)
                train_loss_lst.append(train_loss_value)
                train_top1acc_lst.append(train_top1acc_value)
                train_top5acc_lst.append(train_top5acc_value)
              
                # test 
                test_loss_value, test_top1acc_value, test_top5acc_value = evaluatehippo(arguments, test_loader, model, loss_f)
                test_loss_lst.append(test_loss_value)
                test_top1acc_lst.append(test_top1acc_value)
                test_top5acc_lst.append(test_top5acc_value)

            if arguments['model_name'] in trains_yolo:
                # train
                scaler = GradScaler()
                train_loss_val, train_class_acc, train_noobj_acc, train_obj_acc = trainyolov4(arguments, train_loader, model, optimizer, scheduler, loss_f, scaled_anchors, scaler, mode = 'ciou')
                # computing map on train data is expensive, so we dont do this, because:
                # 1. train data consists of around 150 k bounding boxes. Given that most images have multiple bounding boxes
                # and that before the model settles makes thousands of predictions, bounding box predictions
                # explode in either can. Since we then iterate over each bounding box predictions and check with true 
                # bounding box to obtain map.
                
                # so we comment it out the 3 lines
                # train_pred_boxes, train_true_boxes = get_bouding_boxes(train_loader, model, iou_threshold = arguments["nms_iou_thresh"], threshold=arguments["conf_thresh"], anchors = anchors)
                # print(f"Train number bounding boxes:  predictions:{len(train_pred_boxes)}  true:{len(train_true_boxes)}")
                # train_map_val = mean_average_precision(train_pred_boxes, train_true_boxes, arguments["map_iou_thresh"], nclasses=arguments["nclasses"], mode = 'iou')
                # train_map_lst.append(train_map_val)

                train_loss_lst.append(train_loss_val)
                train_yolo_noobj_acc_lst.append(train_noobj_acc)
                train_yolo_obj_acc_lst.append(train_obj_acc)
                train_yolo_class_acc_lst.append(train_class_acc)

                # test
                test_loss_val, test_class_acc, test_noobj_acc, test_obj_acc = testyolov4(arguments, test_loader, model, loss_f, scaled_anchors, mode = 'ciou')

                test_yolo_noobj_acc_lst.append(test_noobj_acc)
                test_yolo_obj_acc_lst.append(test_obj_acc)
                test_yolo_class_acc_lst.append(test_class_acc)
                test_loss_lst.append(test_loss_val)
                
                # test number of bounding boxes and test map every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch + 1 == arguments["nepochs"]:
                    test_pred_boxes, test_true_boxes = get_bouding_boxes(test_loader, model, iou_threshold = arguments["nms_iou_thresh"], confidence_threshold = arguments["conf_thresh"], anchors = anchors)
                    print(f"Test number bounding boxes:  predictions:{len(test_pred_boxes)}  true:{len(test_true_boxes)}")
                    test_map_val, test_recall_val, test_precision_val = mean_average_precision(test_pred_boxes, test_true_boxes, arguments["map_iou_thresh"], nclasses = arguments["nclasses"])
                    test_map_lst.append(test_map_val)
                    test_yolo_recall_lst.append(test_recall_val)
                    test_yolo_precision_lst.append(test_precision_val)
                print(f"Epoch:{epoch + 1}  Train[Loss:{train_loss_val} Class Acc:{train_class_acc} NoObj acc:{train_noobj_acc} Obj Acc:{train_obj_acc}]")
                print(f"Epoch:{epoch + 1}  Train[mAP:{train_map_val if train_map_val is not None else ''} Precision:{train_precision_val if train_precision_val is not None else ''} Recall:{train_recall_val if train_recall_val is not None else ''}]")
                print(f"Epoch:{epoch + 1}  Test[Loss:{test_loss_val} Class Acc:{test_class_acc} NoObj Acc:{test_noobj_acc} Obj Acc:{test_obj_acc}]")
                print(f"Epoch:{epoch + 1}  Test[mAP:{test_map_val if test_map_val is not None else ''} Precision:{test_precision_val if test_precision_val is not None else ''} Recall:{test_recall_val if test_recall_val is not None else ''}]")

            else:
                print(f"Epoch:{epoch + 1}   Train[Loss:{train_loss_value} Top1 Acc:{train_top1acc_value}  Top5 Acc:{train_top5acc_value}]")
                print(f"Epoch:{epoch + 1}   Test[Loss:{test_loss_value}   Top1 Acc:{test_top1acc_value}   Top5 Acc:{test_top5acc_value}]")

            if arguments["cosine_anneal"] == True or arguments["exponential_decay"] == True:
                scheduler.step()
            
            if arguments["save_model"] == True and ((epoch + 1) % 10) == 0 or epoch + 1 == arguments["nepochs"]:
                torch.save({
                    'epoch': epoch,
                    'run': run,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, arguments["path_cpt_file"])
                print(f"Checkpoint and evaluation at epoch {epoch + 1} stored")
                
                if arguments['model_name'] in trains_normal:
                    with open(f'results/{arguments["current_model"]}rnn_train_loss_run{run + 1}.txt','w') as values:
                        values.write(str(train_loss_lst))
                    with open(f'results/{arguments["current_model"]}rnn_train_top1acc_run{run + 1}.txt','w') as values:
                        values.write(str(train_top1acc_lst))
                    with open(f'results/{arguments["current_model"]}rnn_train_top5acc_run{run + 1}.txt','w') as values:
                        values.write(str(train_top5acc_lst))

                    with open(f'results/{arguments["current_model"]}rnn_test_loss_run{run + 1}.txt','w') as values:
                        values.write(str(test_loss_lst))
                    with open(f'results/{arguments["current_model"]}rnn_test_top1acc_run{run + 1}.txt','w') as values:
                        values.write(str(test_top1acc_lst))
                    with open(f'results/{arguments["current_model"]}rnn_test_top5acc_run{run + 1}.txt','w') as values:
                        values.write(str(test_top5acc_lst))
                
                if arguments['model_name'] in trains_hippo:
                    with open(f'results/{arguments["current_model"]}_train_loss_run{run + 1}.txt','w') as values:
                        values.write(str(train_loss_lst))
                    with open(f'results/{arguments["current_model"]}_train_top1acc_run{run + 1}.txt','w') as values:
                        values.write(str(train_top1acc_lst))
                    with open(f'results/{arguments["current_model"]}_train_top5acc_run{run + 1}.txt','w') as values:
                        values.write(str(train_top5acc_lst))

                    with open(f'results/{arguments["current_model"]}_test_loss_run{run + 1}.txt','w') as values:
                        values.write(str(test_loss_lst))
                    with open(f'results/{arguments["current_model"]}_test_top1acc_run{run + 1}.txt','w') as values:
                        values.write(str(test_top1acc_lst))
                    with open(f'results/{arguments["current_model"]}_test_top5acc_run{run + 1}.txt','w') as values:
                        values.write(str(test_top5acc_lst))
                
                if arguments['model_name'] in trains_yolo:
                    with open(f'results/{arguments["current_model"]}_train_loss_run{run + 1}.txt','w') as values:
                        values.write(str(train_loss_lst))
                    with open(f'results/{arguments["current_model"]}_train_map_run{run + 1}.txt','w') as values:
                        values.write(str(train_map_lst))

                    with open(f'results/{arguments["current_model"]}_test_loss_run{run + 1}.txt','w') as values:
                        values.write(str(test_loss_lst))
                    with open(f'results/{arguments["current_model"]}_test_map_run{run + 1}.txt','w') as values:
                        values.write(str(test_map_lst))

                    with open(f'results/{arguments["current_model"]}_train_noobj_acc_run{run + 1}.txt','w') as values:
                        values.write(str(train_yolo_noobj_acc_lst))
                    with open(f'results/{arguments["current_model"]}_train_obj_acc_run{run + 1}.txt','w') as values:
                        values.write(str(train_yolo_obj_acc_lst))
                    with open(f'results/{arguments["current_model"]}_train_class_acc_run{run + 1}.txt','w') as values:
                        values.write(str(train_yolo_class_acc_lst))

                    with open(f'results/{arguments["current_model"]}_test_noobj_acc_run{run + 1}.txt','w') as values:
                        values.write(str(test_yolo_noobj_acc_lst))
                    with open(f'results/{arguments["current_model"]}_test_obj_acc_run{run + 1}.txt','w') as values:
                        values.write(str(test_yolo_obj_acc_lst))
                    with open(f'results/{arguments["current_model"]}_test_class_acc_run{run + 1}.txt','w') as values:
                        values.write(str(test_yolo_class_acc_lst))

                    with open(f'results/{arguments["current_model"]}_test_recall_run{run + 1}.txt','w') as values:
                        values.write(str(test_yolo_recall_lst))
                    with open(f'results/{arguments["current_model"]}_test_precision_run{run + 1}.txt','w') as values:
                        values.write(str(test_yolo_precision_lst))
            # if epoch has reached last epoch reset last_epoch variable to zero
            # to ensure that once we start another run we start at the first epoch
            # and not an epoch we held onto from continuing training
            if epoch == arguments["nepochs"] - 1:
                last_epoch = 0
                continue_training = False

# if __name__ == "__main__":
#     main()
