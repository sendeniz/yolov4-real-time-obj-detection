import sys
from train.train import initialize_with_args

argument_list = [
  "model_name",
  "dataset_name",
  "save_model",
  "continue_training",
  "sequence_length",
  "input_size",
  "hidden_size",
  "output_size",
  "nclasses",
  "batch_size",
  "nepochs",
  "nruns",
  "lr",
  "warmup_lr",
  "momentum",
  "warmup",
  "min_lr",
  "cosine_anneal",
  "exponential_decay",
  "one_cycle",
  "weight_decay",
  "N",
  "image_size",
  "conf_tresh",
  "map_iou_thresh",
  "nms_iou_thresh", 
]

def train_model_with_args():
  args = sys.argv[1:]
  if len(args) % 2 != 0:
    print("You need to name every argument so the parity of the length of the arguments is even.")
    return

  resList = []
  for i in range(int(len(args)/2)):
    resList.append(args[i*2:(i+1)*2])

  arguments = {}
  for cmdpair in resList:
    if cmdpair[0][0:2] != "--":
      print("please prefix your argument names with '--'")
      return
    if cmdpair[0][2:] not in argument_list:
      print(f"Could not find argument {cmdpair[0][2:]}")
      return
    arguments[cmdpair[0][2:]] = cmdpair[1]
  initialize_with_args(arguments)

train_model_with_args()
