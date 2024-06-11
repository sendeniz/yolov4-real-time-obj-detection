import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.rnn import SimpleRNN, GruRNN, LstmRNN, UrLstmRNN, HippoRNN
from models.hippo import Hippo
from rnn_train_fn import train, evaluate
from hippo_train_fn import trainhippo, evaluatehippo
from torch.cuda.amp import GradScaler
import os

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import argparse

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    # Rest of your code
    image_size = 608
    path_cpt_file = "cpts/"
    setup(rank, world_size)
    # yolo anchors rescaled between 0,1
    # yolo scales and anchors for image size 608, 608
    S = [19, 38, 76]
    anchors = [
        [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
        [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
        [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
    ]
    writer = SummaryWriter(log_dir="logs/results_yolo/")

    train_dataset = torchvision.datasets.MNIST(root = data_dir,
                                                train = True, 
                                                transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
                                                download = True)

    test_dataset = torchvision.datasets.MNIST(root =  data_dir,
                                                train = False, 
                                                transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]))

    
    loss_f = nn.CrossEntropyLoss()

    sampler_train = DistributedSampler(train_dataset, shuffle=True)
    sampler_test = DistributedSampler(test_dataset, shuffle=False)

    # we drop the last batch to ensure each batch has the same size
    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=4,
        batch_size=args.batch_size,
        drop_last=False,
        sampler=sampler_train,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=4,
        batch_size=args.batch_size,
        drop_last=False,
        sampler=sampler_test,
    )
    model =  HippoRNN(input_size = 1, hidden_size = 512, output_size = 10, maxlength=28*28).to(rank)

    ddp_model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
    optimizer = optim.Adam(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.momentum, 0.999),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.nepochs,
        steps_per_epoch=len(train_loader),
        anneal_strategy="cos",
    )
    print(f"Rank:{rank} initalised")
    world_size = dist.get_world_size()
    for run in range(args.nruns):
        for epoch in range(args.nepochs):

            # train
            scaler = GradScaler()
            train_loss, train_top1acc, train_top5acc = train(arguments, train_loader, model, optimizer, loss_f)

            # Perform all-reduce only on the master process (rank 0)
            dist.all_reduce(train_loss)
            dist.all_reduce(train_top1acc)
            dist.all_reduce(train_top5acc)

            # Divide by the world size to compute the average
            train_loss /= world_size
            train_top1acc /= world_size
            train_top5acc /= world_size


            test_loss, test_top1acc, test_top5acc = evaluate(arguments, test_loader, model, loss_f)

            # Perform all-reduce only on the master process (rank 0)
            dist.all_reduce(test_loss)
            dist.all_reduce(test_top1acc)
            dist.all_reduce(test_top5acc)

            # Divide by the world size to compute the average
            test_loss /= world_size
            test_top1acc /= world_size
            test_top5acc /= world_size

            dist.barrier()

            if dist.get_rank() == 0:
                print(
                    f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs} Train[Loss:{train_loss} Top-1 Acc:{train_top1acc} Top-5 Acc:{train_top5acc}]"
                )
                print(
                    f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs} Test[Loss:{test_loss} Top-1 Acc:{test_top1acc} Top-5 Acc:{test_top5acc}]"
                )
            # ddp throws a timeout error at last epoch so we add additional save model option
            # to also save before the last epoch and an additional barrier
            dist.barrier()
            if args.save_model and (
                (epoch + 1 == 1)
                or ((epoch + 1) % 10) == 0
                or (epoch + 1 == args.nepochs)
                or (epoch + 1 == args.nepochs - 1)
                or (epoch == args.nepochs - 1)
            ):
                dist.barrier()
                if dist.get_rank() == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "run": run,
                            "model_state_dict": ddp_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        path_cpt_file + f"hippornn_smnistrun_{run}.cpt",
                    )

                    writer.add_scalar("Loss/train", train_loss, epoch)
                    writer.add_scalar("Top-1 Acc/train", train_top1acc, epoch)
                    writer.add_scalar("Top-5 Acc/train", train_top5acc, epoch)

                    writer.add_scalar("Loss/test", test_loss, epoch)
                    writer.add_scalar("Top-1 Acc/test", test_top1acc, epoch)
                    writer.add_scalar("Top-5 Acc/test", test_top5acc, epoch)

                    print(f"Checkpoint and evaluation at epoch {epoch + 1} stored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nruns", type=int, default=1, help="Number of runs to perform"
    )
    parser.add_argument(
        "--nepochs", type=int, default=50, help="Number of epochs to perform"
    )
    parser.add_argument(
        "--input_size", type=int, default=1, help="Size of input to Rnn"
    )
    parser.add_argument(
        "--input_size", type=int, default=1, help="Size of input to Rnn"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.900, help="Momentum for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0000, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument("--nclasses", type=int, default=10, help="Number of classes")
    parser.add_argument(
        "--ngpus", type=int, default=1, help="Number of gpus used for training"
    )
    parser.add_argument(
        "--save_model", type=bool, default=False, help="Store model checkpoints"
    )

    parsed_args = parser.parse_args()
    print("YoloV4 initialized with the following parameters:")
    print(f"  - Number of runs: {parsed_args.nruns}")
    print(f"  - Number of epochs: {parsed_args.nepochs}")
    print(f"  - Batch size: {parsed_args.batch_size}")
    print(f"  - Target batch size: {parsed_args.target_batch_size}")
    print(f"  - Learning rate: {parsed_args.lr}")
    print(f"  - Momentum: {parsed_args.momentum}")
    print(f"  - Weight decay: {parsed_args.weight_decay}")
    print(f"  - Number of gpus: {parsed_args.ngpus}")

    # pass ngpus as rank and nprocs
    mp.spawn(
        main,
        args=(parsed_args.ngpus, parsed_args),
        nprocs=parsed_args.ngpus,
        join=True,
    )
