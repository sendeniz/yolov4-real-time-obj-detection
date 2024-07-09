import cv2 as cv
import numpy as np
import torch
import torch.optim as optim
from utils.utils import non_max_suppression, cells_to_bboxes
from models.holov4_enas import HoloV4_Enas_EfficentNet
from utils.utils import draw_bounding_box_vid
from utils.augmentations import AugmentImage
import time
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os 
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12323"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    S = [19, 38, 76]
    anchors = [
        [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
        [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
        [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
        ]

    scaled_anchors = torch.tensor(anchors) / (1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
    setup(rank, world_size)

    gate = "hippolstm"
    conf_thresh = 0.8
    nms_iou_thresh = 0.5
    map_iou_thresh = 0.5
    nworkers = 2
    nclasses = 80
    lr = 0.00001
    weight_decay = 0.0005
    path_cpt_file = 'cpts/Holov4Enas_608_vid_run_0.cpt'
    loaded_checkpoint = torch.load(path_cpt_file) #, map_location=torch.device('cpu'))
    model = HoloV4_Enas_EfficentNet(hidden_size=1024, maxlength=200, nclasses=30, gate = gate).to(rank)
    ddp_model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ddp_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    print("Pretrained YoloV Enas 608 EfficientNet S Net initialized.")
    print(f"Rank:{rank} initalised")
    world_size = dist.get_world_size()

    model.eval()


    # Path to the video file
    video_path = 'application/test_00076005.mp4'
    output_path = 'application/holo_test_00076005_processed.mp4'

    cap = cv.VideoCapture(video_path)

    fps = 0
    fps_start = 0
    prev = 0 

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_rec = cv.VideoWriter(output_path, 
                            fourcc,
                            30, (608, 608))
    # Initialize variables for FPS calculation
    fps_start = time.time()
    prev = fps_start

    c = 0
    t=0 
    carry = ((None, None), (None, None), (None, None), (None, None))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.array(frame)
        frame = cv.resize(frame, (608, 608))  # Resize frame to input size
        input_frame = frame.astype(np.float32)  # Convert frame to float32
        input_frame = AugmentImage.normalize(input_frame)  # Normalize frame if necessary
        input_frame = torch.tensor(input_frame).unsqueeze(0).permute(0, 3, 1, 2).to(rank)  # Convert to tensor
        
        # Calculate FPS
        fps_end = time.time() 
        time_diff = fps_end - prev
        fps = int(1 / time_diff)
        prev = fps_end
        
        # Draw FPS on the frame
        fps_txt = "Holo FPS: {}".format(fps)
        height, width = frame.shape[:2]
        frame = cv.putText(frame, fps_txt, (width - 140, 20), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

        # Perform inference
        with torch.no_grad():
            preds, carry = ddp_model(input_frame, t, carry)
        
        t = t + 1
        boxes = []

        anchor = torch.tensor([*anchors[0]]).to(rank) * preds.shape[2]
        boxes = cells_to_bboxes(preds.to(rank), is_preds=True, S=preds.shape[2], anchors=anchor.to(rank))[0]
        
        boxes = non_max_suppression(boxes, iou_threshold=nms_iou_thresh, confidence_threshold=conf_thresh)
        boxes
        # Draw bounding boxes on the frame
        frame = draw_bounding_box_vid(frame, boxes)
        frame = frame.astype(np.uint8)
        
        # Write frame to output video
        #cv.imwrite(f'application/img_{c}.png', frame)
        video_rec.write(frame)
        #c = c + 1
        # Uncomment if you want to exit by pressing 'q'
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release resources
    video_rec.release()
    cap.release()
    # cv.destroyAllWindows()  # Uncomment if you need to close all OpenCV windows

    print("Video processed in real time and saved.")

if __name__ == "__main__":
    # pass ngpus as rank and nprocs
    mp.spawn(
        main,
        args=(1,),
        nprocs=1,
        join=True,
    )