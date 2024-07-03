# YoloV4: Real-Time Object Detection on Still Images and Videos
 
**General:**
<br>
This repo contains torch implementations for 1) YoloV4 and 2) Recurrent YoloV4 that utilize RNNs on the bounding box trajectories. The approach in 2) is novel since we are using a UR-LSTM and HiPPO LSTM to model temporal dependencies between consceucitive bounding box trajectories. 
A short demo of our YoloV4 still image detection system trained on MS COCO 2017 can be seen in Fig. 1, where we use a scene from the movie Cheng Kung Express. The full demonstration can be found [here](https://www.youtube.com/watch?v=Q30_ScFp8us). 
The short demo for YoloV4, UrYolo and Holo trained on ImageNet VID 2015 can be seen in Fig. 2.

<p align="center">
  <img src="figures/yolov4_demo.gif" alt="animated" />
  <figcaption>Fig.1 - Real-time inference using YoloV. </figcaption>
</p>


**Example Dictionary Structure**

<details>
<summary style="font-size:14px">View dictionary structure</summary>
<p>

```
.
├── application                # Real time inference tools
    └── __init__.py 
    └── yolo_watches_you.py  		# Yolo inference on webcam or video you choose
├── cpts				# Weights as checkpoint .cpt files
    └── ...
    └── efficentnet_yolov4_mscoco.cpt	# Pretrained yolov4 still-image detector
    └── efficentnet_yolov4_imagenetvid.cpt	# Pretrained yolov4 video detector
├── figures                    # Figures and graphs
    └── ....
├── loss                       # Custom PyTorch loss
    └── __init__.py  		
    └── yolov3_loss.py
├── models                     # Pytorch models
    └── __init__.py  		
    └── rnn.py                 # Rnns in base torch (simple, gru, lstm)
    └── yolov4.py		            # yolov4 architecture in base torch
├── results                    # Result textfiles
    └── ....
├── train                      # Training files
    └── __init__.py  
    └── train_rnn.py
    └── train_yolo.py 
├── utils                      	# Tools and utilities
    └── __init__.py
    └── coco_json_to_yolo.py
    └── create_csv.py
    └── get_mscoco2017.sh
    └── graphs.py
    └── utils.py
├── requierments.txt           		# Python libraries
├── setup.py                   		
├── terminal.ipynb             		# If you want to run experiments from a notebook or on google collab
├── LICENSE
└── README.md
```

</p></details>


**Getting started:**
<br>
In order to get started first `cd` into the `./thesis` dictionary and run the following lines:
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -e .
```
Depending on what libraries you may already have, you may wish to `pip install -r requirements.txt`. To run our validation or sanity check experiments the MNIST data set is requiered, which torch will download for you, so there is nothing you need to do for. However, to train the video object detector from scratch, you will need 1) the MS COCO VOC and 2) ImageNet VID data-set. You can download [MS COCO VOC]([http://host.robots.ox.ac.uk/pascal/VOC/](https://cocodataset.org/#home)) manually or by calling the following shell file: `utils/get_mscocovoc_data.sh`, which will automatically download and sort the data into the approriate folders and format for training. For [ImageNet VID]([https://www.image-net.org/)) you will have to sign up, request access and download the data by following the website guide.

