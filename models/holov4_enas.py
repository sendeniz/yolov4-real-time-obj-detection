import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.modules.upsampling import Upsample
from torch.utils.checkpoint import checkpoint
from models.rnn import HippoRNN_v2, UrLstmRNN_v2, HippoRNN_v3
from models.rnn import HippoRNN, UrLstmRNN
from torch.cuda.amp import autocast

class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # Leaky ReLU is activation in yolov3, but what about mish used in yolo4 ?
        # mish is primarily used in the backbone, where the activation of darknet53 is switched out with mish.
        # rest is Leaky Relu
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=1, padding=0):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            CnnBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.Upsample(scale_factor=scale),
        )

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Downsample, self).__init__()

        self.downsample = CnnBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.downsample(x)


class CnnBlockNoBnActiv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        out = self.conv(x)
        return out


class ScaledRecurrentPrediction(nn.Module):
    def __init__(
        self,
        channels,
        input_size,
        hidden_size,
        output_size, 
        maxlength,
        nclasses,
        gate="urlstm",
    ):
        super().__init__()
        self.nclasses = nclasses
        self.gate = gate
        self.scaled_pred1 = CnnBlock(channels, channels * 2, kernel_size=3, padding=1)
        self.scaled_pred2 = CnnBlock(channels * 2, (nclasses + 5) * 3, kernel_size=1, padding=0)

        if gate == "hippolstm":

            self.rnn_pred_x = HippoRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    maxlength=maxlength,
                )
            
            
            self.rnn_pred_y = HippoRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_h = HippoRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_w = HippoRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    maxlength=maxlength,
                )
        
        
        elif gate == "urlstm":

            self.rnn_pred_x = UrLstmRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                )
             
            self.rnn_pred_y = UrLstmRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                )
            
            self.rnn_pred_h = UrLstmRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,

                )
            
            self.rnn_pred_w = UrLstmRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                )
            
        elif gate == "hippolstm2":

            self.rnn_pred_x = HippoRNN_v3(
                    input_size=1,
                    hidden_size=hidden_size,
                    output_size=19*19*3,
                    maxlength=maxlength,
                )
            
            
            self.rnn_pred_y = HippoRNN_v3(
                    input_size=1,
                    hidden_size=hidden_size,
                    output_size=19*19*3,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_h = HippoRNN_v3(
                    input_size=1,
                    hidden_size=hidden_size,
                    output_size=19*19*3,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_w = HippoRNN_v3(
                    input_size=1,
                    hidden_size=hidden_size,
                    output_size=19*19*3,
                    maxlength=maxlength,
                )
            

        elif gate == 'none':
            pass
    
    def forward(self, x, t, carry):#, last_ts):
        
        carry_x = carry[0]
        carry_y = carry[1]
        carry_h = carry[2]
        carry_w = carry[3]

        out = self.scaled_pred1(x)
        out = self.scaled_pred2(out)
        out = out.reshape(
            x.shape[0], 3, self.nclasses + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)

        if self.gate == "urlstm" or self.gate == "hippolstm":
            
            # flatten input, reshape, bn and batchnorm happening inside of rnn
            # in rnn

            out_bbox_x, carry_x = checkpoint(self.rnn_pred_x, out[..., 1:2], t, carry_x, use_reentrant=False)
            out_bbox_y, carry_y = checkpoint(self.rnn_pred_y, out[..., 2:3], t, carry_y, use_reentrant=False)
            out_bbox_h, carry_h = checkpoint(self.rnn_pred_h, out[..., 3:4], t, carry_h, use_reentrant=False)
            out_bbox_w, carry_w = checkpoint(self.rnn_pred_w, out[..., 4:5], t, carry_w, use_reentrant=False)

            out[..., 1:2] = out_bbox_x
            out[..., 2:3] = out_bbox_y
            out[..., 3:4] = out_bbox_h
            out[..., 4:5] = out_bbox_w
        
        elif self.gate == "hippolstm2":
            out_bbox_x, _ = checkpoint(self.rnn_pred_x, out[..., 1:2], use_reentrant=False)
            out_bbox_y, _ = checkpoint(self.rnn_pred_y, out[..., 2:3], use_reentrant=False)
            out_bbox_h, _ = checkpoint(self.rnn_pred_h, out[..., 3:4], use_reentrant=False)
            out_bbox_w, _ = checkpoint(self.rnn_pred_w, out[..., 4:5], use_reentrant=False)

            out[..., 1:2] = out_bbox_x
            out[..., 2:3] = out_bbox_y
            out[..., 3:4] = out_bbox_h
            out[..., 4:5] = out_bbox_w

        elif self.gate == "linear":
            out_bbox_x, carry_x = checkpoint(self.rnn_pred_x, out[..., 1:2], use_reentrant=False)
            out_bbox_y, carry_y = checkpoint(self.rnn_pred_y, out[..., 2:3], use_reentrant=False)
            out_bbox_h, carry_h = checkpoint(self.rnn_pred_h, out[..., 3:4], use_reentrant=False)
            out_bbox_w, carry_w = checkpoint(self.rnn_pred_w, out[..., 4:5], use_reentrant=False)
            
            out[..., 1:2] = out_bbox_x
            out[..., 2:3] = out_bbox_y
            out[..., 3:4] = out_bbox_h
            out[..., 4:5] = out_bbox_w

        carry = (carry_x, carry_y, carry_h, carry_w)

        return out, carry


class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super().__init__()

        self.pyramid = nn.Sequential(
            nn.MaxPool2d(5, 1, 5 // 2),
            nn.MaxPool2d(9, 1, 9 // 2),
            nn.MaxPool2d(13, 1, 13 // 2),
        )

    def forward(self, x):
        features = [block(x) for block in self.pyramid]
        features = torch.cat([x] + features, dim=1)
        return features


class PathAggregationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_transform3 = CnnBlock(
            in_channels=64, out_channels=128, kernel_size=1
        )

        self.feature_transform4 = CnnBlock(
            in_channels=160, out_channels=256, kernel_size=1
        )

        self.resample5_4 = Upsample(in_channels=512, out_channels=256)
        self.resample4_3 = Upsample(in_channels=256, out_channels=128)
        self.resample3_4 = Downsample(in_channels=128, out_channels=256)
        self.resample4_5 = Downsample(in_channels=256, out_channels=512)

        self.downstream_conv5 = nn.Sequential(
            # 2048, 512
            CnnBlock(in_channels=2048, out_channels=512, kernel_size=1),
            # 512, 1024
            CnnBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            # 1024, 512
            CnnBlock(in_channels=1024, out_channels=512, kernel_size=1),
        )

        self.downstream_conv4 = nn.Sequential(
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
        )
        self.downstream_conv3 = nn.Sequential(
            CnnBlock(in_channels=256, out_channels=128, kernel_size=1),
            CnnBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            CnnBlock(in_channels=256, out_channels=128, kernel_size=1),
            CnnBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            CnnBlock(in_channels=256, out_channels=128, kernel_size=1),
        )

        self.upstream_conv4 = nn.Sequential(
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
        )
        self.upstream_conv5 = nn.Sequential(
            CnnBlock(in_channels=1024, out_channels=512, kernel_size=1),
            CnnBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            CnnBlock(in_channels=1024, out_channels=512, kernel_size=1),
            CnnBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            CnnBlock(in_channels=1024, out_channels=512, kernel_size=1),
        )

    def forward(self, scale1, scale2, scale3):
        return checkpoint(self._forward, scale1, scale2, scale3, use_reentrant=False)

    def _forward(self, scale1, scale2, scale3):
        x1 = self.feature_transform3(scale1)
        x2 = self.feature_transform4(scale2)
        x3 = scale3

        downstream_feature5 = self.downstream_conv5(x3)
        route1 = torch.cat((x2, self.resample5_4(downstream_feature5)), dim=1)
        downstream_feature4 = self.downstream_conv4(route1)
        route2 = torch.cat((x1, self.resample4_3(downstream_feature4)), dim=1)
        downstream_feature3 = self.downstream_conv3(route2)

        route3 = torch.cat(
            (self.resample3_4(downstream_feature3), downstream_feature4), dim=1
        )
        upstream_feature4 = self.upstream_conv4(route3)
        route4 = torch.cat(
            (self.resample4_5(upstream_feature4), downstream_feature5), dim=1
        )
        upstream_feature5 = self.upstream_conv5(route4)


        return upstream_feature5


# We try to stay true as close as possible to the darknet yolov3.cfg
# we however made changes and do not count [route], [shortcut] or
# [yolo] blocks as seperate layers in the network. These are generally
# not counted as seprate layers by the darknet framework either.
class HoloV4_Enas_EfficentNet(nn.Module):
    def __init__(self, *,  hidden_size, maxlength, nclasses=30, gate ="hippolstm2"):  # , scaled_anchors):
        super(HoloV4_Enas_EfficentNet, self).__init__()
        self.nclasses = nclasses
        self.gate = gate

        self.efficientnetbackbone = nn.Sequential(
            *list(models.efficientnet_v2_s(weights="DEFAULT").children())[:-2]
        )

        self.yolov4coadaptation = nn.Sequential(
            CnnBlock(
                in_channels=1280, out_channels=512, kernel_size=1, padding=0
            ),  # L1 (done)
            CnnBlock(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1
            ),  # L2 (done)
            CnnBlock(
                in_channels=1024, out_channels=512, kernel_size=1, padding=0
            ),  # L3 (done)
        )

        self.yolov4neck = nn.Sequential(
            SpatialPyramidPooling(),
            PathAggregationNet(),
        )

        if self.gate == "urlstm" or self.gate == "hippolstm" or self.gate == "none":
            self.yolov4head = nn.Sequential(
                # scale 19, 19
                # 1. x,y feature map :-: [batchsize, 3, 19, 19, 2]
                # 2. h,w feature map :-: [batchsize, 3, 19, 19, 2]
                # so input size :-: [batchsize, 3, 19, 19, 2] * 2 or [batchsize, 3, 19, 19, 4]
                ScaledRecurrentPrediction(
                    channels = 512,
                    input_size =  3 *19 * 19 * 1,
                    output_size = 3 *19 * 19 * 1,
                    hidden_size = hidden_size,
                    maxlength = maxlength,
                    nclasses = nclasses,
                    gate = self.gate,
                ),
            )
        elif self.gate == "hippolstm2":
            self.yolov4head = nn.Sequential(
                # scale 19, 19
                # 1. x,y feature map :-: [batchsize, 3, 19, 19, 2]
                # 2. h,w feature map :-: [batchsize, 3, 19, 19, 2]
                # so input size :-: [batchsize, 3, 19, 19, 2] * 2 or [batchsize, 3, 19, 19, 4]
                ScaledRecurrentPrediction(
                    channels = 512,
                    input_size =  1,
                    output_size = 3 *19 * 19 * 1,
                    hidden_size = hidden_size,
                    maxlength = 3 *19 * 19 * 1,
                    nclasses = nclasses,
                    gate = self.gate,
                ),
            )


    def forward(self, x, t, carry): #, last_ts):
    #def forward(self, x):
        # the original yolov4 backbone Darknet53 CPS returns features maps at
        # different scales, which are then further processed by the SSP and
        # PaNet. Lastly the predictions are also made at 3 different scales.
        # We adjust the backbone to accomodate an efficentnet backbone. The
        # principle however stays the same.
        backbone_scale1 = checkpoint(
            self.efficientnetbackbone[0][:4], x, use_reentrant=False
        )
        backbone_scale2 = checkpoint(
            self.efficientnetbackbone[0][4:6], backbone_scale1, use_reentrant=False
        )
        # scale 3 is out of final passed onto following parts of the architecture
        backbone_scale3 = checkpoint(
            self.efficientnetbackbone[0][6:], backbone_scale2, use_reentrant=False
        )

        x = self.yolov4coadaptation(backbone_scale3)
        ssp_out = self.yolov4neck[0](x)

        panet_scale1 = self.yolov4neck[1](
            backbone_scale1, backbone_scale2, ssp_out
        )
        if self.gate == "urlstm" or self.gate == "hippolstm" or self.gate == "none":
            #scaled_pred, carry, last_ts = checkpoint(self.yolov4head[0], panet_scale1, t=t, carry=carry, last_ts=last_ts, use_reentrant=False)
            scaled_pred, carry = checkpoint(self.yolov4head[0], panet_scale1, t=t, carry=carry, use_reentrant=False)
        elif self.gate == "hippolstm2":
            scaled_pred, carry = checkpoint(self.yolov4head[0], panet_scale1, t=t, carry=carry, use_reentrant=False)
        return (scaled_pred), (carry) #, (last_ts)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 3
    img_size = 608
    nclasses = 30
    x = []
    #carry = ((None, None), (None, None))
    # carry one for cell state one for hidden state for each carry
    
    # create n number of test data and store tensor to list
    for _ in range(n):
        tensor = torch.randn((2, 3, img_size, img_size))
        x.append(tensor)

    model = HoloV4_Enas_EfficentNet(hidden_size = 64, maxlength=5, nclasses=nclasses, gate = "hippolstm2").to(device)
    for i in range(2):
        carry = ((None, None), (None, None), (None, None), (None, None))

        for seq_idx in range(len(x)):
           
            x_t = x[seq_idx].to(device)
           
            out, carry = model(x=x_t, t=seq_idx, carry=carry) 
 
    print("Success!")