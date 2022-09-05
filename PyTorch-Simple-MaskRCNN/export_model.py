import argparse
import os
import time

import torch

import pytorch_mask_rcnn as pmr
from torchvision.utils import save_image
import torchvision

torch.manual_seed(40)
torch.backends.cudnn.deterministic = True

import onnx
from onnx import shape_inference
from onnxsim import simplify

from torch.utils.tensorboard import SummaryWriter

from collections import namedtuple

from torchvision import models


# default `log_dir` is "runs" - we'll be more specific here
def export_onnx(net, dummy_input, filename="default"):
    path = os.path.join(".", "nets-dump")
    if not os.path.exists(path):
        os.makedirs(path)

    filename = os.path.join(path, filename)

    torch.onnx.export(net, dummy_input, filename + ".onnx", input_names=['input'], output_names=['output'], opset_version=11)
    net = onnx.load(filename + ".onnx")

    net = shape_inference.infer_shapes(net)

    model_simp, check = simplify(net)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, filename + "_simplified.onnx")


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


@torch.no_grad()
def export_local_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    
    d_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True)

    num_classes = max(d_test.classes) + 1
    model = pmr.maskrcnn_resnet50_fpn(True, num_classes).to(device)
    model.eval()

    dummy_input = torch.rand((3, 426, 640))
    export_onnx(model, dummy_input, "mask-rcnn")


@torch.no_grad()
def export_torchvision_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    writer = SummaryWriter('nets-dump/model_experiment_1')
    
    d_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True)

    num_classes = max(d_test.classes) + 1

    model = TraceWrapper(models.detection.maskrcnn_resnet50_fpn(num_classes=num_classes, pretrained=True))
    model.eval()

    images, targets = d_test[0]
    images = torch.unsqueeze(images, dim=0)
    model.export = True

    export_onnx(model, images, "mask-rcnn2")

    writer.add_graph(model, images)
    writer.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--data-dir", default="/home/cvadim/fiftyone/coco-2017")
    parser.add_argument("--ckpt-path", default="/home/cvadim/PycharmProjects/ml-models/PyTorch-Simple-MaskRCNN/assets/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth")
    parser.add_argument("--index", type=int, default=7)
    parser.add_argument("--iters", type=int, default=2)
    args = parser.parse_args()
    
    export_torchvision_model(args)

