import argparse
import os
import time

import torch

import pytorch_mask_rcnn as pmr
from torchvision.utils import save_image
import torchvision

torch.manual_seed(40)
torch.backends.cudnn.deterministic = True

@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda: pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    
    d_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True)

    print(args)
    num_classes = max(d_test.classes) + 1
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    model.eval()

    
    print("\nevaluating...\n")
    
    for i, (image, target) in enumerate(d_test):
        if i < args.index:
            continue
        
        output = model(image)

        boxes = output["boxes"]
        masks = output["masks"]
        scores = output["scores"]
        labels = output["labels"]

        score_threashold = 0.5
        keep_index = scores > score_threashold

        print(scores)
        print(keep_index)

        scores = scores[keep_index]
        boxes = boxes[keep_index]
        masks = masks[keep_index]
        labels = labels[keep_index]

        if(len(scores.tolist()) == 0):
            continue


        labels_name = d_test.get_class_name(labels.tolist())
        print(labels)
        print(labels_name)

        img = image.mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)
        masks = masks > 0.5

        img = torchvision.utils.draw_bounding_boxes(img, boxes, labels=labels_name, font_size=100)
        img = torchvision.utils.draw_segmentation_masks(img, masks)
        img = torchvision.transforms.ToPILImage()(img)
        img.show()

        if(i >= args.index + args.iters - 1):
            break;
    
    print("Done.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--data-dir", default="/home/cvadim/fiftyone/coco-2017")
    parser.add_argument("--ckpt-path", default="/home/cvadim/PycharmProjects/ml-models/PyTorch-Simple-MaskRCNN/assets/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth")
    parser.add_argument("--index", type=int, default=7)
    parser.add_argument("--iters", type=int, default=2)
    args = parser.parse_args()
    
    args.use_cuda = True
    
    main(args)
    
    