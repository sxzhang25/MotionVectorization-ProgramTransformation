import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from .core.raft import RAFT
from .core.utils import flow_viz
from .core.utils.utils import InputPadder
from tqdm import tqdm


DEVICE = 'cuda'

def load_image(imfile, max_size=512):
    img = Image.open(imfile)
    w, h = img.size
    if w > max_size and h > max_size:
        if w > h:
            img = img.resize((512, int(h * 512 / w)))
        else:
            img = img.resize((int(w * 512 / h), 512))
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    output = output[0].permute(1, 2, 0)
    return output.detach().cpu().numpy()


def viz(img, img2, flo):
    warped = warp(img2, flo)
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo, warped], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    return np.uint8(img_flo[:, :, [2,1,0]])
    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey(0)


def demo(args):
    orig_folder = os.path.join(args.path, 'rgb')
    flow_dir = os.path.join(args.path, 'flow')
    dirs = ['viz', 'forward', 'backward']
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)
    for dirname in dirs:
        if not os.path.exists(os.path.join(flow_dir, dirname)):
            os.makedirs(os.path.join(flow_dir, dirname))

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(orig_folder, '*.png')) + \
                 glob.glob(os.path.join(orig_folder, '*.jpg'))
        
        images = sorted(images)
        j = 0
        for _, (imfile1, imfile2) in tqdm(enumerate(zip(images[:-1], images[1:]))):
            if j >= args.max_frames:
                if args.add_back:
                    _, flow_backward = model(image2, image1, iters=20, test_mode=True)
                    #vis_img_backward = viz(image2, image1, flow_backward)
                    flow_backward = flow_backward[0].permute(1,2,0).cpu().numpy()
                    np.save(os.path.join(flow_dir, 'backward', f'{outname_prefix}.npy'), flow_backward)
                break
    
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_forward = model(image1, image2, iters=20, test_mode=True)
            vis_img_forward = viz(image1, image2, flow_forward)
            
            if args.add_back:
                _, flow_backward = model(image2, image1, iters=20, test_mode=True)
                vis_img_backward = viz(image2, image1, flow_backward)

            if args.add_back:
                vis_img = np.concatenate([vis_img_forward, vis_img_backward], axis=1)
            else:
                vis_img = vis_img_forward

            outname = imfile1.split('/')[-1]
            cv2.imwrite(os.path.join(flow_dir, 'viz', outname), vis_img)

            outname_prefix = os.path.splitext(outname)[0]
            #flow_low = flow_low[0].permute(1,2,0).cpu().numpy()
            flow_forward = flow_forward[0].permute(1,2,0).cpu().numpy()
            np.save(os.path.join(flow_dir, 'forward', f'{outname_prefix}.npy'), flow_forward)
            if args.add_back:
                flow_backward = flow_backward[0].permute(1,2,0).cpu().numpy()
                np.save(os.path.join(flow_dir, 'backward', f'{outname_prefix}.npy'), flow_backward)
            j += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--max_frames', type=int, default=100, help='number of frames to extract flow')
    parser.add_argument('--add_back', action='store_true', default=False, help='also compute backwards optical flow')
    args = parser.parse_args()

    demo(args)
