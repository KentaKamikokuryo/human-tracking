import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# from torch.backends import cudnn

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer
from fast_reid.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

# cudnn.benchmark = True

class FastReIDModelType:
    
    MOT17 = "MOT17"
    MOT20 = "MOT20"
    Market1501 = "Market1501"
    DukeMTMC = "DukeMTMC"  
    
    def __init__(self, path_model: str, model_type: str):
        self.path_model: str = path_model
        self.model_type: str = model_type
        
        model_dict = {
            "MOT17": {
                "config": os.path.join(path_model, "fast_reid/configs/MOT17/sbs_S50.yml"),
                "checkpoint": os.path.join(path_model, "fast_reid/mot17_sbs_S50.pth")
            }, 
            "MOT20": {
                "config": os.path.join(path_model, "fast_reid/configs/MOT20/sbs_S50.yml"),
                "checkpoint": os.path.join(path_model, "fast_reid/mot20_sbs_S50.pth")
            },
            "Market1501": {
                "config": os.path.join(path_model, "fast_reid/configs/Market1501/bagtricks_R101-ibn.yml"),
                "checkpoint": os.path.join(path_model, "fast_reid/market_bot_R101-ibn.pth")
            
            },
            "DukeMTMC": {
                "config": os.path.join(path_model, "fast_reid/configs/DukeMTMC/bagtricks_R101-ibn.yml"),
                "checkpoint": os.path.join(path_model, "fast_reid/duke_bot_R101-ibn.pth")
            
            }
        }
        self.config_file: str = model_dict[model_type]["config"]
        self.checkpoint_file: str = model_dict[model_type]["checkpoint"]
        
        # Check if the file exists
        if not os.path.exists(self.config_file):
            print(f"Error: Config file {self.config_file} does not exist")
            sys.exit(1)
        
        if not os.path.exists(self.checkpoint_file):
            print(f"Error: Checkpoint file {self.checkpoint_file} does not exist")
            sys.exit(1)


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


class FastReIDInterface:
    def __init__(self, config_file, weights_path, device: str = "cuda", batch_size: int=8):
        super(FastReIDInterface, self).__init__()
        if device != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.batch_size = batch_size

        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)

        if self.device != 'cpu':
            self.model = self.model.eval().to(device='cuda').half()
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def inference(self, image, detections):

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]

            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image.
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)
            # patch, scale = preprocess(patch, self.cfg.INPUT.SIZE_TEST[::-1])

            # plt.figure()
            # plt.imshow(patch)
            # plt.show()

            # Make shape with a new batch dimension which is adapted for network input
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            patch = patch.to(device=self.device).half()

            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 2048))
        # features = np.zeros((0, 768))

        for patches in batch_patches:

            # Run model
            patches_ = torch.clone(patches)
            pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)

            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():
                for n in range(np.size(nans)):
                    if nans[n]:
                        # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
                        patch_np = patches_[n, ...]
                        patch_np_ = torch.unsqueeze(patch_np, 0)
                        pred_ = self.model(patch_np_)

                        patch_np = torch.squeeze(patch_np).cpu()
                        patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                        patch_np = patch_np.numpy()

                        # plt.figure()
                        # plt.imshow(patch_np)
                        # plt.show()

            features = np.vstack((features, feat))

        return features
    
    def __repr__(self) -> str:
        return f"FastReIDInterface(config_file={self.cfg}, device={self.device}, batch_size={self.batch_size})"