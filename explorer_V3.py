import uuid
from pathlib import Path
import os
import argparse
import logging
import zipfile
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("Error: pycocotools not found. Please install it: pip install pycocotools")
    exit(1)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    AnchorGenerator,
    FastRCNNConvFCHead,
    FastRCNNPredictor,
)
try:
    from torchvision.datasets import CocoDetection
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.faster_rcnn import (
        AnchorGenerator, FastRCNNPredictor
    )
    from torchvision.models.detection.mask_rcnn import (
        MaskRCNNHeads, MaskRCNNPredictor, RPNHead
    )
    from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
except ImportError:
    print("Error: torchvision not found or incomplete. Please install/update it: pip install torchvision")
    exit(1)
try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    print("Error: transformers not found. Please install it: pip install transformers")
    exit(1)
try:
    import lightning as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping, LearningRateMonitor, ModelCheckpoint
    )
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    print("Error: PyTorch Lightning (lightning) not found. Please install it: pip install lightning")
    exit(1)
try:
    import wandb
except ImportError:
    print("Error: wandb not found. Please install it: pip install wandb")
    exit(1)
from tqdm import tqdm
import requests
import torch.distributed
try:
    import cv2
except ImportError:
    print("Error: OpenCV is required for video processing. Please install it: pip install opencv-python")
    cv2 = None

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Instance Segmentation (Mask R-CNN + SAM Backbone) - Supports Image, Directory, and Video Input")
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--image-path', type=str, default=None,
                             help="Path to a single image for mask generation.")
    input_group.add_argument('--input-dir', type=str, default=None,
                             help="Path to a directory containing multiple image frames for processing.")
    input_group.add_argument('--video-path', type=str, default=None,
                             help="Path to a video file for processing.")
    parser.add_argument('--output-dir', type=str, default="output_processed",
                        help="Directory to save processed frames/images with masks.")
    parser.add_argument('--frame-save-score-threshold', type=float, default=0.3,
                        help="Score threshold for masks to be saved in output frames/images.")
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--base-dir', type=str, default="/home/vamsik1211/Data/Assignments/Sem-2/CV/CourseProject/Instance_Segmentation Code CV Project/dataset/coco",
                             help="Base directory for COCO dataset (for training)")
    train_group.add_argument('--train-ann', type=str, default="annotations/instances_train2017.json",
                             help="Path to training annotations relative to base-dir")
    train_group.add_argument('--val-ann', type=str, default="annotations/instances_val2017.json",
                             help="Path to validation annotations relative to base-dir")
    train_group.add_argument('--model-name', type=str, default="Zigeng/SlimSAM-uniform-77",
                             help="Hugging Face model name for SAM backbone")
    train_group.add_argument('--num-epochs', type=int, default=10,
                             help="Number of training epochs (Set to 0 for inference only)")
    train_group.add_argument('--batch-size', type=int, default=16,
                             help="Per-GPU batch size for data loaders")
    train_group.add_argument('--num-workers', type=int, default=16,
                             help="Number of workers for data loaders")
    train_group.add_argument('--gradient-accumulation-steps', type=int, default=32,
                             help="Number of batches to accumulate gradients over")
    train_group.add_argument('--freeze-backbone', default=False, action='store_true',
                             help="Freeze the backbone during training")
    train_group.add_argument('--learning-rate', type=float, default=0.001,
                             help="Initial learning rate for AdamW optimizer")
    train_group.add_argument('--weight-decay', type=float, default=0.0005,
                             help="Weight decay for AdamW optimizer")
    train_group.add_argument('--target-size', type=int, nargs=2, default=(1024, 1024),
                             help="Target size for image resizing (height width)")
    train_group.add_argument('--seed', type=int, default=42,
                             help="Random seed for reproducibility")
    train_group.add_argument('--wandb-project', type=str, default="cv-sam-fpn-6",
                             help="Weights and Biases project name")
    train_group.add_argument("--log_steps", type=int, default=5,
                             help="Logging interval in steps")
    train_group.add_argument("--resume-ckpt-path", type=str, default=None,
                             help="Path for the checkpoint to resume training from")
    train_group.add_argument("--load-model-weights", type=str, default=None,
                             help="Path to load model weights for inference or fine-tuning (ignores optimizer state)")
    train_group.add_argument("--logs-dir", default="logs",
                             help="Folder to save logs, weights, etc.")
    train_group.add_argument("--classfier-loss-weight", type=float, default=1.0, help="Classifier loss weight")
    train_group.add_argument("--bbox-reg-loss-weight", type=float, default=1.0, help="Bounding box regression loss weight")
    train_group.add_argument("--mask-loss-weight", type=float, default=2.0, help="Mask loss weight")
    train_group.add_argument("--objectness-loss-weight", type=float, default=1.0, help="Objectness loss weight")
    train_group.add_argument("--rpn-bbox-reg-loss-weight", type=float, default=1.0, help="RPN bbox loss weight")
    train_group.add_argument("--one-cycle-lr-pct", type=float, default=0.3, help="OneCycleLR pct_start")
    train_group.add_argument("--one-cycle-lr-three-phase", action="store_true", default=True, help="Enable three phase OneCycleLR")

    parsed_args = parser.parse_args()

    # Enhanced argument validation
    if parsed_args.num_epochs < 0:
        parser.error("num_epochs cannot be negative.")
    if parsed_args.num_epochs == 0 and not (parsed_args.image_path or parsed_args.input_dir or parsed_args.video_path):
        parser.error("Please specify an input source (--image-path, --input-dir, or --video-path) when not training (num_epochs = 0).")
    if parsed_args.video_path and cv2 is None:
        parser.error("OpenCV (cv2) is required for video processing (--video-path) but could not be imported.")
    if not 0 < parsed_args.one_cycle_lr_pct < 1:
        parser.error("one_cycle_lr_pct must be between 0 and 1 (exclusive).")
    if not (isinstance(parsed_args.target_size, (list, tuple)) and len(parsed_args.target_size) == 2 and all(isinstance(x, int) and x > 0 for x in parsed_args.target_size)):
        parser.error("target_size must be two positive integers (height width).")
    if parsed_args.batch_size < 1:
        parser.error("batch_size must be at least 1.")
    if parsed_args.num_workers < 0:
        parser.error("num_workers cannot be negative.")
    if parsed_args.gradient_accumulation_steps < 1:
        parser.error("gradient_accumulation_steps must be at least 1.")
    if parsed_args.learning_rate <= 0:
        parser.error("learning_rate must be positive.")
    if parsed_args.weight_decay < 0:
        parser.error("weight_decay cannot be negative.")

    return parsed_args

# --- Global Settings & Logging ---
args = parse_args()
args.logs_dir = Path(args.logs_dir)
args.output_dir = Path(args.output_dir)

if args.resume_ckpt_path and args.load_model_weights:
    print("Warning: Both --resume-ckpt-path and --load-model-weights are set. --resume-ckpt-path will be used for resuming training.")
if args.resume_ckpt_path:
    args.resume_ckpt_path = Path(args.resume_ckpt_path)
    if not args.resume_ckpt_path.is_file():
        print(f"Warning: Resume checkpoint path {args.resume_ckpt_path} does not exist.")
if args.load_model_weights:
    args.load_model_weights = Path(args.load_model_weights)
    if not args.load_model_weights.is_file():
        print(f"Warning: Model weights path {args.load_model_weights} does not exist.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_logger = logging.getLogger(__name__)

loss_weights = {
    'loss_classifier': args.classfier_loss_weight, 'loss_box_reg': args.bbox_reg_loss_weight,
    'loss_mask': args.mask_loss_weight, 'loss_objectness': args.objectness_loss_weight,
    'loss_rpn_box_reg': args.rpn_bbox_reg_loss_weight
}

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- Dataset Download Utilities ---
def download_file(url, dest_path_str):
    try:
        dest_path = Path(dest_path_str)
        console_logger.info(f"Starting download: {url} -> {dest_path}")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name, total=total_size, unit='iB', unit_scale=True,
            unit_divisor=1024, disable=(total_size == 0), leave=False
        ) as bar:
            for data in response.iter_content(chunk_size=block_size):
                size = file.write(data)
                bar.update(size)
        if total_size != 0 and bar.n != total_size:
            console_logger.warning(f"Download size mismatch for {dest_path.name}: Expected {total_size}, got {bar.n}")
        console_logger.info(f"Download completed: {dest_path} ({bar.n / (1024*1024):.2f} MB)")
    except requests.exceptions.Timeout:
        console_logger.error(f"Timeout downloading {url}")
        if Path(dest_path_str).exists():
            Path(dest_path_str).unlink()
        raise
    except requests.exceptions.RequestException as e:
        console_logger.error(f"Failed to download {url}: {e}")
        if Path(dest_path_str).exists():
            Path(dest_path_str).unlink()
        raise
    except IOError as e:
        console_logger.error(f"Failed to write to {dest_path_str}: {e}")
        raise

def extract_zip(zip_path_str, extract_to_str):
    zip_path = Path(zip_path_str)
    extract_to = Path(extract_to_str)
    try:
        console_logger.info(f"Extracting {zip_path.name} to {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        console_logger.info(f"Extracted {zip_path.name}")
    except zipfile.BadZipFile as e:
        console_logger.error(f"Bad zip file: {zip_path} - {e}. Please delete it and retry.")
        raise
    except Exception as e:
        console_logger.error(f"Failed to extract {zip_path}: {e}")
        raise

def download_coco_dataset(base_dir_str):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        base_dir = Path(base_dir_str)
        images_dir = base_dir / "images"
        annotations_dir = base_dir / "annotations"
        temp_download_dir = base_dir / "downloads_temp"
        console_logger.info("Checking COCO dataset status...")
        base_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        annotations_dir.mkdir(exist_ok=True)
        temp_download_dir.mkdir(exist_ok=True)
        required_ann_files_rel = [
            "instances_train2017.json", "instances_val2017.json",
        ]
        required_img_dirs_rel = ["train2017", "val2017"]
        all_complete = True
        for fname in required_ann_files_rel:
            if not (annotations_dir / fname).is_file():
                console_logger.warning(f"Annotation file missing: {annotations_dir / fname}")
                all_complete = False
                break
        if all_complete:
            for dname in required_img_dirs_rel:
                img_dir = images_dir / dname
                if not img_dir.is_dir() or not any(img_dir.iterdir()):
                    console_logger.warning(f"Image directory missing or empty: {img_dir}")
                    all_complete = False
                    break
        if not all_complete:
            console_logger.info("Dataset incomplete. Attempting download/extraction...")
            for key, url in COCO_URLS.items():
                zip_path = temp_download_dir / f"{key}.zip"
                is_annotation = (key == "annotations")
                extract_target = annotations_dir if is_annotation else images_dir
                content_dir_name = "" if is_annotation else key
                final_content_path = annotations_dir if is_annotation else images_dir / content_dir_name
                content_present = False
                if is_annotation:
                    content_present = all((annotations_dir / f).is_file() for f in required_ann_files_rel)
                elif content_dir_name:
                    content_present = final_content_path.is_dir() and any(final_content_path.iterdir())
                if content_present:
                    console_logger.info(f"Content for '{key}' seems present. Skipping.")
                    continue
                if not zip_path.exists():
                    try:
                        download_file(url, str(zip_path))
                    except Exception as e:
                        console_logger.error(f"Download failed for {url}. Error: {e}. Skipping.")
                        continue
                if zip_path.exists():
                    try:
                        extract_zip(str(zip_path), str(extract_target))
                    except Exception as e:
                        console_logger.error(f"Extraction failed for {zip_path}. Check file integrity or delete and retry. Error: {e}")
                else:
                    console_logger.warning(f"Cannot extract {zip_path.name} as it wasn't downloaded successfully.")
            console_logger.info(f"Dataset download/extraction process finished.")
        else:
            console_logger.info("COCO dataset appears complete.")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

# --- Model Definition ---
def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False

class SamEmbeddingModelWithFPN(torch.nn.Module):
    def __init__(self, model_name, encoder_output_channels=168, fpn_out_channels=256):
        super().__init__()
        self.model_name = model_name
        self.encoder_output_channels = encoder_output_channels
        self.fpn_out_channels = fpn_out_channels
        self.out_channels = fpn_out_channels
        self.patch_size = 16
        try:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, config=config)
        except Exception as e:
            console_logger.error(f"Failed to load SAM model {model_name}: {e}")
            raise
        if args.freeze_backbone:
            freeze_model(self.model)
        self._fpn_in_channels_list = [encoder_output_channels]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self._fpn_in_channels_list,
            out_channels=fpn_out_channels,
        )

    def forward(self, inputs):
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        if inputs.ndim != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got shape {inputs.shape}")
        batch_size, channels, height, width = inputs.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"Input dimensions ({height}, {width}) must be divisible by patch_size ({self.patch_size})")
        try:
            outputs = self.model(pixel_values=inputs, output_hidden_states=True)
            if outputs.vision_hidden_states is None:
                raise ValueError("SAM model did not return vision_hidden_states.")
            features = outputs.vision_hidden_states[-1]
        except Exception as e:
            console_logger.error(f"SAM forward pass failed: {e}")
            raise
        features_reshaped = features.permute(0, 3, 1, 2)
        feature_channels = features_reshaped.shape[1]
        fpn_expected_channels = self._fpn_in_channels_list[0]
        if feature_channels != fpn_expected_channels:
            raise ValueError(
                f"Channel dimension mismatch: Features have {feature_channels} channels, "
                f"but FPN expects {fpn_expected_channels}."
            )
        features_for_fpn = {'0': features_reshaped}
        fpn_output = self.fpn(features_for_fpn)
        return fpn_output

class MaskRCNNLightning(pl.LightningModule):
    def __init__(self, model_name=args.model_name, lr=args.learning_rate, weight_decay=args.weight_decay):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        # Initialize Backbone
        backbone = SamEmbeddingModelWithFPN(model_name=model_name).eval()

        # Define RPN and RoI heads components
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        mask_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)

        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7),
            [256, 256, 256, 256],
            [1024],
            norm_layer=torch.nn.BatchNorm2d
        )
        mask_head = MaskRCNNHeads(
            backbone.out_channels,
            [256, 256, 256, 256],
            1,
            norm_layer=torch.nn.BatchNorm2d
        )

        box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=91)
        rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=91)

        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=None,
            min_size=args.target_size[0],
            max_size=args.target_size[1],
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            rpn_pre_nms_top_n_train=3000,
            rpn_pre_nms_top_n_test=3000,
            rpn_post_nms_top_n_train=3000,
            rpn_post_nms_top_n_test=3000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=1024,
            rpn_positive_fraction=0.5,
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=3000,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
            mask_roi_pool=mask_roi_pooler,
            mask_head=mask_head,
            mask_predictor=mask_predictor
        )

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.model(images, targets)
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        if images is None or targets is None or len(images) == 0:
            console_logger.warning(f"[{self.global_rank}] Empty or invalid batch at step {self.global_step}. Returning zero loss.")
            self.log('train/loss', 0.0, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=1)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        actual_batch_size = len(images)
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        try:
            self.model.train()
            loss_dict = self.model(images, targets)
        except Exception as e:
            console_logger.error(f"[{self.global_rank}] Model forward/loss calculation failed at step {self.global_step}: {e}", exc_info=True)
            self.log('train/loss', 0.0, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if not loss_dict or not isinstance(loss_dict, dict):
            console_logger.warning(f"[{self.global_rank}] Model returned empty or invalid loss_dict at step {self.global_step}. Returning zero loss.")
            self.log('train/loss', 0.0, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        losses = []
        valid_loss_keys = []
        for key, loss in loss_dict.items():
            if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
                weight = loss_weights.get(key, 1.0)
                losses.append(loss * weight)
                valid_loss_keys.append(key)
            else:
                console_logger.warning(f"[{self.global_rank}] Invalid or non-tensor value for loss key '{key}' at step {self.global_step}: {loss}. Ignoring this component.")

        if not losses:
            console_logger.warning(f"[{self.global_rank}] No valid loss components found in loss_dict at step {self.global_step}. Returning zero loss.")
            self.log('train/loss', 0.0, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = sum(losses)

        if not torch.is_tensor(total_loss) or torch.isnan(total_loss) or torch.isinf(total_loss):
            console_logger.error(f"[{self.global_rank}] Invalid total loss calculated ({total_loss}) at step {self.global_step} from keys {valid_loss_keys}. Returning zero loss.")
            self.log('train/loss', 0.0, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log('train/loss', total_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
        for key in valid_loss_keys:
            loss_val = loss_dict[key]
            if torch.is_tensor(loss_val) and not torch.isnan(loss_val) and not torch.isinf(loss_val):
                self.log(f'train/{key}', loss_val, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)

        if self.global_step % 50 == 0 and self.logger and hasattr(self.logger, 'experiment') and self.logger.experiment and self.trainer.is_global_zero:
            try:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and 'backbone' not in name:
                            if not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any():
                                self.logger.experiment.log({
                                    f"gradients/{name}": wandb.Histogram(param.grad.detach().cpu().numpy())
                                }, step=self.global_step)
                            else:
                                console_logger.warning(f"[{self.global_rank}] Found NaN/Inf in gradient for {name} at step {self.global_step}. Skipping histogram logging.")
            except Exception as e:
                console_logger.warning(f"[{self.global_rank}] Failed to log gradient histogram at step {self.global_step}: {e}")
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        if images is None or targets is None or len(images) == 0:
            console_logger.warning(f"[{self.global_rank}] Empty or invalid validation batch at step {self.global_step}. Returning zero loss.")
            self.log('val/loss', 0.0, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=1)
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        actual_batch_size = len(images)
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        try:
            self.model.eval()
            with torch.no_grad():
                loss_dict = self.model(images, targets)
        except Exception as e:
            console_logger.error(f"[{self.global_rank}] Validation forward/loss calculation failed at step {self.global_step}: {e}")
            self.log('val/loss', 0.0, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        if not loss_dict or not isinstance(loss_dict, dict):
            console_logger.warning(f"[{self.global_rank}] Model returned empty or invalid loss_dict in validation at step {self.global_step}. Returning zero loss.")
            self.log('val/loss', 0.0, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        losses = []
        valid_loss_keys = []
        for key, loss in loss_dict.items():
            if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
                weight = loss_weights.get(key, 1.0)
                losses.append(loss * weight)
                valid_loss_keys.append(key)
            else:
                console_logger.warning(f"[{self.global_rank}] Invalid or non-tensor value for validation loss key '{key}' at step {self.global_step}: {loss}. Ignoring this component.")

        if not losses:
            console_logger.warning(f"[{self.global_rank}] No valid loss components found in validation loss_dict at step {self.global_step}. Returning zero loss.")
            self.log('val/loss', 0.0, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        total_loss = sum(losses)

        if not torch.is_tensor(total_loss) or torch.isnan(total_loss) or torch.isinf(total_loss):
            console_logger.error(f"[{self.global_rank}] Invalid total validation loss calculated ({total_loss}) at step {self.global_step} from keys {valid_loss_keys}. Returning zero loss.")
            self.log('val/loss', 0.0, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        self.log('val/loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)
        for key in valid_loss_keys:
            loss_val = loss_dict[key]
            if torch.is_tensor(loss_val) and not torch.isnan(loss_val) and not torch.isinf(loss_val):
                self.log(f'val/{key}', loss_val, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=actual_batch_size)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=True
        )

        total_steps = 0
        if self.trainer and hasattr(self.trainer, 'datamodule') and self.trainer.datamodule and self.trainer.max_epochs is not None and self.trainer.max_epochs > 0:
            try:
                train_loader = self.trainer.datamodule.train_dataloader()
                steps_per_epoch_per_process = len(train_loader)
                if steps_per_epoch_per_process <= 0:
                    raise ValueError("len(train_dataloader) returned <= 0")
                total_optimization_steps = steps_per_epoch_per_process * self.trainer.max_epochs
                console_logger.info(f"[{self.global_rank}] Scheduler calculation: Steps/epoch/process = {steps_per_epoch_per_process}, Max epochs = {self.trainer.max_epochs}")
                console_logger.info(f"[{self.global_rank}] Calculated total optimization steps for OneCycleLR: {total_optimization_steps}")
                if total_optimization_steps <= 0:
                    raise ValueError("Calculated total_optimization_steps is not positive.")
                total_steps = total_optimization_steps
            except Exception as e:
                console_logger.warning(f"[{self.global_rank}] Failed to accurately calculate total steps for LR scheduler: {e}. Using fallback estimate (2000 * max_epochs).")
                fallback_estimate = 2000 * self.trainer.max_epochs
                total_steps = max(1000, fallback_estimate)
                console_logger.info(f"[{self.global_rank}] Using fallback total_steps = {total_steps} for scheduler.")
        else:
            console_logger.warning(f"[{self.global_rank}] Trainer/datamodule/max_epochs not fully available during optimizer configuration. Using fallback total_steps = 10000 for scheduler.")
            total_steps = 10000

        console_logger.info(f"[{self.global_rank}] OneCycleLR using total_steps = {total_steps}")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 10,
            total_steps=total_steps,
            pct_start=args.one_cycle_lr_pct,
            final_div_factor=1e5,
            three_phase=args.one_cycle_lr_three_phase
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

# --- Data Handling ---
def resize_and_pad_image(image: Image.Image, target_size=(1024, 1024)):
    try:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        img_array = np.array(image)
        if img_array.ndim == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        elif img_array.shape[2] != 3:
            raise ValueError(f"Unexpected number of channels: {img_array.shape[2]}")
        h, w = img_array.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid original image dimensions: {w}x{h}")
        target_h, target_w = target_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h <= 0 or new_w <= 0:
            raise ValueError(f"Invalid resized dimensions: {new_w}x{new_h}")
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized_array = np.array(resized)
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        top = max(0, (target_h - new_h) // 2)
        left = max(0, (target_w - new_w) // 2)
        slice_h = min(new_h, target_h - top)
        slice_w = min(new_w, target_w - left)
        if slice_h > 0 and slice_w > 0:
            padded[top:top+slice_h, left:left+slice_w, :] = resized_array[:slice_h, :slice_w, :]
        else:
            console_logger.warning("Zero-sized slice during padding.")
        metadata = {'original_size': (h, w), 'resized_size': (new_h, new_w), 'padding': (top, left), 'scale': scale}
        return Image.fromarray(padded), metadata
    except Exception as e:
        console_logger.error(f"Error in resize_and_pad_image: {e}")
        padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        metadata = {'original_size': (0, 0), 'resized_size': (0, 0), 'padding': (0, 0), 'scale': 1.0}
        return Image.fromarray(padded), metadata

def prepare_coco_datasets(args):
    base_dir = Path(args.base_dir)
    train_dir = base_dir / "images" / "train2017"
    val_dir = base_dir / "images" / "val2017"
    train_ann = base_dir / args.train_ann
    val_ann = base_dir / args.val_ann
    try:
        download_coco_dataset(str(base_dir))
    except Exception as e:
        console_logger.error(f"Failed to download COCO dataset: {e}")
        raise
    for path in [train_ann, val_ann, train_dir, val_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        console_logger.info(f"Verified: {path}")
    
    # Define data augmentation for training
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.3),
        T.ToTensor(),
    ])
    val_transform = T.ToTensor()
    
    try:
        train_dataset = CocoDetection(root=str(train_dir), annFile=str(train_ann), transform=train_transform)
        val_dataset = CocoDetection(root=str(val_dir), annFile=str(val_ann), transform=val_transform)
        console_logger.info(f"Loaded COCO datasets: {len(train_dataset)} train, {len(val_dataset)} val images.")
    except Exception as e:
        console_logger.error(f"Error loading COCO dataset from {base_dir}: {e}")
        raise
    return train_dataset, val_dataset, None

class COCODataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, tensor_transform=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tensor_transform = tensor_transform
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = True

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            shuffle = False
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=lambda b: collate_fn(b, lambda x: x),
            pin_memory=self.pin_memory, sampler=sampler,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )

    def val_dataloader(self):
        sampler = None
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda b: collate_fn(b, lambda x: x),
            pin_memory=self.pin_memory, sampler=sampler,
            persistent_workers=self.num_workers > 0
        )

def collate_fn(batch, tensor_transform):
    if not callable(tensor_transform):
        console_logger.error(f"tensor_transform is not callable: {type(tensor_transform)}")
        return None, None
    pil_images = []
    targets_raw = []
    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            img_data, ann_data = item
            if isinstance(img_data, torch.Tensor):
                pil_images.append(T.ToPILImage()(img_data))
                targets_raw.append(ann_data)
            elif isinstance(img_data, Image.Image):
                pil_images.append(img_data)
                targets_raw.append(ann_data)
            elif isinstance(img_data, np.ndarray):
                try:
                    pil_images.append(Image.fromarray(img_data))
                    targets_raw.append(ann_data)
                except Exception as e:
                    console_logger.warning(f"Failed to convert numpy array to PIL: {e}. Skipping item.")
            else:
                console_logger.warning(f"Unexpected image data type: {type(img_data)}. Skipping item.")
        else:
            console_logger.warning(f"Unexpected batch item format: {type(item)}. Skipping item.")
    if not pil_images:
        console_logger.warning("No valid images in batch.")
        return None, None
    processed_images = []
    all_metadata = []
    target_size_h, target_size_w = tuple(args.target_size)
    for img_pil in pil_images:
        processed_pil, metadata = resize_and_pad_image(img_pil, tuple(args.target_size))
        try:
            processed_images.append(tensor_transform(processed_pil))
            all_metadata.append(metadata)
        except Exception as e:
            console_logger.error(f"ToTensor transform failed: {e}. Using zero tensor.")
            processed_images.append(torch.zeros((3, target_size_h, target_size_w)))
            all_metadata.append({'original_size': (0, 0), 'resized_size': (0, 0), 'padding': (0, 0), 'scale': 1.0})
    processed_targets = []
    for idx, target_list in enumerate(targets_raw):
        metadata = all_metadata[idx]
        orig_h, orig_w = metadata['original_size']
        new_h, new_w = metadata['resized_size']
        pad_top, pad_left = metadata['padding']
        scale = metadata['scale']
        if orig_h <= 0 or orig_w <= 0:
            processed_targets.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32), 'labels': torch.empty(0, dtype=torch.int64),
                'masks': torch.empty((0, target_size_h, target_size_w), dtype=torch.uint8),
                'area': torch.empty(0, dtype=torch.float32), 'iscrowd': torch.empty(0, dtype=torch.uint8)
            })
            continue
        target_dict = {'boxes': [], 'labels': [], 'masks': [], 'area': [], 'iscrowd': []}
        valid_indices = []
        if not isinstance(target_list, (list, tuple)):
            console_logger.warning(f"Annotations for image {idx} is not a list/tuple ({type(target_list)}). Skipping targets.")
            target_list = []
        for i, t in enumerate(target_list):
            if not isinstance(t, dict) or not all(k in t for k in ['bbox', 'category_id', 'segmentation']):
                console_logger.warning(f"Skipping invalid annotation format in image {idx}, target {i}: {t}")
                continue
            x, y, w, h = t['bbox']
            if w <= 0 or h <= 0:
                continue
            x_min, y_min, x_max, y_max = x, y, x + w, y + h
            scaled_x_min, scaled_y_min = x_min * scale, y_min * scale
            scaled_x_max, scaled_y_max = x_max * scale, y_max * scale
            padded_x_min = max(0, scaled_x_min + pad_left)
            padded_y_min = max(0, scaled_y_min + pad_top)
            padded_x_max = min(target_size_w, scaled_x_max + pad_left)
            padded_y_max = min(target_size_h, scaled_y_max + pad_top)
            if padded_x_max <= padded_x_min or padded_y_max <= padded_y_min:
                continue
            target_dict['boxes'].append([padded_x_min, padded_y_min, padded_x_max, padded_y_max])
            target_dict['labels'].append(t['category_id'])
            target_dict['area'].append(t.get('area', 0.0))
            target_dict['iscrowd'].append(t.get('iscrowd', 0))
            valid_indices.append(i)
        masks_np = []
        original_valid_targets = [target_list[i] for i in valid_indices]
        for t in original_valid_targets:
            seg = t['segmentation']
            mask = None
            try:
                if isinstance(seg, dict):
                    rle_input = {'counts': seg.get('counts'), 'size': seg.get('size')}
                    if not rle_input['counts'] or not rle_input['size']:
                        raise ValueError("Invalid RLE format")
                    if isinstance(rle_input['counts'], list):
                        rle = coco_mask.frPyObjects([rle_input], orig_h, orig_w)
                    else:
                        rle = [rle_input]
                    mask = coco_mask.decode(rle)
                    if mask.ndim == 3:
                        mask = np.sum(mask, axis=2, dtype=np.uint8).clip(0, 1)
                elif isinstance(seg, list) and seg:
                    rles = coco_mask.frPyObjects(seg, orig_h, orig_w)
                    mask = coco_mask.decode(rles)
                    if mask.ndim == 3:
                        mask = np.sum(mask, axis=2, dtype=np.uint8).clip(0, 1)
                else:
                    raise ValueError(f"Unsupported segmentation format: {type(seg)}")
                if mask is None or mask.ndim != 2 or mask.shape != (orig_h, orig_w):
                    console_logger.warning(f"Mask decode/validation failed. Using zero mask.")
                    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                mask_pil = Image.fromarray(mask * 255, mode='L')
                resized_mask_pil = mask_pil.resize((new_w, new_h), Image.Resampling.NEAREST)
                resized_mask_np = np.array(resized_mask_pil) / 255.0
                padded_mask_np = np.zeros((target_size_h, target_size_w), dtype=np.uint8)
                slice_h = min(new_h, target_size_h - pad_top)
                slice_w = min(new_w, target_size_w - pad_left)
                if slice_h > 0 and slice_w > 0:
                    padded_mask_np[pad_top:pad_top+slice_h, pad_left:pad_left+slice_w] = (resized_mask_np[:slice_h, :slice_w] > 0.5).astype(np.uint8)
                masks_np.append(padded_mask_np)
            except Exception as e:
                console_logger.error(f"Error processing segmentation (Area: {t.get('area', '?')} ImgIdx: {idx}): {e}")
                masks_np.append(np.zeros((target_size_h, target_size_w), dtype=np.uint8))
        if target_dict['boxes']:
            target_dict['boxes'] = torch.tensor(target_dict['boxes'], dtype=torch.float32)
            target_dict['labels'] = torch.tensor(target_dict['labels'], dtype=torch.int64)
            if masks_np:
                target_dict['masks'] = torch.tensor(np.stack(masks_np), dtype=torch.uint8)
            else:
                target_dict['masks'] = torch.empty((0, target_size_h, target_size_w), dtype=torch.uint8)
            target_dict['area'] = torch.tensor(target_dict['area'], dtype=torch.float32)
            target_dict['iscrowd'] = torch.tensor(target_dict['iscrowd'], dtype=torch.uint8)
        else:
            target_dict = {
                'boxes': torch.empty((0, 4), dtype=torch.float32), 'labels': torch.empty(0, dtype=torch.int64),
                'masks': torch.empty((0, target_size_h, target_size_w), dtype=torch.uint8),
                'area': torch.empty(0, dtype=torch.float32), 'iscrowd': torch.empty(0, dtype=torch.uint8)
            }
        processed_targets.append(target_dict)
    try:
        images_batch = torch.stack(processed_images)
    except Exception as e:
        console_logger.error(f"Failed to stack images into batch: {e}")
        return None, None
    return images_batch, processed_targets

# --- Inference Helper Functions ---
def process_and_save_frame(model, image_path, output_path, device, target_size, score_threshold):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        console_logger.error(f"Error opening {image_path}: {e}")
        return False
    processed_image_pil, _ = resize_and_pad_image(image, tuple(target_size))
    try:
        img_tensor = T.ToTensor()(processed_image_pil).to(device)
    except Exception as e:
        console_logger.error(f"Error converting image to tensor: {e}")
        return False
    images = [img_tensor]
    model.eval()
    try:
        with torch.no_grad():
            predictions = model(images)
    except Exception as e:
        console_logger.error(f"Prediction failed for {image_path}: {e}")
        return False
    if not predictions:
        console_logger.error(f"No predictions returned for {image_path}")
        return False
    pred = predictions[0]
    masks, scores, labels = pred['masks'], pred['scores'], pred['labels']
    valid_indices = scores > score_threshold
    valid_masks = masks[valid_indices]
    valid_scores = scores[valid_indices].cpu().numpy()
    valid_labels = labels[valid_indices].cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(processed_image_pil)
    ax.axis('off')
    try:
        if len(valid_masks) > 0:
            binary_masks = (valid_masks > 0.5).squeeze(1).cpu().numpy()
            legend_elements = []
            for i in range(len(binary_masks)):
                mask = binary_masks[i]
                color = plt.cm.tab10(i % 10)[:3]
                overlay = np.zeros((*mask.shape, 4))
                overlay[mask > 0, :3] = color
                overlay[mask > 0, 3] = 0.5
                ax.imshow(overlay)
                label_idx = valid_labels[i]
                label_txt = COCO_CLASSES[label_idx] if 0 <= label_idx < len(COCO_CLASSES) else f"Cls{label_idx}"
                legend_elements.append(Patch(color=color, label=f"{label_txt} ({valid_scores[i]:.2f})"))
            if legend_elements:
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        console_logger.debug(f"Saved: {output_path}")
    except Exception as e:
        console_logger.error(f"Failed to save {output_path}: {e}")
        plt.close(fig)
        return False
    finally:
        plt.close(fig)
    return True

def process_cv2_frame(model, frame_bgr, output_path, device, target_size, score_threshold):
    try:
        image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    except Exception as e:
        console_logger.error(f"Error converting frame: {e}")
        return False
    processed_image_pil, _ = resize_and_pad_image(image_pil, tuple(target_size))
    try:
        img_tensor = T.ToTensor()(processed_image_pil).to(device)
    except Exception as e:
        console_logger.error(f"Error converting frame to tensor: {e}")
        return False
    images = [img_tensor]
    model.eval()
    try:
        with torch.no_grad():
            predictions = model(images)
    except Exception as e:
        console_logger.error(f"Prediction failed for frame: {e}")
        return False
    if not predictions:
        console_logger.error("No predictions returned for frame")
        return False
    pred = predictions[0]
    masks, scores, labels = pred['masks'], pred['scores'], pred['labels']
    valid_indices = scores > score_threshold
    valid_masks = masks[valid_indices]
    valid_scores = scores[valid_indices].cpu().numpy()
    valid_labels = labels[valid_indices].cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(processed_image_pil)
    ax.axis('off')
    try:
        if len(valid_masks) > 0:
            binary_masks = (valid_masks > 0.5).squeeze(1).cpu().numpy()
            legend_elements = []
            for i in range(len(binary_masks)):
                mask = binary_masks[i]
                color = plt.cm.tab10(i % 10)[:3]
                overlay = np.zeros((*mask.shape, 4))
                overlay[mask > 0, :3] = color
                overlay[mask > 0, 3] = 0.5
                ax.imshow(overlay)
                label_idx = valid_labels[i]
                label_txt = COCO_CLASSES[label_idx] if 0 <= label_idx < len(COCO_CLASSES) else f"Cls{label_idx}"
                legend_elements.append(Patch(color=color, label=f"{label_txt} ({valid_scores[i]:.2f})"))
            if legend_elements:
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        console_logger.debug(f"Saved frame: {output_path}")
    except Exception as e:
        console_logger.error(f"Failed to save frame {output_path}: {e}")
        plt.close(fig)
        return False
    finally:
        plt.close(fig)
    return True

# --- Main Execution Block ---
if __name__ == "__main__":
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    console_logger.info("Starting script...")
    console_logger.debug(f"Arguments: {vars(args)}")

    model = None
    data_module = None
    trainer = None
    best_model_path_from_training = None

    if args.num_epochs > 0:
        try:
            train_dataset, val_dataset, tensor_transform = prepare_coco_datasets(args)
            data_module = COCODataModule(train_dataset, val_dataset, tensor_transform)
            console_logger.info("DataModule initialized.")
        except Exception as e:
            console_logger.error(f"Dataset/DataModule setup failed: {e}. Cannot train.")
            exit(1)

    try:
        model = MaskRCNNLightning(
            model_name=args.model_name,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        console_logger.info(f"Model '{args.model_name}' initialized.")
    except Exception as e:
        console_logger.error(f"Failed to initialize model: {e}")
        exit(1)

    if args.num_epochs > 0 and data_module is not None:
        console_logger.info("--- Starting Training Phase ---")

        ckpt_path_for_fit = None
        if args.resume_ckpt_path and args.resume_ckpt_path.is_file():
            ckpt_path_for_fit = str(args.resume_ckpt_path)
            console_logger.info(f"Training will resume from checkpoint: {ckpt_path_for_fit}")
        elif args.load_model_weights and args.load_model_weights.is_file():
            console_logger.info(f"Loading weights for training from: {args.load_model_weights}")
            try:
                checkpoint = torch.load(str(args.load_model_weights), map_location='cpu')
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                    console_logger.info("Loaded state_dict from checkpoint.")
                elif isinstance(checkpoint, dict):
                    model.model.load_state_dict(checkpoint, strict=False)
                    console_logger.info("Loaded raw model weights.")
                else:
                    console_logger.warning(f"Unrecognized format in {args.load_model_weights}. Could not load weights.")
                console_logger.info("Weights loaded successfully for fine-tuning/training.")
            except Exception as e:
                console_logger.error(f"Error loading weights from {args.load_model_weights}: {e}. Training may start from scratch/pretrained.")

        args.logs_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = args.logs_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        chkpt_cb = ModelCheckpoint(
            monitor='val/loss',
            dirpath=checkpoint_dir,
            filename='best-{epoch:02d}-{val/loss:.4f}',
            save_top_k=3,
            mode='min',
            save_last=True
        )
        lr_mon = LearningRateMonitor(logging_interval='step')
        early_stopping = EarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min',
            verbose=True
        )
        callbacks = [chkpt_cb, lr_mon, early_stopping]

        wandb_logger = None
        if args.wandb_project:
            try:
                wandb_logger = WandbLogger(
                    project=args.wandb_project,
                    log_model="all",
                    save_dir=str(args.logs_dir),
                    config=vars(args)
                )
                console_logger.info(f"Initialized WandB logger for project: {args.wandb_project}")
            except Exception as e:
                console_logger.warning(f"WandB setup failed: {e}. Falling back to default logger.")
                wandb_logger = None

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            console_logger.error("No GPU available. This script requires a GPU for execution.")
            exit(1)

        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            accelerator='gpu',
            devices=-1,
            strategy='ddp_find_unused_parameters_true',
            callbacks=callbacks,
            logger=wandb_logger if wandb_logger else True,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            precision='32-true',
            log_every_n_steps=args.log_steps,
            sync_batchnorm=True,
            gradient_clip_val=0.5,
            val_check_interval=1.0,
            check_val_every_n_epoch=1
        )

        if wandb_logger and trainer.is_global_zero:
            try:
                wandb_logger.watch(model, log='all', log_freq=max(100, args.log_steps))
                console_logger.info("WandB watching model graph and parameters.")
            except Exception as e:
                console_logger.warning(f"Failed to setup wandb.watch: {e}")

        if trainer.is_global_zero:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            console_logger.info(f"Trainable parameters: {num_params/1e6:.2f} M")
            trainable_params = [name for name, p in model.model.named_parameters() if p.requires_grad]
            console_logger.info(f"Trainable parameter names (first 10): {trainable_params[:10]}")
            if torch.cuda.is_available():
                try:
                    free, total = torch.cuda.mem_get_info()
                    console_logger.info(f"GPU Memory (Rank 0 Before Fit): Free={free/1e9:.2f} GB, Total={total/1e9:.2f} GB")
                except Exception as e:
                    console_logger.warning(f"Failed to get GPU memory info: {e}")

        try:
            console_logger.info("Starting training...")
            trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path_for_fit)
            console_logger.info("Training finished.")
            best_model_path_from_training = chkpt_cb.best_model_path
        except Exception as e:
            console_logger.error(f"Training failed: {e}")
            exit(1)
    elif args.num_epochs == 0:
        console_logger.info("Skipping training phase (num_epochs = 0).")
    else:
        console_logger.error("Cannot train because DataModule initialization failed.")
        exit(1)

    run_inference = args.image_path or args.input_dir or args.video_path
    is_rank_zero = torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True
    if run_inference and is_rank_zero:
        console_logger.info("--- Starting Inference Phase (Rank 0) ---")
        ckpt_to_load_inf = None
        if args.load_model_weights and args.load_model_weights.is_file():
            ckpt_to_load_inf = str(args.load_model_weights)
            console_logger.info(f"Using explicitly provided weights for inference: {ckpt_to_load_inf}")
        elif best_model_path_from_training and Path(best_model_path_from_training).is_file():
            ckpt_to_load_inf = best_model_path_from_training
            console_logger.info(f"Using best checkpoint from training for inference: {ckpt_to_load_inf}")
        elif args.resume_ckpt_path and args.resume_ckpt_path.is_file():
            if args.num_epochs == 0:
                ckpt_to_load_inf = str(args.resume_ckpt_path)
                console_logger.info(f"Using resumed checkpoint for inference (no new training): {ckpt_to_load_inf}")
            else:
                console_logger.info("Using model state after resumed training for inference.")
        else:
            console_logger.warning("No specific checkpoint specified or found for inference. Using current model state.")
        inference_model = model
        if ckpt_to_load_inf:
            console_logger.info(f"Loading model state from: {ckpt_to_load_inf}")
            try:
                inference_model = MaskRCNNLightning.load_from_checkpoint(
                    ckpt_to_load_inf, map_location='cpu'
                )
                console_logger.info("Loaded full Lightning checkpoint.")
            except Exception as e1:
                console_logger.warning(f"Failed to load as Lightning checkpoint ({e1}). Trying to load state_dict.")
                try:
                    checkpoint = torch.load(ckpt_to_load_inf, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        inference_model.load_state_dict(checkpoint['state_dict'], strict=False)
                        console_logger.info("Loaded state_dict from checkpoint.")
                    elif isinstance(checkpoint, dict):
                        inference_model.model.load_state_dict(checkpoint, strict=False)
                        console_logger.info("Loaded raw weights into model.")
                    else:
                        raise ValueError("Checkpoint format not recognized.")
                except Exception as e2:
                    console_logger.error(f"Failed to load weights/state_dict from {ckpt_to_load_inf}: {e2}. Using previous model state.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_model.to(device)
        inference_model.eval()
        inference_nn_model = inference_model.model
        console_logger.info(f"Inference using device: {device}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.video_path:
            if cv2 is None:
                console_logger.error("OpenCV not available, cannot process video.")
                exit(1)
            video_path = Path(args.video_path)
            if not video_path.is_file():
                console_logger.error(f"Video file not found: {video_path}")
                exit(1)
            console_logger.info(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                console_logger.error(f"Cannot open video: {video_path}")
                cap.release()
                exit(1)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            console_logger.info(f"Video Info: ~{frame_count} frames, {fps:.2f} FPS")
            frame_num = 0
            try:
                with tqdm(total=frame_count if frame_count > 0 else None, desc="Video Processing") as pbar:
                    while True:
                        ret, frame = cap.read()
                        frame_num += 1
                        if not ret:
                            console_logger.info("End of video or read error.")
                            break
                        out_path = args.output_dir / f"frame_{frame_num:06d}.png"
                        process_cv2_frame(inference_nn_model, frame, str(out_path), device, args.target_size, args.frame_save_score_threshold)
                        pbar.update(1)
            finally:
                cap.release()
            console_logger.info(f"Video processing finished. Output in {args.output_dir}")
        elif args.input_dir:
            input_dir = Path(args.input_dir)
            if not input_dir.is_dir():
                console_logger.error(f"Input directory not found: {input_dir}")
                exit(1)
            console_logger.info(f"Processing image directory: {input_dir}")
            img_ext = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in img_ext])
            if not files:
                console_logger.warning("No supported image files found.")
            else:
                console_logger.info(f"Found {len(files)} images.")
                for f in tqdm(files, desc="Image Processing"):
                    out_path = args.output_dir / f"{f.stem}_processed.png"
                    process_and_save_frame(inference_nn_model, str(f), str(out_path), device, args.target_size, args.frame_save_score_threshold)
                console_logger.info(f"Directory processing finished. Output in {args.output_dir}")
        elif args.image_path:
            img_path = Path(args.image_path)
            if not img_path.is_file():
                console_logger.error(f"Input image not found: {img_path}")
                exit(1)
            console_logger.info(f"Processing single image: {img_path}")
            out_path = args.output_dir / f"{img_path.stem}_processed.png"
            success = process_and_save_frame(inference_nn_model, str(img_path), str(out_path), device, args.target_size, args.frame_save_score_threshold)
            if success:
                console_logger.info(f"Single image processing finished. Output: {out_path}")
            else:
                console_logger.error("Single image processing failed.")
        else:
            console_logger.info("No inference input specified.")
    elif run_inference and not is_rank_zero:
        console_logger.debug(f"Skipping inference on Rank {torch.distributed.get_rank()}.")
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
        except Exception as e:
            console_logger.warning(f"Distributed barrier failed at script end: {e}")

    console_logger.info("Script finished.")