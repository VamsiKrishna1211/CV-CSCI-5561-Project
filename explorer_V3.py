# Imports (ensure all needed libs are imported)
from pathlib import Path
import os
import argparse
import logging
import zipfile
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
# Ensure pycocotools is installed: pip install pycocotools
try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("Error: pycocotools not found. Please install it: pip install pycocotools")
    exit(1)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
# Ensure torchvision is installed: pip install torchvision
try:
    from torchvision.datasets import CocoDetection
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.faster_rcnn import (
        AnchorGenerator, FastRCNNPredictor, FastRCNNConvFCHead # Keep ConvFCHead if complex head needed
    )
    from torchvision.models.detection.mask_rcnn import (
        MaskRCNNHeads, MaskRCNNPredictor, RPNHead
    )
    from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
except ImportError:
    print("Error: torchvision not found or incomplete. Please install/update it: pip install torchvision")
    exit(1)
# Ensure transformers is installed: pip install transformers
try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    print("Error: transformers not found. Please install it: pip install transformers")
    exit(1)
# Ensure PyTorch Lightning is installed: pip install lightning
try:
    import lightning as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping, LearningRateMonitor, ModelCheckpoint
    )
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    print("Error: PyTorch Lightning (lightning) not found. Please install it: pip install lightning")
    exit(1)
from tqdm import tqdm
import requests
import torch.distributed

# Ensure OpenCV is installed for video: pip install opencv-python
try:
    import cv2
except ImportError:
    print("Error: OpenCV is required for video processing. Please install it: pip install opencv-python")
    # Do not exit immediately, allow running without video support if possible
    cv2 = None


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Instance Segmentation (Mask R-CNN + SAM Backbone) - Supports Image, Directory, and Video Input")

    # Input Modes (Mutually Exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--image-path', type=str, default=None,
                        help="Path to a single image for mask generation.")
    input_group.add_argument('--input-dir', type=str, default=None,
                        help="Path to a directory containing multiple image frames for processing.")
    input_group.add_argument('--video-path', type=str, default=None,
                        help="Path to a video file for processing.")

    # Output Arguments
    parser.add_argument('--output-dir', type=str, default="output_processed",
                        help="Directory to save processed frames/images with masks.")
    parser.add_argument('--frame-save-score-threshold', type=float, default=0.3,
                        help="Score threshold for masks to be saved in output frames/images.")

    # Training Arguments
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
    train_group.add_argument('--learning-rate', type=float, default=0.005,
                        help="Initial learning rate for AdamW optimizer")
    train_group.add_argument('--weight-decay', type=float, default=0.0005,
                        help="Weight decay for AdamW optimizer")
    train_group.add_argument('--target-size', type=int, nargs=2, default=(1024, 1024),
                        help="Target size for image resizing (height width)")
    train_group.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    train_group.add_argument('--wandb-project', type=str, default="cv-course-project",
                        help="Weights and Biases project name")
    train_group.add_argument("--log_steps", type=int, default=5,
                        help="Logging interval in steps")
    train_group.add_argument("--resume-ckpt-path", type=str, default=None,
                        help="Path for the checkpoint to resume training from")
    train_group.add_argument("--load-model-weights", type=str, default=None,
                        help="Path to load model weights for inference or fine-tuning (ignores optimizer state)")
    train_group.add_argument("--logs-dir", default="logs",
                        help="Folder to save logs, weights, etc.")
    # Loss Weights
    train_group.add_argument("--classfier-loss-weight", type=float, default=1.0, help="Classifier loss weight")
    train_group.add_argument("--bbox-reg-loss-weight", type=float, default=1.0, help="Bounding box regression loss weight")
    train_group.add_argument("--mask-loss-weight", type=float, default=1.0, help="Mask loss weight")
    train_group.add_argument("--objectness-loss-weight", type=float, default=1.0, help="Objectness loss weight")
    train_group.add_argument("--rpn-bbox-reg-loss-weight", type=float, default=1.0, help="RPN bbox loss weight")
    # LR Scheduler
    train_group.add_argument("--one-cycle-lr-pct", type=float, default=0.1, help="OneCycleLR pct_start")
    train_group.add_argument("--one-cycle-lr-three-phase", action="store_true", default=False, help="Enable three phase OneCycleLR")

    parsed_args = parser.parse_args()

    # Input validation
    if parsed_args.num_epochs <= 0 and not (parsed_args.image_path or parsed_args.input_dir or parsed_args.video_path):
         parser.error("Please specify an input source (--image-path, --input-dir, or --video-path) when not training (num_epochs <= 0).")
    if parsed_args.video_path and cv2 is None:
         parser.error("OpenCV (cv2) is required for video processing (--video-path) but could not be imported. Please install it: pip install opencv-python")

    # Check pct_start range
    if not 0 < parsed_args.one_cycle_lr_pct < 1:
        parser.error("one_cycle_lr_pct must be between 0 and 1 (exclusive).")

    return parsed_args

# --- Global Settings & Logging ---
args = parse_args()
args.logs_dir = Path(args.logs_dir)
args.output_dir = Path(args.output_dir)

# Handle checkpoint loading precedence
if args.resume_ckpt_path and args.load_model_weights:
    print("Warning: Both --resume-ckpt-path and --load-model-weights are set. --resume-ckpt-path will be used for resuming training. --load-model-weights might be used for inference if specified.")
if args.resume_ckpt_path: args.resume_ckpt_path = Path(args.resume_ckpt_path)
if args.load_model_weights: args.load_model_weights = Path(args.load_model_weights)

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

COCO_CLASSES = [ # Ensure 91 classes for standard COCO
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
] # Corrected COCO list with 91 entries (index 0 = background)


# --- Dataset Download Utilities ---
def download_file(url, dest_path_str):
    # (Implementation remains the same as before)
    try:
        dest_path = Path(dest_path_str)
        console_logger.info(f"Starting download: {url} -> {dest_path}")
        response = requests.get(url, stream=True, timeout=120) # Increased timeout
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 MB chunks

        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name, total=total_size, unit='iB', unit_scale=True,
            unit_divisor=1024, disable=(total_size == 0), leave=False
        ) as bar:
            for data in response.iter_content(chunk_size=block_size):
                size = file.write(data)
                bar.update(size)

        # Final check on size if available
        if total_size != 0 and bar.n != total_size:
             console_logger.warning(f"Download size mismatch for {dest_path.name}: Expected {total_size}, got {bar.n}")

        console_logger.info(f"Download completed: {dest_path} ({bar.n / (1024*1024):.2f} MB)")
    except requests.exceptions.Timeout:
         console_logger.error(f"Timeout downloading {url}")
         if Path(dest_path_str).exists(): Path(dest_path_str).unlink() # Clean up partial file
         raise
    except requests.exceptions.RequestException as e:
        console_logger.error(f"Failed to download {url}: {e}")
        if Path(dest_path_str).exists(): Path(dest_path_str).unlink() # Clean up
        raise
    except IOError as e:
        console_logger.error(f"Failed to write to {dest_path_str}: {e}")
        raise

def extract_zip(zip_path_str, extract_to_str):
    # (Implementation remains the same as before)
    zip_path = Path(zip_path_str)
    extract_to = Path(extract_to_str)
    try:
        console_logger.info(f"Extracting {zip_path.name} to {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True) # Ensure target exists
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        console_logger.info(f"Extracted {zip_path.name}")
        # Keep the zip file by default, uncomment below to remove
        # zip_path.unlink()
    except zipfile.BadZipFile as e:
        console_logger.error(f"Bad zip file: {zip_path} - {e}. Please delete it and retry.")
        raise
    except Exception as e: # Catch other potential errors
        console_logger.error(f"Failed to extract {zip_path}: {e}", exc_info=True)
        raise

def download_coco_dataset(base_dir_str):
    # (Implementation mostly the same, minor logging/path improvements)
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

        required_ann_files_rel = [ # Relative to annotations_dir
             "instances_train2017.json", "instances_val2017.json",
             # Add others if needed: "captions_...", "person_keypoints_..."
        ]
        required_img_dirs_rel = ["train2017", "val2017"] # Relative to images_dir

        all_complete = True
        # Check annotations
        for fname in required_ann_files_rel:
            if not (annotations_dir / fname).is_file():
                console_logger.warning(f"Annotation file missing: {annotations_dir / fname}")
                all_complete = False
                break
        # Check images
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
                content_dir_name = "" if is_annotation else key # train2017 or val2017
                final_content_path = annotations_dir if is_annotation else images_dir / content_dir_name

                # Check if final extracted content exists before downloading/extracting
                content_present = False
                if is_annotation:
                    content_present = all((annotations_dir / f).is_file() for f in required_ann_files_rel)
                elif content_dir_name: # Check image dir
                    content_present = final_content_path.is_dir() and any(final_content_path.iterdir())

                if content_present:
                    console_logger.info(f"Content for '{key}' seems present. Skipping.")
                    continue

                # Download if zip doesn't exist
                if not zip_path.exists():
                    try:
                        download_file(url, str(zip_path))
                    except Exception as e:
                        console_logger.error(f"Download failed for {url}. Error: {e}. Skipping.")
                        continue

                # Extract
                if zip_path.exists():
                    try:
                        # Extract directly to the final location if possible
                        # For annotations.zip, extract to annotations_dir
                        # For train2017.zip, extract to images_dir (will create train2017 folder)
                        extract_zip(str(zip_path), str(extract_target))
                    except Exception as e:
                         console_logger.error(f"Extraction failed for {zip_path}. Check file integrity or delete and retry. Error: {e}")
                else:
                     console_logger.warning(f"Cannot extract {zip_path.name} as it wasn't downloaded successfully.")

            console_logger.info(f"Dataset download/extraction process finished.")
            # Optional: Clean up temp dir
            # try: temp_download_dir.rmdir() # Only if empty
            # except OSError: pass
        else:
            console_logger.info("COCO dataset appears complete.")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# --- Model Definition ---
def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False

class SamEmbeddingModelWithFPN(torch.nn.Module):
    # (Implementation remains the same as before, assuming FPN outputs standard keys)
    # Ensure comments reflect that FPN output keys need to match downstream usage.
    def __init__(self, model_name=args.model_name):
         super().__init__()
         try: from transformers import AutoModel, AutoConfig
         except ImportError: raise ImportError("Please install Hugging Face Transformers: pip install transformers")

         config = AutoConfig.from_pretrained(model_name, trust_remote_code=True) # Add trust_remote_code if needed
         self.encoder_hidden_dim = getattr(config.vision_config, 'hidden_size', 768)
         self.patch_size = getattr(config.vision_config, 'patch_size', 16)

         model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
         self.model = model.vision_encoder

         if args.freeze_backbone:
             console_logger.info("Freezing backbone parameters.")
             freeze_model(self.model)

         fpn_input_channels = self.encoder_hidden_dim
         self.fpn = FeaturePyramidNetwork(
             in_channels_list=[fpn_input_channels],
             out_channels=256,
             extra_blocks=None # Or LastLevelMaxPool()
         )
         self.out_channels = 256 # FPN output channel size
         console_logger.info(f"Initialized SAM+FPN backbone. FPN input channels: {fpn_input_channels}, Output channels: {self.out_channels}")


     # Inside the SamEmbeddingModelWithFPN class:
    def forward(self, inputs):
        if isinstance(inputs, dict):
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None: raise ValueError("Input dict needs 'pixel_values'")
            inputs = pixel_values
        elif not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Input must be Tensor or dict with 'pixel_values'. Got {type(inputs)}")

        # SAM Vision Encoder Forward Pass
        encoder_output = self.model(inputs, output_hidden_states=False)

        # Extract the primary feature map
        if hasattr(encoder_output, 'last_hidden_state'):
            features = encoder_output.last_hidden_state
        elif isinstance(encoder_output, torch.Tensor):
            features = encoder_output # If the encoder directly returns the tensor
        else:
            raise TypeError(f"Unexpected output type from vision encoder: {type(encoder_output)}")

        console_logger.debug(f"Features shape from vision encoder: {features.shape}")

        # --- Determine feature map shape and prepare for FPN ---
        if features.ndim == 4:
            # Shape is likely [B, C, H, W] - Already spatial
            console_logger.debug("Features appear to be in [B, C, H, W] format.")
            # Directly use these features if channel count matches FPN input expectations
            features_reshaped = features
        elif features.ndim == 3:
            # Shape is likely [B, N, C] - Need to reshape to spatial [B, C, H, W]
            console_logger.debug("Features appear to be in [B, N, C] format. Reshaping...")
            B, N, C = features.shape # This was the line causing the error before the ndim check

            # Infer H, W of the feature grid from input image size and patch size
            H_in, W_in = inputs.shape[-2], inputs.shape[-1]
            expected_H, expected_W = H_in // self.patch_size, W_in // self.patch_size

            # Verify patch count N matches expected grid size
            if N != expected_H * expected_W:
                # Fallback check: Maybe it's a square grid inferred differently?
                H_W_sqrt = int(N**0.5)
                if H_W_sqrt * H_W_sqrt == N:
                    expected_H, expected_W = H_W_sqrt, H_W_sqrt
                    console_logger.warning(f"Inferred square feature grid H=W={H_W_sqrt} from patch count {N}. Input image size was {H_in}x{W_in}.")
                else:
                    # If neither matches, raise error
                    raise ValueError(
                        f"Patch count {N} from vision encoder doesn't match expected grid size "
                        f"{expected_H}x{expected_W} (derived from input {H_in}x{W_in} and patch size {self.patch_size}). "
                        f"Check model architecture or input processing."
                    )

            # Reshape: [B, N, C] -> [B, H, W, C] -> [B, C, H, W]
            try:
                features_reshaped = features.view(B, expected_H, expected_W, C).permute(0, 3, 1, 2)
                console_logger.debug(f"Reshaped features to: {features_reshaped.shape}")
            except Exception as e:
                 console_logger.error(f"Failed to reshape features from {features.shape} to spatial format: {e}", exc_info=True)
                 raise e

        else:
            # Handle unexpected number of dimensions
            raise ValueError(f"Unexpected number of dimensions in features tensor: {features.ndim}. Shape: {features.shape}")

        # --- Prepare Input for FPN ---
        # FPN expects a dictionary. Use '0' as key, assuming FPN takes it as the first input level.
        # Ensure the channel dimension (C) of the features matches what FPN expects.
        fpn_expected_channels = self.fpn.in_channels_list[0]
        feature_channels = features_reshaped.shape[1]
        if feature_channels != fpn_expected_channels:
            raise ValueError(
                f"Channel dimension mismatch for FPN input: Features have {feature_channels} channels, "
                f"but FPN expects {fpn_expected_channels} (based on encoder_hidden_dim={self.encoder_hidden_dim}). "
                f"Check model configuration."
            )

        features_for_fpn = {'0': features_reshaped}

        # --- FPN Forward Pass ---
        fpn_output = self.fpn(features_for_fpn)
        # console_logger.debug(f"FPN output keys: {fpn_output.keys()}") # Uncomment to verify keys if needed

        return fpn_output


class MaskRCNNLightning(pl.LightningModule):
    # (Implementation mostly the same, ensuring fpn_featmap_names and box_head are correct)
     def __init__(self, model_name=args.model_name, lr=args.learning_rate, weight_decay=args.weight_decay):
         super().__init__()
         self.lr = lr
         self.weight_decay = weight_decay
         self.save_hyperparameters(ignore=['model_name'])

         backbone = SamEmbeddingModelWithFPN(model_name=model_name)
         backbone_out_channels = backbone.out_channels # 256

         # CRITICAL: These names must match the keys in the dict returned by backbone.forward() (i.e., FPN output)
         # Check FPN docs/debug output if necessary. Standard torchvision FPN usually outputs these keys.
         fpn_featmap_names = ['0', '1', '2', '3'] # Assumes FPN outputs these levels

         # RPN
         anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) # Match FPN levels if possible
         aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
         anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
         rpn_head = RPNHead(backbone_out_channels, anchor_generator.num_anchors_per_location()[0])

         # RoI Heads
         box_roi_pooler = MultiScaleRoIAlign(featmap_names=fpn_featmap_names, output_size=7, sampling_ratio=2)
         mask_roi_pooler = MultiScaleRoIAlign(featmap_names=fpn_featmap_names, output_size=14, sampling_ratio=2)

         # Box Head (Simplified: Flatten -> FC -> ReLU)
         resolution = box_roi_pooler.output_size[0] # 7
         representation_size = 1024
         box_head_input_features = backbone_out_channels * resolution ** 2 # 256 * 7 * 7
         box_head_fc = torch.nn.Linear(box_head_input_features, representation_size)
         box_head = torch.nn.Sequential(torch.nn.Flatten(start_dim=1), box_head_fc, torch.nn.ReLU()) # Define the head structure
         box_predictor = FastRCNNPredictor(representation_size, len(COCO_CLASSES)) # Use len(COCO_CLASSES)

         # Mask Head
         mask_layers = (256, 256, 256, 256) # Channels in mask head convs
         mask_dilation = 1
         mask_head = MaskRCNNHeads(backbone_out_channels, mask_layers, mask_dilation)
         mask_predictor = MaskRCNNPredictor(mask_layers[-1], mask_layers[-1], len(COCO_CLASSES)) # Use len(COCO_CLASSES)

         # MaskRCNN Model
         self.model = MaskRCNN(
             backbone, num_classes=None, # Use num_classes from predictors
             rpn_anchor_generator=anchor_generator, rpn_head=rpn_head,
             box_roi_pool=box_roi_pooler, box_head=box_head, box_predictor=box_predictor,
             mask_roi_pool=mask_roi_pooler, mask_head=mask_head, mask_predictor=mask_predictor,
             # Add other MaskRCNN parameters if needed (nms thresholds, detections per img, etc.)
             min_size=int(min(args.target_size)), # Inform model about expected size range
             max_size=int(max(args.target_size)),
             # image_mean=[...], image_std=[...] # Add if normalization needed beyond ToTensor()
         )

         # SyncBatchNorm for DDP
         if torch.distributed.is_initialized() and torch.distributed.is_available():
             self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
             console_logger.info("Converted BatchNorm to SyncBatchNorm for DDP.")

     def forward(self, images, targets=None):
         # (Implementation remains the same)
         if self.training and targets is None:
             console_logger.warning("Targets are None during training step.")
         return self.model(images, targets)

     def training_step(self, batch, batch_idx):
         # (Implementation remains the same, includes NaN/inf checks)
        if batch is None or len(batch) != 2:
            console_logger.warning(f"Training step {batch_idx}: Received invalid batch format. Skipping.")
            return None
        images, targets = batch
        if images is None or targets is None:
             console_logger.warning(f"Training step {batch_idx}: Received None in batch. Skipping.")
             return None

        loss_dict = self.forward(images, targets)

        if not loss_dict or not isinstance(loss_dict, dict):
             console_logger.warning(f"Training step {batch_idx}: Model returned invalid loss_dict: {loss_dict}. Skipping loss calculation.")
             return None

        losses = 0.0
        valid_loss_keys = 0
        for key, loss in loss_dict.items():
            if key in loss_weights and torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
                weighted_loss = loss * loss_weights[key]
                losses += weighted_loss
                self.log(f'train/{key}', loss.detach(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
                valid_loss_keys += 1
            else:
                 console_logger.warning(f"Training step {batch_idx}: Invalid or missing loss '{key}': {loss}. Skipping.")

        if valid_loss_keys > 0:
             self.log('train/loss', losses.detach(), prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
             return losses
        else:
             console_logger.warning(f"Training step {batch_idx}: No valid losses found in loss_dict. Skipping optimizer step.")
             return None

     def configure_optimizers(self):
         # (Implementation remains the same)
         decay, no_decay = [], []
         for name, param in self.model.named_parameters():
             if not param.requires_grad: continue
             if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name.lower(): no_decay.append(param)
             else: decay.append(param)

         optimizer_grouped_parameters = [
             {'params': decay, 'weight_decay': self.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}
         ]
         optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

         total_steps = 10000 # Default fallback
         if hasattr(self, 'trainer') and self.trainer.datamodule and hasattr(self.trainer, 'max_epochs'):
             try:
                 len_train_loader = len(self.trainer.datamodule.train_dataloader())
                 accum = self.trainer.accumulate_grad_batches
                 epochs = self.trainer.max_epochs
                 if len_train_loader > 0 and accum > 0 and epochs > 0:
                      total_steps = (len_train_loader // accum) * epochs
                 else: raise ValueError("Invalid dataloader length, accumulation steps, or epochs for step calculation.")
             except Exception as e: console_logger.error(f"Error calculating total_steps: {e}. Using fallback {total_steps}.")
         else: console_logger.warning(f"Trainer context unavailable for step calculation. Using fallback {total_steps}.")

         console_logger.info(f"[Optimizer] AdamW | LR: {self.lr} | WD: {self.weight_decay}")
         console_logger.info(f"[Scheduler] OneCycleLR | Max LR: {self.lr} | Steps: {total_steps} | PctStart: {args.one_cycle_lr_pct}")
         scheduler = torch.optim.lr_scheduler.OneCycleLR(
             optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=args.one_cycle_lr_pct,
             final_div_factor=1e4, three_phase=args.one_cycle_lr_three_phase )
         return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


# --- Data Handling ---
def resize_and_pad_image(image: Image.Image, target_size=(1024, 1024)):
    # (Implementation remains the same, includes ndim/alpha checks and error handling)
    try:
        # Ensure input is PIL Image
        if not isinstance(image, Image.Image):
             raise TypeError(f"Expected PIL Image, got {type(image)}")

        img_array = np.array(image)
        if img_array.ndim == 2: img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[2] == 4: img_array = img_array[:, :, :3]
        elif img_array.shape[2] != 3: raise ValueError(f"Unexpected number of channels: {img_array.shape[2]}")

        h, w = img_array.shape[:2]
        if h <= 0 or w <= 0: raise ValueError(f"Invalid original image dimensions: {w}x{h}")
        target_h, target_w = target_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h <= 0 or new_w <= 0: raise ValueError(f"Invalid resized dimensions: {new_w}x{new_h}")

        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized_array = np.array(resized)

        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        top = max(0, (target_h - new_h) // 2)
        left = max(0, (target_w - new_w) // 2)
        slice_h = min(new_h, target_h - top)
        slice_w = min(new_w, target_w - left)

        if slice_h > 0 and slice_w > 0:
            padded[top:top+slice_h, left:left+slice_w, :] = resized_array[:slice_h, :slice_w, :]
        else: console_logger.warning("Zero-sized slice during padding.")

        metadata = {'original_size': (h, w), 'resized_size': (new_h, new_w), 'padding': (top, left), 'scale': scale}
        return Image.fromarray(padded), metadata

    except Exception as e:
         console_logger.error(f"Error in resize_and_pad_image: {e}", exc_info=True)
         # Return a dummy black image and default metadata on failure
         padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
         metadata = {'original_size': (0, 0), 'resized_size': (0, 0), 'padding': (0, 0), 'scale': 1.0}
         return Image.fromarray(padded), metadata

# COCO Dataset Preparation
def prepare_coco_datasets(args):
    # (Implementation remains the same)
    base_dir = Path(args.base_dir)
    train_dir = base_dir / "images" / "train2017"
    val_dir = base_dir / "images" / "val2017"
    train_ann = base_dir / args.train_ann
    val_ann = base_dir / args.val_ann

    download_coco_dataset(str(base_dir))

    for path in [train_ann, val_ann, train_dir, val_dir]:
        if not path.exists(): raise FileNotFoundError(f"Dataset path not found: {path}")
        console_logger.info(f"Verified: {path}")

    transform_tensor = T.ToTensor()
    try:
        train_dataset = CocoDetection(root=str(train_dir), annFile=str(train_ann), transform=None)
        val_dataset = CocoDetection(root=str(val_dir), annFile=str(val_ann), transform=None)
        console_logger.info(f"Loaded COCO datasets: {len(train_dataset)} train, {len(val_dataset)} val images.")
    except Exception as e:
        console_logger.error(f"Error loading COCO dataset from {base_dir}: {e}", exc_info=True)
        raise
    return train_dataset, val_dataset, transform_tensor

# COCO Data Module (Fixed DataLoader call)
class COCODataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, tensor_transform):
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
            # *** Fixed: Pass transform via lambda ***
            collate_fn=lambda b: collate_fn(b, self.tensor_transform),
            pin_memory=self.pin_memory, sampler=sampler,
            persistent_workers=self.num_workers > 0,
            drop_last=True # Drop last incomplete batch, often helps stabilize training
        )

    def val_dataloader(self):
        sampler = None
        if torch.distributed.is_initialized():
             sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            # *** Fixed: Pass transform via lambda ***
            collate_fn=lambda b: collate_fn(b, self.tensor_transform),
            pin_memory=self.pin_memory, sampler=sampler,
            persistent_workers=self.num_workers > 0
        )

# Collate Function (Accepts ToTensor transform)
def collate_fn(batch, tensor_transform):
    # (Implementation remains the same - complex logic for COCO annotations, masks, boxes)
    # Ensure error handling inside is robust
    pil_images = []
    targets_raw = []
    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            # Ensure item[0] is PIL or can be converted
            img_data, ann_data = item
            if isinstance(img_data, Image.Image):
                pil_images.append(img_data)
                targets_raw.append(ann_data)
            elif isinstance(img_data, np.ndarray):
                 try:
                      pil_images.append(Image.fromarray(img_data))
                      targets_raw.append(ann_data)
                 except Exception as e:
                      console_logger.warning(f"Failed to convert numpy array in batch to PIL: {e}. Skipping item.")
            else:
                 console_logger.warning(f"Unexpected image data type in batch: {type(img_data)}. Skipping item.")
        else:
             console_logger.warning(f"Unexpected batch item format: {type(item)}. Skipping item.")

    if not pil_images: # If all items failed
         return None, None

    processed_images = []
    all_metadata = []
    target_size_h, target_size_w = tuple(args.target_size)

    # Resize/Pad PIL images & Apply ToTensor
    for img_pil in pil_images:
        processed_pil, metadata = resize_and_pad_image(img_pil, tuple(args.target_size))
        try:
             processed_images.append(tensor_transform(processed_pil))
             all_metadata.append(metadata)
        except Exception as e:
             console_logger.error(f"ToTensor transform failed: {e}. Using zero tensor.")
             processed_images.append(torch.zeros((3, target_size_h, target_size_w)))
             all_metadata.append({'original_size': (0, 0), 'resized_size': (0, 0), 'padding': (0, 0), 'scale': 1.0})

    # Process targets using metadata
    processed_targets = []
    for idx, target_list in enumerate(targets_raw):
        metadata = all_metadata[idx]
        orig_h, orig_w = metadata['original_size']
        new_h, new_w = metadata['resized_size']
        pad_top, pad_left = metadata['padding']
        scale = metadata['scale']

        # Skip if original size is invalid (from error handling above)
        if orig_h <= 0 or orig_w <= 0:
             processed_targets.append({ # Empty target dict
                 'boxes': torch.empty((0, 4), dtype=torch.float32), 'labels': torch.empty(0, dtype=torch.int64),
                 'masks': torch.empty((0, target_size_h, target_size_w), dtype=torch.uint8),
                 'area': torch.empty(0, dtype=torch.float32), 'iscrowd': torch.empty(0, dtype=torch.uint8) })
             continue

        target_dict = {'boxes': [], 'labels': [], 'masks': [], 'area': [], 'iscrowd': []}
        valid_indices = [] # Indices of targets in original list that were valid after processing

        # Ensure target_list is iterable (usually a list of dicts)
        if not isinstance(target_list, (list, tuple)):
             console_logger.warning(f"Annotations for image {idx} is not a list/tuple ({type(target_list)}). Skipping targets for this image.")
             target_list = [] # Process as empty list

        for i, t in enumerate(target_list):
            # Validate annotation format
            if not isinstance(t, dict) or not all(k in t for k in ['bbox', 'category_id', 'segmentation']):
                 console_logger.warning(f"Skipping invalid annotation format in image {idx}, target {i}: {t}")
                 continue

            # 1. Bounding Box Processing
            x, y, w, h = t['bbox']
            if w <= 0 or h <= 0: continue # Skip boxes with zero or negative dimensions
            x_min, y_min, x_max, y_max = x, y, x + w, y + h

            scaled_x_min, scaled_y_min = x_min * scale, y_min * scale
            scaled_x_max, scaled_y_max = x_max * scale, y_max * scale
            padded_x_min = max(0, scaled_x_min + pad_left)
            padded_y_min = max(0, scaled_y_min + pad_top)
            padded_x_max = min(target_size_w, scaled_x_max + pad_left)
            padded_y_max = min(target_size_h, scaled_y_max + pad_top)

            # Check for valid box after transform
            if padded_x_max <= padded_x_min or padded_y_max <= padded_y_min: continue

            target_dict['boxes'].append([padded_x_min, padded_y_min, padded_x_max, padded_y_max])
            target_dict['labels'].append(t['category_id'])
            target_dict['area'].append(t.get('area', 0.0))
            target_dict['iscrowd'].append(t.get('iscrowd', 0))
            valid_indices.append(i) # Track valid original index

        # 2. Mask Processing (only for targets with valid boxes)
        masks_np = []
        original_valid_targets = [target_list[i] for i in valid_indices]
        for t in original_valid_targets:
            seg = t['segmentation']
            mask = None
            try:
                # Decode Mask (RLE or Polygon)
                if isinstance(seg, dict): # RLE
                    rle_input = {'counts': seg.get('counts'), 'size': seg.get('size')}
                    if not rle_input['counts'] or not rle_input['size']: raise ValueError("Invalid RLE format")
                    # Handle different RLE counts types if necessary (bytes vs list)
                    if isinstance(rle_input['counts'], list): rle = coco_mask.frPyObjects([rle_input], orig_h, orig_w)
                    else: rle = [rle_input] # Assume compressed RLE dict
                    mask = coco_mask.decode(rle)
                    if mask.ndim == 3: mask = np.sum(mask, axis=2, dtype=np.uint8).clip(0, 1)
                elif isinstance(seg, list) and seg: # Polygon
                    rles = coco_mask.frPyObjects(seg, orig_h, orig_w)
                    mask = coco_mask.decode(rles)
                    if mask.ndim == 3: mask = np.sum(mask, axis=2, dtype=np.uint8).clip(0, 1)
                else: raise ValueError(f"Unsupported segmentation format: {type(seg)}")

                # Validate decoded mask
                if mask is None or mask.ndim != 2 or mask.shape != (orig_h, orig_w):
                     console_logger.warning(f"Mask decode/validation failed. Using zero mask.")
                     mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

                # Resize and Pad Mask (NEAREST interpolation)
                mask_pil = Image.fromarray(mask * 255, mode='L')
                resized_mask_pil = mask_pil.resize((new_w, new_h), Image.Resampling.NEAREST)
                resized_mask_np = np.array(resized_mask_pil) / 255.0 # Normalize [0, 1]

                # Create padded mask
                padded_mask_np = np.zeros((target_size_h, target_size_w), dtype=np.uint8)
                slice_h = min(new_h, target_size_h - pad_top)
                slice_w = min(new_w, target_size_w - pad_left)
                if slice_h > 0 and slice_w > 0:
                     padded_mask_np[pad_top:pad_top+slice_h, pad_left:pad_left+slice_w] = (resized_mask_np[:slice_h, :slice_w] > 0.5).astype(np.uint8)

                masks_np.append(padded_mask_np)

            except Exception as e:
                 console_logger.error(f"Error processing segmentation (Area: {t.get('area', '?')} ImgIdx: {idx}): {e}", exc_info=False)
                 masks_np.append(np.zeros((target_size_h, target_size_w), dtype=np.uint8)) # Append zero mask on error

        # Finalize target dictionary
        if target_dict['boxes']:
            target_dict['boxes'] = torch.tensor(target_dict['boxes'], dtype=torch.float32)
            target_dict['labels'] = torch.tensor(target_dict['labels'], dtype=torch.int64)
            # Ensure masks_np is not empty before stacking
            if masks_np:
                 target_dict['masks'] = torch.tensor(np.stack(masks_np), dtype=torch.uint8)
            else: # Should not happen if boxes exist, but for safety
                 target_dict['masks'] = torch.empty((0, target_size_h, target_size_w), dtype=torch.uint8)
            target_dict['area'] = torch.tensor(target_dict['area'], dtype=torch.float32)
            target_dict['iscrowd'] = torch.tensor(target_dict['iscrowd'], dtype=torch.uint8)
        else: # Create empty tensors if no valid annotations survived
             target_dict = {
                 'boxes': torch.empty((0, 4), dtype=torch.float32), 'labels': torch.empty(0, dtype=torch.int64),
                 'masks': torch.empty((0, target_size_h, target_size_w), dtype=torch.uint8),
                 'area': torch.empty(0, dtype=torch.float32), 'iscrowd': torch.empty(0, dtype=torch.uint8)
             }
        processed_targets.append(target_dict)

    # Batch images
    try:
        images_batch = torch.stack(processed_images)
    except Exception as e:
        console_logger.error(f"Failed to stack images into batch: {e}")
        return None, None # Indicate failure

    return images_batch, processed_targets

# --- Inference Helper Functions ---
# Process single image file
def process_and_save_frame(model, image_path, output_path, device, target_size, score_threshold):
    # (Implementation remains the same)
    try: image = Image.open(image_path).convert("RGB")
    except Exception as e: console_logger.error(f"Err opening {image_path}: {e}"); return False
    processed_image_pil, _ = resize_and_pad_image(image, tuple(target_size))
    img_tensor = T.ToTensor()(processed_image_pil).to(device)
    images = [img_tensor]
    model.eval()
    with torch.no_grad(): predictions = model(images)
    if not predictions: console_logger.error(f"Pred failed {image_path}"); return False
    pred = predictions[0]; masks, scores, labels = pred['masks'], pred['scores'], pred['labels']
    valid_indices = scores > score_threshold
    valid_masks, valid_scores, valid_labels = masks[valid_indices], scores[valid_indices].cpu().numpy(), labels[valid_indices].cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9)); ax.imshow(processed_image_pil); ax.axis('off')
    if len(valid_masks) > 0:
        binary_masks = (valid_masks > 0.5).squeeze(1).cpu().numpy()
        legend_elements = []
        for i in range(len(binary_masks)):
            mask = binary_masks[i]; color = plt.cm.get_cmap('tab10')(i % 10)[:3]
            overlay = np.zeros((*mask.shape, 4)); overlay[mask > 0, :3] = color; overlay[mask > 0, 3] = 0.5
            ax.imshow(overlay)
            label_txt = COCO_CLASSES[valid_labels[i]] if valid_labels[i] < len(COCO_CLASSES) else f"Cls{valid_labels[i]}"
            legend_elements.append(Patch(color=color, label=f"{label_txt} ({valid_scores[i]:.2f})"))
        if legend_elements: ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    try: plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1); console_logger.debug(f"Saved: {output_path}")
    except Exception as e: console_logger.error(f"Failed save {output_path}: {e}"); plt.close(fig); return False
    plt.close(fig); return True

# Process OpenCV frame (NumPy array)
def process_cv2_frame(model, frame_bgr, output_path, device, target_size, score_threshold):
    # (Implementation remains the same)
    try: image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    except Exception as e: console_logger.error(f"Err converting frame: {e}"); return False
    processed_image_pil, _ = resize_and_pad_image(image_pil, tuple(target_size))
    img_tensor = T.ToTensor()(processed_image_pil).to(device)
    images = [img_tensor]
    model.eval()
    with torch.no_grad(): predictions = model(images)
    if not predictions: console_logger.error(f"Pred failed frame"); return False
    pred = predictions[0]; masks, scores, labels = pred['masks'], pred['scores'], pred['labels']
    valid_indices = scores > score_threshold
    valid_masks, valid_scores, valid_labels = masks[valid_indices], scores[valid_indices].cpu().numpy(), labels[valid_indices].cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9)); ax.imshow(processed_image_pil); ax.axis('off')
    if len(valid_masks) > 0:
        binary_masks = (valid_masks > 0.5).squeeze(1).cpu().numpy()
        legend_elements = []
        for i in range(len(binary_masks)):
            mask = binary_masks[i]; color = plt.cm.get_cmap('tab10')(i % 10)[:3]
            overlay = np.zeros((*mask.shape, 4)); overlay[mask > 0, :3] = color; overlay[mask > 0, 3] = 0.5
            ax.imshow(overlay)
            label_txt = COCO_CLASSES[valid_labels[i]] if valid_labels[i] < len(COCO_CLASSES) else f"Cls{valid_labels[i]}"
            legend_elements.append(Patch(color=color, label=f"{label_txt} ({valid_scores[i]:.2f})"))
        if legend_elements: ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    try: plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1); console_logger.debug(f"Saved frame: {output_path}")
    except Exception as e: console_logger.error(f"Failed save frame {output_path}: {e}"); plt.close(fig); return False
    plt.close(fig); return True


# --- Main Execution Block ---
if __name__ == "__main__":
    # Setup
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True # Set True for reproducibility, False for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    console_logger.info("Starting script...")
    console_logger.debug(f"Arguments: {vars(args)}")

    model = None
    data_module = None
    trainer = None
    best_model_path_from_training = None

    # Initialize DataModule only if training
    if args.num_epochs > 0:
        try:
            train_dataset, val_dataset, tensor_transform = prepare_coco_datasets(args)
            data_module = COCODataModule(train_dataset, val_dataset, tensor_transform)
            console_logger.info("DataModule initialized.")
        except Exception as e:
            console_logger.error(f"Dataset/DataModule setup failed: {e}. Cannot train.", exc_info=True)
            exit(1)

    # Initialize Model
    try:
        model = MaskRCNNLightning(model_name=args.model_name, lr=args.learning_rate, weight_decay=args.weight_decay)
        console_logger.info(f"Model '{args.model_name}' initialized.")
    except Exception as e:
         console_logger.error(f"Failed to initialize model: {e}", exc_info=True)
         exit(1)

    # Training Phase
    if args.num_epochs > 0 and data_module is not None:
        console_logger.info("--- Starting Training Phase ---")
        # Load weights *before* Trainer if not resuming full state
        ckpt_path_for_fit = None
        if args.resume_ckpt_path and args.resume_ckpt_path.is_file():
             ckpt_path_for_fit = str(args.resume_ckpt_path)
             console_logger.info(f"Training will resume from checkpoint: {ckpt_path_for_fit}")
        elif args.load_model_weights and args.load_model_weights.is_file():
            console_logger.info(f"Loading weights for training from: {args.load_model_weights}")
            try:
                checkpoint = torch.load(str(args.load_model_weights), map_location='cpu')
                if 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'], strict=False)
                else: model.model.load_state_dict(checkpoint, strict=False) # Raw weights
                console_logger.info("Weights loaded successfully.")
            except Exception as e: console_logger.error(f"Error loading weights from {args.load_model_weights}: {e}. Training may start from scratch/pretrained.")

        # Callbacks & Logger
        args.logs_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = args.logs_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        chkpt_cb = ModelCheckpoint(monitor='train/loss', dirpath=checkpoint_dir, filename='best-{epoch:02d}-{train/loss:.4f}', save_top_k=3, mode='min', save_last=True)
        lr_mon = LearningRateMonitor(logging_interval='step')
        callbacks = [chkpt_cb, lr_mon]
        logger = True # Default TensorBoardLogger
        try: # Try WandB
             if args.wandb_project:
                 logger = WandbLogger(project=args.wandb_project, log_model="all", save_dir=str(args.logs_dir))
                 console_logger.info(f"Using WandB logger for project: {args.wandb_project}")
        except Exception as e: console_logger.warning(f"WandB setup failed: {e}. Using default logger.")

        # Trainer
        trainer = pl.Trainer(
            max_epochs=args.num_epochs, accelerator='gpu', devices=-1,
            strategy='ddp_find_unused_parameters_true', callbacks=callbacks, logger=logger,
            accumulate_grad_batches=args.gradient_accumulation_steps, precision='bf16-mixed',
            log_every_n_steps=args.log_steps, sync_batchnorm=True,
        )
        # Log info before fit
        if trainer.is_global_zero:
             num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
             console_logger.info(f"Trainable parameters: {num_params/1e6:.2f} M")
             if hasattr(torch.cuda, 'mem_get_info') and torch.cuda.is_available():
                 free, total = torch.cuda.mem_get_info()
                 console_logger.info(f"GPU Memory (Rank 0): Free={free/1e9:.2f} GB, Total={total/1e9:.2f} GB")

        # Fit
        try:
            console_logger.info(f"Starting training...")
            trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path_for_fit)
            console_logger.info("Training finished.")
            best_model_path_from_training = chkpt_cb.best_model_path
        except Exception as e:
            console_logger.error(f"Training failed: {e}", exc_info=True)
            exit(1)
    elif args.num_epochs <= 0:
         console_logger.info("Skipping training phase (num_epochs <= 0).")
    else: # Should not happen if datamodule failed before
         console_logger.error("Cannot train because DataModule initialization failed.")
         exit(1)


    # --- Inference Phase ---
    run_inference = args.image_path or args.input_dir or args.video_path
    is_rank_zero = torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True

    if run_inference and is_rank_zero:
        console_logger.info("--- Starting Inference Phase (Rank 0) ---")

        # Determine checkpoint to load for inference
        ckpt_to_load_inf = None
        if args.load_model_weights and args.load_model_weights.is_file():
            ckpt_to_load_inf = str(args.load_model_weights)
            console_logger.info(f"Using explicitly provided weights for inference: {ckpt_to_load_inf}")
        elif best_model_path_from_training and Path(best_model_path_from_training).is_file():
            ckpt_to_load_inf = best_model_path_from_training
            console_logger.info(f"Using best checkpoint from training for inference: {ckpt_to_load_inf}")
        elif args.resume_ckpt_path and args.resume_ckpt_path.is_file():
             # Use resumed checkpoint if no other specified and no training happened to update it
             if args.num_epochs <= 0:
                  ckpt_to_load_inf = str(args.resume_ckpt_path)
                  console_logger.info(f"Using resumed checkpoint for inference (no new training): {ckpt_to_load_inf}")
             else: # Model state after fit (from resume) is already in `model`
                  console_logger.info("Using model state after resumed training for inference.")
        else: # No explicit weights, no best path found, no resume path used for inference
             console_logger.warning("No specific checkpoint specified or found for inference. Using current model state (might be randomly initialized or from pretrained backbone).")

        # Load the inference model state if needed
        inference_model = model # Start with current model
        if ckpt_to_load_inf:
            console_logger.info(f"Loading model state from: {ckpt_to_load_inf}")
            try: # Try loading full lightning checkpoint first
                inference_model = MaskRCNNLightning.load_from_checkpoint(
                    ckpt_to_load_inf, map_location='cpu' # Load on CPU first
                )
                console_logger.info("Loaded full Lightning checkpoint.")
            except Exception as e1:
                console_logger.warning(f"Failed to load as Lightning checkpoint ({e1}). Trying to load state_dict.")
                try: # Try loading state dict (might be just weights or part of lightning ckpt)
                     checkpoint = torch.load(ckpt_to_load_inf, map_location='cpu')
                     if 'state_dict' in checkpoint:
                          inference_model.load_state_dict(checkpoint['state_dict'], strict=False)
                          console_logger.info("Loaded state_dict from checkpoint.")
                     elif isinstance(checkpoint, dict): # Assume raw weights dict for model.model
                          inference_model.model.load_state_dict(checkpoint, strict=False)
                          console_logger.info("Loaded raw weights into model.")
                     else: raise ValueError("Checkpoint format not recognized.")
                except Exception as e2:
                     console_logger.error(f"Failed to load weights/state_dict from {ckpt_to_load_inf}: {e2}. Inference will use previous model state.", exc_info=True)
                     # `inference_model` remains `model` (the state before attempting load)

        # Prepare for inference run
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_model.to(device)
        inference_model.eval()
        inference_nn_model = inference_model.model # Get the underlying nn.Module
        console_logger.info(f"Inference using device: {device}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Run selected inference mode ---
        if args.video_path:
            if cv2 is None: console_logger.error("OpenCV not available, cannot process video."); exit(1)
            video_path = Path(args.video_path)
            if not video_path.is_file(): console_logger.error(f"Video file not found: {video_path}")
            else:
                console_logger.info(f"Processing video: {video_path}")
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened(): console_logger.error(f"Cannot open video: {video_path}")
                else:
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    console_logger.info(f"Video Info: ~{frame_count} frames, {fps:.2f} FPS")
                    frame_num = 0
                    with tqdm(total=frame_count if frame_count > 0 else None, desc="Video Processing") as pbar:
                        while True:
                            ret, frame = cap.read(); frame_num += 1
                            if not ret: console_logger.info("End of video or read error."); break
                            out_path = args.output_dir / f"frame_{frame_num:06d}.png"
                            process_cv2_frame(inference_nn_model, frame, str(out_path), device, args.target_size, args.frame_save_score_threshold)
                            pbar.update(1)
                    cap.release(); console_logger.info(f"Video processing finished. Output in {args.output_dir}")

        elif args.input_dir:
            input_dir = Path(args.input_dir)
            if not input_dir.is_dir(): console_logger.error(f"Input directory not found: {input_dir}")
            else:
                console_logger.info(f"Processing image directory: {input_dir}")
                img_ext = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
                files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in img_ext])
                if not files: console_logger.warning("No supported image files found.")
                else:
                    console_logger.info(f"Found {len(files)} images.")
                    for f in tqdm(files, desc="Image Processing"):
                        out_path = args.output_dir / f"{f.stem}_processed.png"
                        process_and_save_frame(inference_nn_model, str(f), str(out_path), device, args.target_size, args.frame_save_score_threshold)
                    console_logger.info(f"Directory processing finished. Output in {args.output_dir}")

        elif args.image_path:
            img_path = Path(args.image_path)
            if not img_path.is_file(): console_logger.error(f"Input image not found: {img_path}")
            else:
                console_logger.info(f"Processing single image: {img_path}")
                out_path = args.output_dir / f"{img_path.stem}_processed.png"
                success = process_and_save_frame(inference_nn_model, str(img_path), str(out_path), device, args.target_size, args.frame_save_score_threshold)
                if success: console_logger.info(f"Single image processing finished. Output: {out_path}")
                else: console_logger.error("Single image processing failed.")
        else:
             console_logger.info("No inference input specified.")

    elif run_inference and not is_rank_zero:
         console_logger.debug(f"Skipping inference on Rank {torch.distributed.get_rank()}.")

    # Final barrier for DDP processes
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    console_logger.info("Script finished.")