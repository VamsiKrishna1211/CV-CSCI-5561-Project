import logging
import os
import zipfile
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import (
    AnchorGenerator,
    FastRCNNConvFCHead,
    FastRCNNPredictor,
)
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNHeads,
    MaskRCNNPredictor,
    RPNHead,
)
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from transformers import AutoModel
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
import requests

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Instance Segmentation with Mask R-CNN and SAM backbone using PyTorch Lightning")
    parser.add_argument('--base-dir', type=str, default="/home/vamsik1211/Data/Assignments/Sem-2/CV/CourseProject/Instance_Segmentation Code CV Project/dataset/coco",
                        help="Base directory for COCO dataset")
    parser.add_argument('--train-ann', type=str, default="annotations/instances_train2017.json",
                        help="Path to training annotations relative to base-dir")
    parser.add_argument('--val-ann', type=str, default="annotations/instances_val2017.json",
                        help="Path to validation annotations relative to base-dir")
    parser.add_argument('--model-name', type=str, default="Zigeng/SlimSAM-uniform-77",
                        help="Hugging Face model name for SAM backbone")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="Batch size for data loaders")
    parser.add_argument('--num-workers', type=int, default=16,
                        help="Number of workers for data loaders")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=128,
                        help="Number of batches to accumulate gradients over")
    parser.add_argument('--learning-rate', type=float, default=0.005,
                        help="Initial learning rate for SGD optimizer")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="Momentum for SGD optimizer")
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help="Weight decay for SGD optimizer")
    parser.add_argument('--target-size', type=int, nargs=2, default=(1024, 1024),
                        help="Target size for image resizing (height width)")
    parser.add_argument('--image-path', type=str, default="Picture3.png",
                        help="Path to the image for mask generation")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu); defaults to cuda if available")
    return parser.parse_args()

args = parse_args()

# Set random seed
pl.seed_everything(args.seed, workers=True)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set device
device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"

# Set up console logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_logger = logging.getLogger(__name__)

# COCO dataset URLs and classes
COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Utility functions for dataset download
def download_file(url, dest_path):
    try:
        console_logger.info(f"Starting download: {url} to {dest_path}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        console_logger.info(f"Download completed: {dest_path} ({total_size / (1024*1024):.2f} MB)")
    except requests.exceptions.RequestException as e:
        console_logger.error(f"Failed to download {url}: {e}")
        raise
    except IOError as e:
        console_logger.error(f"Failed to write to {dest_path}: {e}")
        raise

def extract_zip(zip_path, extract_to):
    try:
        console_logger.info(f"Extracting {zip_path} to {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        console_logger.info(f"Extracted and removed {zip_path}")
    except zipfile.BadZipFile as e:
        console_logger.error(f"Bad zip file: {zip_path} - {e}")
        raise
    except OSError as e:
        console_logger.error(f"Failed to extract or remove {zip_path}: {e}")
        raise

def download_coco_dataset(base_dir):
    images_dir = os.path.join(base_dir, "images")
    annotations_dir = os.path.join(base_dir, "annotations")
    if not os.path.exists(base_dir):
        os.makedirs(images_dir)
        os.makedirs(annotations_dir)
        console_logger.info(f"Created directories: {images_dir}, {annotations_dir}")
    else:
        console_logger.info(f"Directory {base_dir} already exists. Checking contents...")
    required_dirs = {
        "train2017": os.path.join(images_dir, "train2017"),
        "val2017": os.path.join(images_dir, "val2017"),
        "test2017": os.path.join(images_dir, "test2017"),
        "annotations": annotations_dir
    }
    all_complete = True
    for key, dir_path in required_dirs.items():
        if not os.path.exists(dir_path) or not os.listdir(dir_path):
            all_complete = False
            console_logger.info(f"{key} missing or empty at {dir_path}")
            break
        else:
            console_logger.info(f"{key} found at {dir_path} with {len(os.listdir(dir_path))} files")
    if not all_complete:
        for key, url in COCO_URLS.items():
            dest_dir = annotations_dir if key == "annotations" else images_dir
            zip_path = os.path.join(dest_dir, f"{key}.zip")
            extracted_folder = os.path.join(dest_dir, key if key != "annotations" else "")
            if os.path.exists(extracted_folder) and os.listdir(extracted_folder):
                console_logger.info(f"{key} already downloaded and extracted to {extracted_folder}. Skipping...")
                continue
            download_file(url, zip_path)
            extract_zip(zip_path, dest_dir)
        console_logger.info(f"COCO 2017 dataset successfully downloaded and extracted to {base_dir}")
    else:
        console_logger.info("COCO 2017 dataset appears complete. Skipping download.")

# Model backbone
class SamEmbeddingModelWithFPN(torch.nn.Module):
    def __init__(self, model_name=args.model_name):
        super().__init__()
        model = AutoModel.from_pretrained(model_name)
        self.model = model.vision_encoder
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256],
            out_channels=256,
            extra_blocks=None
        )

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs["pixel_values"]
        output = self.model(inputs)["last_hidden_state"]
        features = {"0": output}
        fpn_output = self.fpn(features)
        return fpn_output

def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False

# PyTorch Lightning Module for Mask R-CNN
class MaskRCNNLightning(pl.LightningModule):
    def __init__(self, model_name=args.model_name, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize backbone
        backbone = SamEmbeddingModelWithFPN(model_name=model_name).eval()
        freeze_model(backbone)
        # Initialize Mask R-CNN components
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        mask_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
        box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=torch.nn.BatchNorm2d)
        mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=torch.nn.BatchNorm2d)
        box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=91)
        rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=91)
        # Initialize Mask R-CNN
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
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses, prog_bar=True, sync_dist=True)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        # Calculate total steps accounting for gradient accumulation
        accumulate_grad_batches = self.trainer.accumulate_grad_batches
        total_steps = (len(self.trainer.datamodule.train_dataloader()) // accumulate_grad_batches) * self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.3,
            final_div_factor=1e5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

# PyTorch Lightning DataModule for COCO
class COCODataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.base_dir = args.base_dir
        self.train_dir = os.path.join(self.base_dir, "images/train2017")
        self.val_dir = os.path.join(self.base_dir, "images/val2017")
        self.train_ann = os.path.join(self.base_dir, args.train_ann)
        self.val_ann = os.path.join(self.base_dir, args.val_ann)
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.target_size = tuple(args.target_size)

    def setup(self, stage=None):
        download_coco_dataset(self.base_dir)
        for path in [self.train_ann, self.val_ann]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required dataset file/directory not found: {path}")
            console_logger.info(f"Verified existence of {path}")
        transform = T.Compose([ResizeAndPad(target_size=self.target_size), T.ToTensor()])
        self.train_dataset = CocoDetection(root=self.train_dir, annFile=self.train_ann, transform=transform)
        self.val_dataset = CocoDetection(root=self.val_dir, annFile=self.val_ann, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return None  # Validation disabled as in original code

# Data preprocessing
def resize_and_pad_image(image, target_size=args.target_size):
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = Image.fromarray(img_array).resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_array = np.array(resized)
    padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    padded[top:top+new_h, left:left+new_w] = resized_array
    return Image.fromarray(padded)

class ResizeAndPad:
    def __init__(self, target_size=tuple(args.target_size)):
        self.target_size = target_size
    
    def __call__(self, image):
        return resize_and_pad_image(image, self.target_size)

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    processed_targets = []
    target_size = tuple(args.target_size)
    
    for idx, target_list in enumerate(targets):
        if not target_list:
            processed_targets.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty(0, dtype=torch.int64),
                'masks': torch.empty((0, *target_size), dtype=torch.uint8),
                'area': torch.empty(0, dtype=torch.float32),
                'iscrowd': torch.empty(0, dtype=torch.uint8)
            })
        else:
            orig_h, orig_w = batch[idx][0].shape[-2], batch[idx][0].shape[-1]
            scale_h, scale_w = target_size[0] / orig_h, target_size[1] / orig_w
            
            target_dict = {
                'boxes': torch.tensor([t['bbox'] for t in target_list], dtype=torch.float32),
                'labels': torch.tensor([t['category_id'] for t in target_list], dtype=torch.int64),
                'area': torch.tensor([t['area'] for t in target_list], dtype=torch.float32),
                'iscrowd': torch.tensor([t['iscrowd'] for t in target_list], dtype=torch.uint8)
            }
            boxes = target_dict['boxes']
            target_dict['boxes'] = torch.stack([
                boxes[:, 0] * scale_w, boxes[:, 1] * scale_h,
                (boxes[:, 0] + boxes[:, 2]) * scale_w, (boxes[:, 1] + boxes[:, 3]) * scale_h
            ], dim=1)
            
            valid_boxes = (target_dict['boxes'][:, 2] > target_dict['boxes'][:, 0]) & \
                          (target_dict['boxes'][:, 3] > target_dict['boxes'][:, 1])
            target_dict['boxes'] = target_dict['boxes'][valid_boxes]
            target_dict['labels'] = target_dict['labels'][valid_boxes]
            target_dict['area'] = target_dict['area'][valid_boxes]
            target_dict['iscrowd'] = target_dict['iscrowd'][valid_boxes]
            
            masks = []
            for t, valid in zip(target_list, valid_boxes):
                if not valid:
                    continue
                seg = t['segmentation']
                try:
                    if isinstance(seg, dict):
                        if isinstance(seg['counts'], list):
                            rle = {'counts': seg['counts'], 'size': [orig_h, orig_w]}
                            mask = coco_mask.decode(coco_mask.frPyObjects(rle, orig_h, orig_w))
                        else:
                            mask = coco_mask.decode(seg)
                    elif isinstance(seg, list):
                        rle = coco_mask.frPyObjects(seg, orig_h, orig_w)
                        if isinstance(rle, list):
                            mask_stack = [coco_mask.decode(r) for r in rle]
                            mask = np.any(mask_stack, axis=0).astype(np.uint8)
                        else:
                            mask = coco_mask.decode(rle)
                    else:
                        raise ValueError(f"Unexpected segmentation format: {type(seg)}")
                    
                    if mask.ndim == 3:
                        mask = np.any(mask, axis=2).astype(np.uint8)
                    elif mask.ndim != 2:
                        console_logger.warning(f"Invalid mask dim {mask.ndim}, using zero mask")
                        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    
                    if mask.shape != (orig_h, orig_w):
                        console_logger.warning(f"Mask shape {mask.shape} != ({orig_h}, {orig_w}), using zero mask")
                        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    
                    mask_img = Image.fromarray(mask, mode='L').resize(target_size, Image.Resampling.NEAREST)
                    masks.append(np.array(mask_img, dtype=np.uint8))
                except Exception as e:
                    console_logger.error(f"Error processing segmentation: {e}, seg type: {type(seg)}, using zero mask")
                    masks.append(np.zeros(target_size, dtype=np.uint8))
            target_dict['masks'] = torch.tensor(np.stack(masks), dtype=torch.uint8)
            
            processed_targets.append(target_dict)
    return images, processed_targets

# Mask generation and visualization
def generate_masks(model, image_path=args.image_path):
    image = Image.open(image_path).convert("RGB")
    processed_image = resize_and_pad_image(image)
    img_array = np.array(processed_image).transpose(2, 0, 1)
    img_tensor = torch.tensor(img_array, dtype=torch.float32).to(device) / 255.0
    images = [img_tensor]
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    masks = predictions[0]['masks']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']
    score_threshold = 0.1  # Hardcoded as in original code
    valid_indices = scores > score_threshold
    valid_masks = masks[valid_indices]
    valid_scores = scores[valid_indices]
    valid_labels = labels[valid_indices]
    binary_masks = (valid_masks > 0.1).squeeze(1).cpu().numpy()  # Hardcoded threshold as in original code
    return binary_masks, image, processed_image, {'scores': valid_scores, 'labels': valid_labels}

if __name__ == "__main__":
    # Initialize data module
    data_module = COCODataModule()
    
    # Initialize model
    model = MaskRCNNLightning()
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        filename='maskrcnn-{epoch:02d}-{train_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(
        monitor='train_loss',
        patience=3,
        mode='min',
        min_delta=0.0
    )
    
    # Set up W&B logger
    wandb_logger = WandbLogger(project="maskrcnn", log_model="all")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=wandb_logger,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision='16-mixed',
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Generate and visualize masks
    masks, original_image, processed_image, predictions = generate_masks(model)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.subplot(2, 2, 2)
    plt.imshow(processed_image)
    plt.title("Processed Image (1024x1024)")
    plt.subplot(2, 2, 3)
    plt.imshow(processed_image)
    colors = []
    for i in range(len(masks)):
        mask = masks[i]
        color = np.random.rand(3)
        colors.append(color)
        masked_image = np.zeros_like(processed_image)
        masked_image[mask > 0] = color * 255
        plt.imshow(masked_image, alpha=0.5)
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    legend_elements = [Patch(color=colors[i], label=f"{COCO_CLASSES[labels[i]]} (Score: {scores[i]:.3f})") for i in range(len(masks))]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title("Masks Overlay")
    if len(masks) > 0:
        plt.subplot(2, 2, 4)
        plt.imshow(masks[0], cmap='gray')
        plt.title(f"First Mask ({COCO_CLASSES[labels[0]]}, Score: {scores[0]:.3f})")
    plt.tight_layout()
    plt.show()