import logging
import os
import zipfile

import lightning.pytorch as pl
import matplotlib.pyplot as plt

# os.environ['HF_HOME'] = r"D:\.cache\hf"
import numpy as np
import requests
import torch
import torchvision.transforms as T
from fastprogress.fastprogress import master_bar, progress_bar
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib.patches import Patch
from PIL import Image
from pycocotools import mask as coco_mask
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import (
    AnchorGenerator,
    FastRCNNConvFCHead,
    FastRCNNPredictor,
)
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN,
    MaskRCNNHeads,
    MaskRCNNPredictor,
    RPNHead,
)
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

seed_everything(42)
# Set the random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True 


device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download COCO 2017 dataset to D drive
BASE_DIR = "/home/vamsik1211/Data/Assignments/Sem-2/CV/CourseProject/Instance_Segmentation Code CV Project/dataset/coco"
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


def download_file(url, dest_path):
    try:
        logger.info(f"Starting download: {url} to {dest_path}")
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
        logger.info(f"Download completed: {dest_path} ({total_size / (1024*1024):.2f} MB)")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to write to {dest_path}: {e}")
        raise


def extract_zip(zip_path, extract_to):
    try:
        logger.info(f"Extracting {zip_path} to {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        logger.info(f"Extracted and removed {zip_path}")
    except zipfile.BadZipFile as e:
        logger.error(f"Bad zip file: {zip_path} - {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to extract or remove {zip_path}: {e}")
        raise


def download_coco_dataset(base_dir=BASE_DIR):
    images_dir = os.path.join(base_dir, "images")
    annotations_dir = os.path.join(base_dir, "annotations")
    if not os.path.exists(base_dir):
        os.makedirs(images_dir)
        os.makedirs(annotations_dir)
        logger.info(f"Created directories: {images_dir}, {annotations_dir}")
    else:
        logger.info(f"Directory {base_dir} already exists. Checking contents...")
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
            logger.info(f"{key} missing or empty at {dir_path}")
            break
        else:
            logger.info(f"{key} found at {dir_path} with {len(os.listdir(dir_path))} files")
    if not all_complete:
        for key, url in COCO_URLS.items():
            dest_dir = annotations_dir if key == "annotations" else images_dir
            zip_path = os.path.join(dest_dir, f"{key}.zip")
            extracted_folder = os.path.join(dest_dir, key if key != "annotations" else "")
            if os.path.exists(extracted_folder) and os.listdir(extracted_folder):
                logger.info(f"{key} already downloaded and extracted to {extracted_folder}. Skipping...")
                continue
            download_file(url, zip_path)
            extract_zip(zip_path, dest_dir)
        logger.info(f"COCO 2017 dataset successfully downloaded and extracted to {base_dir}")
    else:
        logger.info("COCO 2017 dataset appears complete. Skipping download.")


# Model definition
class SamEmbeddingModelWithFPN(nn.Module):
    def __init__(self, model_name=None):
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



def train_model(model, train_loader, val_loader, num_epochs=10, gradient_accumulation_steps=4):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler()  # Use GradScaler for mixed precision training
    
    mb = master_bar(range(num_epochs))  # Master bar for epochs
    for epoch in mb:
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        optimizer.zero_grad()  # Reset gradients at the start of each epoch
        
        pb = progress_bar(enumerate(train_loader), parent=mb, total=len(train_loader))  # Progress bar for batches
        model.train()
        for i, (images, targets) in pb:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast(dtype=torch.float16, device_type=device):  # Use autocast for mixed precision
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses = losses / gradient_accumulation_steps  # Scale the loss for accumulation
            
            scaler.scale(losses).backward()  # Scale the loss for mixed precision
            
            # Gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Reset gradients after optimizer step
            
            running_loss += losses.item() * gradient_accumulation_steps  # Multiply back to get the actual loss
            pb.comment = f"Batch {i}, Loss: {losses.item() * gradient_accumulation_steps:.4f}"  # Update progress bar comment
        
        epoch_loss = running_loss / len(train_loader)
        mb.write(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
        # Validation loop
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for images, targets in val_loader:
        #         images = list(image.to(device) for image in images)
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
        #         with torch.amp.autocast(device_type=device, dtype=torch.float16):
        #             loss_dict = model(images, targets)
        #             print(loss_dict)
        #             losses = sum(loss for loss in loss_dict.values())
                
        #         val_loss += losses.item()
        
        # val_loss /= len(val_loader)
        # mb.write(f"Validation Loss: {val_loss:.4f}")
        # model.train()


def train_model_with_onecyclelr(
    model, train_loader, val_loader, num_epochs=10, gradient_accumulation_steps=4
):
    """
    Train the model using OneCycleLR scheduler and mixed precision training.
    """
    # Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    # Define OneCycleLR scheduler
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.005,
        total_steps=total_steps,
        pct_start=0.3,  # Percentage of the cycle spent increasing the learning rate
        # anneal_strategy="linear",
        final_div_factor=1e5,  # Determines the minimum learning rate
    )

    # Use GradScaler for mixed precision training
    scaler = torch.amp.GradScaler()

    min_loss = torch.tensor(float("inf"))

    # Progress bar for epochs
    mb = master_bar(range(num_epochs))
    for epoch in mb:
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        # Progress bar for batches
        pb = progress_bar(enumerate(train_loader), parent=mb, total=len(train_loader))
        model.train()
        for i, (images, targets) in pb:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Mixed precision training with autocast
            with torch.amp.autocast(dtype=torch.float16, device_type=device):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses = losses / gradient_accumulation_steps  # Scale the loss for accumulation

            # Backpropagation with scaled loss
            scaler.scale(losses).backward()

            # Gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)  # Update weights
                scaler.update()  # Update scaler
                optimizer.zero_grad()  # Reset gradients after optimizer step
                scheduler.step()  # Update learning rate

            # Track running loss
            running_loss += losses.item() * gradient_accumulation_steps  # Multiply back to get the actual loss
            pb.comment = f"Batch {i}, Loss: {losses.item() * gradient_accumulation_steps:.4f}, Current LR: {scheduler.get_last_lr()}"  # Update progress bar comment
        # Epoch loss
        epoch_loss = running_loss / len(train_loader)
        mb.write(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), f"segmentation_model_loss_{epoch_loss:.4f}.pth")
            mb.write(f"Model saved with loss: {min_loss:.4f}")
        else:
            mb.write(f"Model not saved, current loss: {epoch_loss:.4f}, previous min loss: {min_loss:.4f}")

        # Validation loop
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for images, targets in val_loader:
        #         images = list(image.to(device) for image in images)
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #         with torch.cuda.amp.autocast(dtype=torch.float16, device_type=device):
        #             loss_dict = model(images, targets)
        #             losses = sum(loss for loss in loss_dict.values())

        #         val_loss += losses.item()

        # val_loss /= len(val_loader)
        # mb.write(f"Validation Loss: {val_loss:.4f}")
        model.train()


def resize_and_pad_image(image, target_size=(1024, 1024)):
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

def generate_masks(image_path):
    image = Image.open(image_path).convert("RGB")
    processed_image = resize_and_pad_image(image)
    img_array = np.array(processed_image).transpose(2, 0, 1)
    img_tensor = torch.tensor(img_array, dtype=torch.float32).to(device) / 255.0
    images = [img_tensor]
    with torch.no_grad():
        predictions = seg_model(images)
    masks = predictions[0]['masks']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']
    score_threshold = 0.1
    valid_indices = scores > score_threshold
    valid_masks = masks[valid_indices]
    valid_scores = scores[valid_indices]
    valid_labels = labels[valid_indices]
    binary_masks = (valid_masks > 0.1).squeeze(1).cpu().numpy()
    return binary_masks, image, processed_image, {'scores': valid_scores, 'labels': valid_labels}

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    processed_targets = []
    target_size = (1024, 1024)
    
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
            
            # Filter out invalid bounding boxes
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
                    if isinstance(seg, dict):  # RLE format
                        # Handle uncompressed RLE (list of integers)
                        if isinstance(seg['counts'], list):
                            rle = {'counts': seg['counts'], 'size': [orig_h, orig_w]}
                            mask = coco_mask.decode(coco_mask.frPyObjects(rle, orig_h, orig_w))
                        else:  # Compressed RLE (bytes)
                            mask = coco_mask.decode(seg)
                    elif isinstance(seg, list):  # Polygon format
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
                        logger.warning(f"Invalid mask dim {mask.ndim}, using zero mask")
                        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    
                    if mask.shape != (orig_h, orig_w):
                        logger.warning(f"Mask shape {mask.shape} != ({orig_h}, {orig_w}), using zero mask")
                        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    
                    mask_img = Image.fromarray(mask, mode='L').resize(target_size, Image.Resampling.NEAREST)
                    masks.append(np.array(mask_img, dtype=np.uint8))
                except Exception as e:
                    logger.error(f"Error processing segmentation: {e}, seg type: {type(seg)}, using zero mask")
                    masks.append(np.zeros(target_size, dtype=np.uint8))
            target_dict['masks'] = torch.tensor(np.stack(masks), dtype=torch.uint8)
            
            processed_targets.append(target_dict)
    return images, processed_targets

class ResizeAndPad:
    def __init__(self, target_size=(1024, 1024)):
        self.target_size = target_size
    
    def __call__(self, image):
        return resize_and_pad_image(image, self.target_size)

if __name__ == "__main__":
    download_coco_dataset()
    
    train_dir = r"dataset/coco/images/train2017"
    val_dir = r"dataset/coco/images/val2017"
    train_ann = r"dataset/coco/annotations/annotations/instances_train2017.json"
    val_ann = r"dataset/coco/annotations/annotations/instances_val2017.json"
    for path in [train_dir, val_dir, train_ann, val_ann]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset file/directory not found: {path}")
        logger.info(f"Verified existence of {path}")

    model_name = "Zigeng/SlimSAM-uniform-77"
    backbone = SamEmbeddingModelWithFPN(model_name=model_name).eval()
    freeze_model(backbone)
    
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    mask_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
    box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d)
    mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d)
    box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=91)
    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=91)
    
    global seg_model
    seg_model = MaskRCNN(
        backbone=backbone,
        num_classes=None,
        min_size=1024,
        max_size=1024,
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
    ).to(device)

    # seg_model = torch.compile(seg_model, backend="cudagraphs")

    transform = T.Compose([ResizeAndPad(target_size=(1024, 1024)), T.ToTensor()])
    train_dataset = CocoDetection(root=train_dir, annFile=train_ann, transform=transform)
    val_dataset = CocoDetection(root=val_dir, annFile=val_ann, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=16, collate_fn=collate_fn, pin_memory=True)
    
    # try:
    #     import psutil
    #     disk = psutil.disk_usage("/home/vamsik1211/Data/Assignments/Sem-2/CV/")
    #     free_space_gb = disk.free / (1024 ** 3)
    #     logger.info(f"Free space on D drive: {free_space_gb:.2f} GB")
    # except ImportError:
    #     logger.info("Install 'psutil' for disk space checking: pip install psutil")
    # seg_model.load_state_dict(torch.load("segmentation_model.pth", map_location=device))
    # train_model(seg_model, train_loader, val_loader, num_epochs=10, gradient_accumulation_steps=128)
    train_model_with_onecyclelr(seg_model, train_loader, val_loader, num_epochs=10, gradient_accumulation_steps=128)
    
    image_path = "Picture3.png"
    masks, original_image, processed_image, predictions = generate_masks(image_path)
    
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
