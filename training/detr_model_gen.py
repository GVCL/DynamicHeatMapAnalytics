import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torchvision
from scipy.optimize import linear_sum_assignment
import json
import torch
from torchvision.ops.boxes import box_area
import matplotlib.pyplot as plt

def plot_loss_curves(epochs_to_plot=None):
    if epochs_to_plot is None:
        # If no specific epochs provided, plot all available checkpoints
        epochs_to_plot = []
        checkpoint_dir = 'detr_checkpoints'
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('checkpoint_epoch_'):
                epoch = int(filename.split('_')[-1].split('.')[0])
                epochs_to_plot.append(epoch)
        epochs_to_plot.sort()
    
    epochs = []
    ce_losses = []
    bbox_losses = []
    giou_losses = []
    
    try:
        for epoch in epochs_to_plot:
            checkpoint_path = os.path.join('detr_checkpoints', f'checkpoint_epoch_{epoch}.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                if 'bbox_loss' in checkpoint and 'giou_loss' in checkpoint and 'ce_loss' in checkpoint:
                    epochs.append(epoch)
                    ce_losses.append(checkpoint['ce_loss'])
                    bbox_losses.append(checkpoint['bbox_loss'])
                    giou_losses.append(checkpoint['giou_loss'])
                else:
                    print(f"Warning: Loss data not found in checkpoint for epoch {epoch}")
        
        if len(epochs) > 0:
            plt.figure(figsize=(15, 8))
            
            plt.subplot(1, 3, 1)
            plt.plot(epochs, ce_losses, 'g-o', linewidth=2, markersize=8)
            plt.xlabel('Epochs')
            plt.ylabel('CE Loss')
            plt.title('Cross Entropy Loss vs Epochs')
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(epochs, bbox_losses, 'b-o', linewidth=2, markersize=8)
            plt.xlabel('Epochs')
            plt.ylabel('BBox Loss')
            plt.title('BBox Loss vs Epochs')
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.plot(epochs, giou_losses, 'r-o', linewidth=2, markersize=8)
            plt.xlabel('Epochs')
            plt.ylabel('IoU Loss')
            plt.title('IoU Loss vs Epochs')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('loss_plots.png')
            plt.close()
            print(f"Loss plots saved as loss_plots.png")
            
            # Also create a combined plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, ce_losses, 'g-o', label='CE Loss', linewidth=2, markersize=5)
            plt.plot(epochs, bbox_losses, 'b-o', label='BBox Loss', linewidth=2, markersize=5)
            plt.plot(epochs, giou_losses, 'r-o', label='IoU Loss', linewidth=2, markersize=5)
            plt.xlabel('Epochs')
            plt.ylabel('Loss Value')
            plt.title('Training Losses vs Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig('combined_loss_plot.png')
            plt.close()
            print(f"Combined plot saved as combined_loss_plot.png")
        else:
            print("No valid checkpoint data found for plotting")
            
    except Exception as e:
        print(f"Error during plotting: {str(e)}")

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    
    The boxes should be in [x0, y0, x1, y1] format
    
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class HotspotDataset(Dataset):
    def __init__(self, annotation_file, image_dir):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        
        # Filter out images without annotations
        self.ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:  # Only keep images with annotations
                self.ids.append(img_id)
                
        self.categories = self.get_categories()
        
        # Define transforms
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded dataset with {len(self.ids)} valid images (with annotations)")
    
    def get_categories(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        categories = {cat['id']: idx for idx, cat in enumerate(cats)}
        print(f"Found {len(categories)} categories")
        return categories
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
    
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Get image dimensions
        width, height = image.size
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
    
        # Prepare boxes and labels
        boxes = []
        labels = []
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            # Ensure bbox dimensions are positive
            if w <= 0 or h <= 0:
                continue
                
            boxes.append([x, y, x + w, y + h])
            labels.append(self.categories[ann['category_id']])
        
        # After normalization:
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Normalize coordinates to [0,1]
            boxes[:, 0] /= width  # x1
            boxes[:, 1] /= height  # y1
            boxes[:, 2] /= width  # x2
            boxes[:, 3] /= height  # y2
            
            # Ensure coordinates are within [0,1] range
            boxes = torch.clamp(boxes, min=0.0, max=1.0)
            
            # Convert from xyxy to cxcywh format
            boxes = box_xyxy_to_cxcywh(boxes)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
    
        labels = torch.as_tensor(labels, dtype=torch.int64)
    
        # Apply transforms to image
        image = self.transforms(image)
    
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.as_tensor([height, width]),  # Store original size
            'size': torch.as_tensor([image.shape[1], image.shape[2]])  # Post-transform size
        }
    
        return image, target
def collate_fn(batch):
    # No need to filter since we already filtered in the dataset
    return tuple(zip(*batch))

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # L1 loss - already in normalized coordinates
        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction='none')
        
        # Add GIOU loss for better convergence
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou_loss = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        
        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': giou_loss.sum() / num_boxes
        }
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # Already normalized [0,1]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # Ensure these are normalized

        # Classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # L1 cost between normalized boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # GIoU cost - first convert boxes to the right format
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox +
            self.cost_class * cost_class +
            self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_model_and_criterion(num_classes, num_queries=100):
    # Build model
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.class_embed = nn.Linear(256, num_classes + 1)
    model.num_queries = num_queries
    
    # Initialize matcher with balanced costs
    matcher = HungarianMatcher(cost_class=1, cost_bbox=2, cost_giou=5)
    
    # Define weight dictionary with reasonable values
    weight_dict = {'loss_ce': 1, 'loss_bbox': 2, 'loss_giou': 5}
    
    # Initialize criterion with proper end-of-sequence coefficient
    criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef=0.1)
    
    return model, criterion

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    loss_ce_total = 0
    loss_bbox_total = 0
    loss_giou_total = 0
    valid_batches = 0
    
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(progress_bar):
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Track individual loss components
        loss_ce_total += loss_dict['loss_ce'].item() * weight_dict['loss_ce']
        loss_bbox_total += loss_dict['loss_bbox'].item() * weight_dict['loss_bbox']
        loss_giou_total += loss_dict['loss_giou'].item() * weight_dict['loss_giou']
        
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
        total_loss += losses.item()
        valid_batches += 1
        
        # Show detailed loss breakdown
        progress_bar.set_postfix({
            'loss': losses.item(),
            'ce_loss': loss_dict['loss_ce'].item(),
            'bbox_loss': loss_dict['loss_bbox'].item(),
            'giou_loss': loss_dict['loss_giou'].item()
        })
        
        # Early debugging - print extreme values
        if losses.item() > 1000:
            print(f"Warning: Very high loss at batch {batch_idx}")
            print(f"  CE loss: {loss_dict['loss_ce'].item()}")
            print(f"  Bbox loss: {loss_dict['loss_bbox'].item()}")
            print(f"  IoU loss: {loss_dict['loss_giou'].item()}")
            print(f"  Number of targets: {[len(t['boxes']) for t in targets]}")
    
    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_ce_loss = loss_ce_total / valid_batches if valid_batches > 0 else float('inf')
    avg_bbox_loss = loss_bbox_total / valid_batches if valid_batches > 0 else float('inf')
    avg_giou_loss = loss_giou_total / valid_batches if valid_batches > 0 else float('inf')
    
    print(f"Epoch {epoch} stats:")
    print(f"  Average total loss: {avg_loss:.4f}")
    print(f"  Average CE loss: {avg_ce_loss:.4f}")
    print(f"  Average bbox loss: {avg_bbox_loss:.4f}")
    print(f"  Average IoU loss: {avg_giou_loss:.4f}")
    
    return avg_loss, avg_ce_loss, avg_bbox_loss, avg_giou_loss  # Return CE loss as well
@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_pred = 0
    
    for images, targets in tqdm(data_loader, desc="Validating"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Calculate accuracy metrics
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        for i, (logits, boxes, target) in enumerate(zip(pred_logits, pred_boxes, targets)):
            # Get predicted classes
            probas = logits.softmax(-1)
            keep = probas[:, -1] < 0.8  # Filter out "no object" predictions
            
            if keep.sum() == 0:
                continue
                
            # Match predictions to targets
            matched_indices = criterion.matcher(
                {"pred_logits": logits[keep].unsqueeze(0), 
                 "pred_boxes": boxes[keep].unsqueeze(0)}, 
                [target]
            )[0]
            
            if len(matched_indices[0]) > 0:
                total_correct += len(matched_indices[0])
            total_pred += keep.sum().item()
        
        total_loss += losses.item()
    
    avg_loss = total_loss / len(data_loader)
    precision = total_correct / total_pred if total_pred > 0 else 0
    
    print(f"Validation results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Correct predictions: {total_correct}/{total_pred}")
    
    return avg_loss

def main():
    # Dataset paths
    annotation_paths = [
        '../data/video1/annotations.json',
        '../data/video2/annotations.json',
        '../data/video3/annotations.json',
        '../data/video4/annotations.json'
    ]   
    image_dirs = [
        '../data/video1/frames',
        '../data/video2/frames',
        '../data/video3/frames',
        '../data/video4/frames'
    ]

    # Create datasets and print statistics for each dataset
    datasets = []
    total_images = 0
    for ann_path, img_dir in zip(annotation_paths, image_dirs):
        dataset = HotspotDataset(ann_path, img_dir)
        datasets.append(dataset)
        total_images += len(dataset)
    
    print(f"Combined dataset has {total_images} valid images")
    
    # Create combined dataset
    combined_dataset = ConcatDataset(datasets)
    
    # Create dataloader with smaller batch size to start
    data_loader = DataLoader(
        combined_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get number of classes from first dataset
    num_classes = len(datasets[0].categories)
    print(f"Training with {num_classes} classes")
    
    # Initialize model, criterion and move to device
    model, criterion = build_model_and_criterion(num_classes)
    model.to(device)
    criterion.to(device)

    # Initialize optimizer with learning rate warmup and lower initial learning rate
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-6}
    ], weight_decay=1e-4)
    
    # Create learning rate scheduler with warmup
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(data_loader) - 1)
    
    def lambda_lr(current_iter):
        if current_iter < warmup_iters:
            return warmup_factor + (1 - warmup_factor) * current_iter / warmup_iters
        return 0.1 ** (current_iter // (len(data_loader) * 30))
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

    # Create output directory
    output_dir = "detr_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    # Save categories mapping
    categories_path = os.path.join(output_dir, 'categories.json')
    with open(categories_path, 'w') as f:
        json.dump(datasets[0].categories, f)
    
    # Save inverse mapping for easier interpretation
    inverse_categories = {v: k for k, v in datasets[0].categories.items()}
    inverse_categories_path = os.path.join(output_dir, 'inverse_categories.json')
    with open(inverse_categories_path, 'w') as f:
        json.dump(inverse_categories, f)

    # Training loop
    num_epochs = 100
    best_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss, epoch_ce_loss, epoch_bbox_loss, epoch_giou_loss = train_one_epoch(model, criterion, data_loader, 
                                                 optimizer, device, epoch)
        # Print learning rate
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        lr_scheduler.step()
        
        # Save checkpoint if it's the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'num_classes': num_classes,
                'loss': epoch_loss,
                'ce_loss': epoch_ce_loss,  # Add this line
                'bbox_loss': epoch_bbox_loss,
                'giou_loss': epoch_giou_loss
            }
            torch.save(
                checkpoint,
                os.path.join(output_dir, 'best_model.pth')
            )
            print(f"Saved best model with loss {epoch_loss:.4f}")
        
        # Save regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'num_classes': num_classes,
                'loss': epoch_loss,
                'ce_loss': epoch_ce_loss,  # Add this line
                'bbox_loss': epoch_bbox_loss,
                'giou_loss': epoch_giou_loss
            }
            torch.save(
                checkpoint,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        print(f'Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Best loss so far: {best_loss:.4f}')

    print("Training completed!")
    plot_bbox_loss()

if __name__ == "__main__":
    main()
