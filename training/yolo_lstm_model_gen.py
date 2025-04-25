import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ultralytics import YOLO
import cv2
import json
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotspotDataset(Dataset):
    def __init__(self, annotation_paths, image_dirs, sequence_length=5):
        self.sequence_length = sequence_length
        self.sequences = []
        self.max_bbox_dim = 640  # Maximum dimension for normalization
        
        for ann_path, img_dir in zip(annotation_paths, image_dirs):
            try:
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                
                # Create image ID to filename mapping
                id_to_filename = {img['id']: img['file_name'] for img in data['images']}
                
                # Group annotations by image id
                annotations_by_frame = {}
                for ann in data['annotations']:
                    img_id = ann['image_id']
                    if img_id not in annotations_by_frame:
                        annotations_by_frame[img_id] = []
                    annotations_by_frame[img_id].append(ann['bbox'])
                
                # Create sequences
                image_ids = sorted(id_to_filename.keys())
                for i in range(len(image_ids) - sequence_length + 1):
                    seq_img_paths = []
                    seq_bboxes = []
                    
                    for j in range(sequence_length):
                        img_id = image_ids[i + j]
                        img_path = os.path.join(img_dir, id_to_filename[img_id])
                        
                        if not os.path.exists(img_path):
                            logger.warning(f"Image not found: {img_path}")
                            break
                            
                        seq_img_paths.append(img_path)
                        
                        # Get bbox or use zeros if no annotation
                        if img_id in annotations_by_frame:
                            # Use the first bbox if multiple exist
                            bbox = annotations_by_frame[img_id][0]
                            # Normalize bbox
                            bbox = [
                                bbox[0] / self.max_bbox_dim,
                                bbox[1] / self.max_bbox_dim,
                                bbox[2] / self.max_bbox_dim,
                                bbox[3] / self.max_bbox_dim
                            ]
                        else:
                            bbox = [0.0, 0.0, 0.0, 0.0]
                        
                        seq_bboxes.append(bbox)
                    
                    if len(seq_img_paths) == sequence_length:
                        self.sequences.append((seq_img_paths, seq_bboxes))
                
            except Exception as e:
                logger.error(f"Error processing {ann_path}: {str(e)}")
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)  # Ensure memory is contiguous
        img = torch.from_numpy(img).float()
        img = img / 255.0  # Normalize to [0, 1]
        return img
    
    def __getitem__(self, idx):
        img_paths, bboxes = self.sequences[idx]
        
        # Load and preprocess images
        try:
            images = [self.preprocess_image(path) for path in img_paths]
            image_sequence = torch.stack(images)
            bbox_sequence = torch.tensor(bboxes, dtype=torch.float32)
            return image_sequence, bbox_sequence
            
        except Exception as e:
            logger.error(f"Error loading sequence {idx}: {str(e)}")
            # Return a zero sequence if there's an error
            return (torch.zeros((self.sequence_length, 3, 640, 640)), 
                   torch.zeros((self.sequence_length, 4)))

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 4),
            nn.Sigmoid()  # Normalize outputs to [0, 1]
        )
    
    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        lstm_out, hidden = self.lstm(x, hidden)
        predictions = self.regression_head(lstm_out[:, -1, :])
        return predictions, hidden

def process_yolo_predictions(results, device):
    """Process YOLO results into tensor format efficiently"""
    batch_size = len(results)
    predictions = np.zeros((batch_size, 2), dtype=np.float32)
    
    for i, result in enumerate(results):
        if len(result.boxes) > 0:
            # Get the box with highest confidence
            conf = result.boxes.conf
            max_conf_idx = conf.argmax().item()
            box = result.boxes.xywh[max_conf_idx].cpu().numpy()
            
            # Normalize coordinates
            predictions[i] = box / 640.0  # Normalize by image size
    
    return torch.from_numpy(predictions).to(device)

# Box and GIoU loss functions
def box_loss(pred_boxes, target_boxes, reduction='mean'):
    """
    Calculate the L1 loss between predicted and target boxes
    
    Args:
        pred_boxes: predicted bounding boxes [x, y, w, h] normalized
        target_boxes: target bounding boxes [x, y, w, h] normalized
        reduction: reduction method ('none', 'mean', 'sum')
    
    Returns:
        Box loss
    """
    loss = torch.abs(pred_boxes - target_boxes)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def bbox_to_corners(boxes):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def giou_loss(pred_boxes, target_boxes, reduction='mean'):
    """
    Calculate the GIoU loss between predicted and target boxes
    
    Args:
        pred_boxes: predicted bounding boxes [x, y, w, h] normalized
        target_boxes: target bounding boxes [x, y, w, h] normalized
        reduction: reduction method ('none', 'mean', 'sum')
    
    Returns:
        GIoU loss
    """
    # Convert from center format to corner format
    pred_corners = bbox_to_corners(pred_boxes)
    target_corners = bbox_to_corners(target_boxes)
    
    # Get coordinates of boxes
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_corners[:, 0], pred_corners[:, 1], pred_corners[:, 2], pred_corners[:, 3]
    target_x1, target_y1, target_x2, target_y2 = target_corners[:, 0], target_corners[:, 1], target_corners[:, 2], target_corners[:, 3]
    
    # Calculate area of prediction and target boxes
    pred_area = torch.abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    target_area = torch.abs((target_x2 - target_x1) * (target_y2 - target_y1))
    
    # Get coordinates of intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    # Calculate intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union area
    union_area = pred_area + target_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-7)
    
    # Get coordinates of enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    
    # Calculate area of enclosing box
    enclose_area = torch.abs((enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1))
    
    # Calculate GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
    
    # GIoU loss
    loss = 1.0 - giou
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def combined_loss(pred_boxes, target_boxes, box_weight=0.5, giou_weight=0.2):
    """
    Combined box loss and GIoU loss
    
    Args:
        pred_boxes: predicted bounding boxes [x, y, w, h] normalized
        target_boxes: target bounding boxes [x, y, w, h] normalized
        box_weight: weight for box loss
        giou_weight: weight for GIoU loss
    
    Returns:
        Combined loss
    """
    return box_weight * box_loss(pred_boxes, target_boxes) + giou_weight * giou_loss(pred_boxes, target_boxes)

def train_hotspot_detector(dataset, yolo_model, lstm_model, epochs=100, 
                         batch_size=2, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Ensure models are on the correct device
    lstm_model = lstm_model.to(device)
    yolo_model.to(device)
    
    # Create data loader with num_workers for faster loading
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=0)
    
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    box_losses = []
    giou_losses = []
    best_loss = float('inf')
    
    try:
        for epoch in range(epochs):
            lstm_model.train()
            epoch_losses = []
            epoch_box_losses = []
            epoch_giou_losses = []
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (image_sequences, target_bboxes) in enumerate(progress_bar):
                try:
                    image_sequences = image_sequences.to(device)
                    target_bboxes = target_bboxes.to(device)
                    
                    # Process sequence through YOLO
                    yolo_predictions = []
                    with torch.no_grad():
                        for i in range(image_sequences.size(1)):
                            results = yolo_model(image_sequences[:, i])
                            batch_preds = process_yolo_predictions(results, device)
                            yolo_predictions.append(batch_preds)
                    
                    # Stack predictions efficiently
                    yolo_predictions = torch.stack(yolo_predictions, dim=1)
                    
                    # LSTM prediction
                    optimizer.zero_grad()
                    predictions, _ = lstm_model(yolo_predictions)
                    
                    # Calculate combined loss on final bbox prediction
                    b_loss = box_loss(predictions, target_bboxes[:, -1])
                    g_loss = giou_loss(predictions, target_bboxes[:, -1])
                    loss = combined_loss(predictions, target_bboxes[:, -1], box_weight=0.5, giou_weight=0.2)
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    epoch_box_losses.append(b_loss.item())
                    epoch_giou_losses.append(g_loss.item())
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.6f}', 
                        'box': f'{b_loss.item():.6f}', 
                        'giou': f'{g_loss.item():.6f}'
                    })
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                avg_box_loss = np.mean(epoch_box_losses)
                avg_giou_loss = np.mean(epoch_giou_losses)
                
                train_losses.append(avg_loss)
                box_losses.append(avg_box_loss)
                giou_losses.append(avg_giou_loss)
                
                scheduler.step(avg_loss)
                
                logger.info(f'Epoch {epoch+1}: Total Loss = {avg_loss:.6f}, Box Loss = {avg_box_loss:.6f}, GIoU Loss = {avg_giou_loss:.6f}')
                
                # Save checkpoints
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'lstm_state_dict': lstm_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, 'best_hotspot_model.pth')
                
                if (epoch + 1) % 5 == 0:
                    torch.save({
                        'epoch': epoch + 1,
                        'lstm_state_dict': lstm_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, f'hotspot_checkpoint_epoch_{epoch+1}.pth')
                    
                    # Plot and save training progress
                    plt.figure(figsize=(12, 8))
                    
                    plt.subplot(2, 1, 1)
                    plt.plot(train_losses, label='Combined Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Hotspot Detection Training Progress')
                    plt.legend()
                    
                    plt.subplot(2, 1, 2)
                    plt.plot(box_losses, label='Box Loss', color='green')
                    plt.plot(giou_losses, label='GIoU Loss', color='orange')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Box and GIoU Loss Components')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig('hotspot_training_progress.png')
                    plt.close()
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        # Save emergency checkpoint
        torch.save({
            'lstm_state_dict': lstm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss if 'best_loss' in locals() else float('inf'),
        }, 'emergency_checkpoint.pth')
    
    return train_losses, box_losses, giou_losses

def main():
    # Update these paths to your actual paths
    image_dirs = [
        '../data/video1/frames',
        '../data/video2/frames',
        '../data/video3/frames',
        '../data/video4/frames'
    ]
    
    annotation_paths = [
        '../data/video1/annotations.json',
        '../data/video2/annotations.json',
        '../data/video3/annotations.json',
        '../data/video4/annotations.json'
    ]
    
    try:
        # Create dataset
        dataset = HotspotDataset(annotation_paths, image_dirs, sequence_length=5)
        
        # Initialize models
        # Use .yaml instead of .pt for the YOLO model
        yolo_model = YOLO('yolov8n.yaml')  # Changed from .pt to .yaml
        lstm_model = LSTMPredictor(input_size=4, hidden_size=128, num_layers=2)
        
        # Train models
        losses, box_losses, giou_losses = train_hotspot_detector(
            dataset=dataset,
            yolo_model=yolo_model,
            lstm_model=lstm_model,
            epochs=100,
            batch_size=2,  # Reduced batch size for stability
            learning_rate=0.0005  # Reduced learning rate
        )
        
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")

if __name__ == '__main__':
    main()
