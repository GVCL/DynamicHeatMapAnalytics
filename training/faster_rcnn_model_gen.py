import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import logging
import random
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_debug.log'),
        logging.StreamHandler()
    ]
)

class COCOHotspotDataset(Dataset):
    """Dataset class for loading COCO format hotspot annotations"""
    
    def __init__(self, image_dirs, annotation_paths):
        self.all_images = []
        self.all_targets = []
        self.image_to_path = {}
        
        # Statistics
        self.frames_with_boxes = 0
        self.empty_frames = 0
        self.total_boxes = 0
        
        for img_dir, ann_path in zip(image_dirs, annotation_paths):
            if not os.path.exists(img_dir) or not os.path.exists(ann_path):
                logging.error(f"Invalid path - Image dir: {img_dir} or Annotation file: {ann_path}")
                continue
            
            # Load COCO format annotations
            with open(ann_path, 'r') as f:
                coco_data = json.load(f)
            
            logging.info(f"Loaded COCO annotations from {ann_path}")
            logging.info(f"Found {len(coco_data.get('images', []))} images and {len(coco_data.get('annotations', []))} annotations")
            
            # Create image_id to filename mapping
            image_id_to_filename = {}
            for img in coco_data.get('images', []):
                image_id_to_filename[img['id']] = img['file_name']
            
            # Create image_id to annotations mapping
            image_annotations = defaultdict(list)
            for ann in coco_data.get('annotations', []):
                image_id = ann['image_id']
                bbox = ann['bbox']  # [x, y, width, height] in COCO format
                
                # Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                image_annotations[image_id].append([x1, y1, x2, y2])
            
            # Process each image
            image_count = 0
            for image_id, filename in image_id_to_filename.items():
                image_path = os.path.join(img_dir, filename)
                
                # Skip if image file doesn't exist
                if not os.path.exists(image_path):
                    logging.warning(f"Image file not found: {image_path}")
                    continue
                
                # Get boxes for this image
                boxes = image_annotations[image_id]
                
                # Add to dataset
                self.all_images.append(image_id)
                self.all_targets.append(boxes)
                self.image_to_path[image_id] = image_path
                
                # Update statistics
                if boxes:
                    self.frames_with_boxes += 1
                    self.total_boxes += len(boxes)
                else:
                    self.empty_frames += 1
                
                image_count += 1
            
            logging.info(f"Processed {image_count} images from {img_dir}")
        
        logging.info(f"Dataset loaded: {len(self.all_images)} images total")
        logging.info(f"Statistics: {self.frames_with_boxes} frames with boxes, {self.empty_frames} empty frames")
        logging.info(f"Total boxes: {self.total_boxes}, avg boxes per frame with annotations: {self.total_boxes/max(1, self.frames_with_boxes):.2f}")
        
        # Verify some samples
        if self.all_images:
            for _ in range(min(3, len(self.all_images))):
                idx = random.randint(0, len(self.all_images) - 1)
                image_id = self.all_images[idx]
                boxes = self.all_targets[idx]
                logging.info(f"Sample {idx}: Image ID {image_id} has {len(boxes)} boxes")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        try:
            image_id = self.all_images[idx]
            img_path = self.image_to_path[image_id]
            
            # Load and convert image
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.FloatTensor(np.array(img) / 255.0).permute(2, 0, 1)
            
            # Get boxes
            boxes = self.all_targets[idx]
            boxes_tensor = torch.FloatTensor(boxes) if boxes else torch.zeros((0, 4))
            
            # Create target dictionary
            target = {
                "boxes": boxes_tensor,
                "labels": torch.ones((len(boxes_tensor),), dtype=torch.int64),  # All boxes are class 1 (hotspot)
                "image_id": torch.tensor([image_id]),
                "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]),
                "iscrowd": torch.zeros((len(boxes_tensor),), dtype=torch.int64)
            }
            
            return img_tensor, target
            
        except Exception as e:
            logging.error(f"Error loading image ID {image_id}: {str(e)}")
            raise

def visualize_dataset(dataset, num_samples=5):
    """Visualize samples from the dataset with bounding boxes"""
    if len(dataset) == 0:
        logging.error("Dataset is empty, cannot visualize samples")
        return
    
    plt.figure(figsize=(20, 4*num_samples))
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(indices):
        img, target = dataset[idx]
        
        # Convert tensor to numpy for visualization
        img_np = img.permute(1, 2, 0).numpy()
        
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img_np)
        plt.title(f"Image ID: {target['image_id'].item()}")
        
        # Draw bounding boxes
        boxes = target['boxes'].numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                              fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png')
    logging.info("Saved dataset visualization to dataset_visualization.png")
    plt.close()

def get_model(num_classes=2):  # Background (0) + Hotspot (1)
    """Initialize the Faster R-CNN model"""
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier with a new one for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Log model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model initialized with {num_classes} classes")
    logging.info(f"Model has {total_params:,} total parameters, {trainable_params:,} trainable")
    
    return model

def collate_fn(batch):
    """Custom collate function for the dataloader"""
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """Train the model for one epoch"""
    model.train()
    
    lr_scheduler = None
    if epoch == 0:
        # Warm-up learning rate scheduler
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        
        def warmup_lr_scheduler(step):
            if step < warmup_iters:
                return warmup_factor * (1 - step / warmup_iters) + step / warmup_iters
            return 1.0
            
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)
    
    running_loss = 0.0
    running_loss_box = 0.0
    running_loss_cls = 0.0
    running_loss_rpn = 0.0
    batch_count = 0
    
    for i, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Skip if no valid boxes
        if all(len(t['boxes']) == 0 for t in targets):
            logging.info(f"Batch {i}: Skipping because no valid boxes")
            continue
        
        # Forward pass 
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Extract individual losses
        loss_box = loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
        loss_cls = loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
        loss_rpn_box = loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
        loss_rpn_cls = loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
        
        # Update running losses
        running_loss += losses.item()
        running_loss_box += loss_box
        running_loss_cls += loss_cls
        running_loss_rpn += loss_rpn_box + loss_rpn_cls
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        batch_count += 1
        
        # Print progress
        if i % print_freq == 0:
            logging.info(f"Epoch {epoch+1} - Batch {i}/{len(data_loader)}: "
                       f"Loss: {losses.item():.4f}, "
                       f"Box: {loss_box:.4f}, "
                       f"Cls: {loss_cls:.4f}, "
                       f"RPN: {loss_rpn_box + loss_rpn_cls:.4f}")
    
    # Calculate average losses
    if batch_count > 0:
        avg_loss = running_loss / batch_count
        avg_loss_box = running_loss_box / batch_count
        avg_loss_cls = running_loss_cls / batch_count
        avg_loss_rpn = running_loss_rpn / batch_count
        
        logging.info(f"Epoch {epoch+1} Complete - "
                   f"Avg Loss: {avg_loss:.4f}, "
                   f"Box: {avg_loss_box:.4f}, "
                   f"Cls: {avg_loss_cls:.4f}, "
                   f"RPN: {avg_loss_rpn:.4f}")
        
        return avg_loss, avg_loss_box, avg_loss_cls, avg_loss_rpn
    else:
        logging.warning("No valid batches processed in this epoch")
        return 0, 0, 0, 0

def train_model(image_dirs, annotation_paths, output_dir='models', num_epochs=100, batch_size=2):
    """Main training function - uses all data for training and plots only bbox loss"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = COCOHotspotDataset(image_dirs, annotation_paths)
    
    if len(dataset) == 0:
        logging.error("Dataset is empty. Training cannot proceed.")
        return
    
    # Visualize samples
    visualize_dataset(dataset)
    
    # Use all data for training - no validation split
    logging.info(f"Using all {len(dataset)} datapoints for training")
    
    # Initialize data loader - all data used for training
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Adjust based on your system
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    model = get_model()
    model.to(device)
    
    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler - reduce by factor of 0.1 every 5 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    history = {
        'epochs': [],
        'box_loss': [],
    }
    
    for epoch in range(num_epochs):
        # Train one epoch
        epoch_loss, epoch_box_loss, epoch_cls_loss, epoch_rpn_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save history - only box loss
        history['epochs'].append(epoch + 1)
        history['box_loss'].append(epoch_box_loss)
        
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'box_loss': epoch_box_loss
        }
        
        # Save model
        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save model at milestone epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
            logging.info(f"Saved model at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Plot only bbox loss history
    plt.figure(figsize=(12, 8))
    plt.plot(history['epochs'], history['box_loss'], 'r-', linewidth=2)
    plt.title('Bounding Box Regression Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')
    plt.grid(True)
    plt.xticks(range(0, num_epochs+1, 5))  # Mark every 5 epochs
    
    # Add trend line
    try:
        z = np.polyfit(history['epochs'], history['box_loss'], 1)
        p = np.poly1d(z)
        plt.plot(history['epochs'], p(history['epochs']), "b--", linewidth=1, 
                 label=f"Trend: y={z[0]:.6f}x+{z[1]:.2f}")
        plt.legend()
    except:
        logging.warning("Could not plot trend line")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_loss_history.png'), dpi=300)
    plt.close()
    
    logging.info(f"Training completed. Models saved to {output_dir}")
    logging.info(f"Box loss plot saved to {output_dir}/bbox_loss_history.png")
    
    return model

def load_and_predict(model_path, image_path, confidence_threshold=0.2, device=None):
    """Load a trained model and make predictions on a single image"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.FloatTensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    
    # Filter by confidence
    boxes = prediction['boxes'][prediction['scores'] >= confidence_threshold]
    scores = prediction['scores'][prediction['scores'] >= confidence_threshold]
    
    # Convert back to numpy for visualization
    image_np = np.array(image)
    
    # Draw boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(image_np)
    
    for i, (box, score) in enumerate(zip(boxes.cpu().numpy(), scores.cpu().numpy())):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                          fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(x1, y1-5, f"Hotspot: {score:.2f}", color='red',
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_output.png')
    plt.close()
    
    logging.info(f"Prediction complete. Detected {len(boxes)} hotspots with confidence threshold {confidence_threshold}")
    logging.info(f"Visualization saved to prediction_output.png")
    
    return boxes.cpu().numpy(), scores.cpu().numpy()

def main():
    # Define paths
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
    
    # Train model with modified parameters
    model = train_model(
        image_dirs=image_dirs,
        annotation_paths=annotation_paths,
        output_dir='hotspot_models',
        num_epochs=100,
        batch_size=2
    )
    
    logging.info("Training complete!")

if __name__ == "__main__":
    main()
