import torch
import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from tqdm import tqdm
import torchvision.transforms as T
from collections import defaultdict
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes=2):
    """
    Create a Faster R-CNN model with custom number of classes
    
    Args:
        num_classes (int): Number of classes (including background)
        
    Returns:
        model: Initialized model with correct architecture
    """
    # Load base model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def load_model(checkpoint_path, num_classes=2):
    """
    Load a Faster R-CNN model with proper checkpoint handling
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        num_classes (int): Number of classes in your model
        
    Returns:
        model: Loaded model with weights
    """
    
    # Initialize model with correct number of classes
    model = create_model(num_classes=num_classes)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # If model weights are nested under 'model' key
            state_dict = checkpoint['model']
        else:
            # If checkpoint contains model weights directly
            state_dict = checkpoint
            
        # Remove unexpected keys if it's a dictionary
        if isinstance(state_dict, dict):
            keys_to_remove = ['optimizer', 'lr_scheduler', 'epoch', 'box_loss']
            for key in keys_to_remove:
                state_dict.pop(key, None)
            
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)  # Using strict=False to handle missing keys
        
    except Exception as e:
        raise
        
    return model
# Load Faster R-CNN model function from your training code
def get_model(num_classes=2):
    """Load Faster R-CNN model"""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace the classifier with a new one for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Custom dataset for COCO formatted test data
class COCOTestDataset(Dataset):
    def __init__(self, coco_annotation_path, image_dir):
        self.coco = COCO(coco_annotation_path)
        self.image_dir = image_dir
        self.ids = list(self.coco.imgs.keys())
        
        # Count frames with annotations
        self.frames_with_annotations = 0
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.frames_with_annotations += 1
    
    def __len__(self):
        return len(self.ids)
    
    def get_annotation_counts(self):
        """Return statistics about annotations in the dataset"""
        return {
            'total_frames': len(self.ids),
            'frames_with_annotations': self.frames_with_annotations,
            'frames_without_annotations': len(self.ids) - self.frames_with_annotations,
            'annotation_percentage': (self.frames_with_annotations / len(self.ids) * 100) if len(self.ids) > 0 else 0
        }
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.FloatTensor(np.array(img) / 255.0).permute(2, 0, 1)
        original_size = img.size
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Process annotations to get bounding boxes
        gt_boxes = []
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            # Convert to [x1, y1, x2, y2] format
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            gt_boxes.append([x1, y1, x2, y2])
        
        return {
            'image': img_tensor,
            'image_id': img_id,
            'gt_boxes': torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4)),
            'original_size': original_size,
            'has_annotations': len(gt_boxes) > 0,
            'original_image': img  # Keep original image for visualization
        }

# Custom collate function for the dataloader
def collate_fn(batch):
    collated_batch = {
        'image': [],
        'image_id': [],
        'gt_boxes': [],
        'original_size': [],
        'has_annotations': [],
        'original_image': []
    }
        
    for item in batch:
        for key in collated_batch:
            collated_batch[key].append(item[key])
                
    return collated_batch

# Function to calculate IoU between two boxes
def box_iou(box1, box2):
    """
    Calculate IoU between box1 and box2
    Boxes are in [x1, y1, x2, y2] format
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

# Function to calculate mIoU for a set of predictions and ground truths
def calculate_miou(pred_boxes, gt_boxes, iou_threshold=0.2):
    """
    Calculate mean IoU for predicted boxes and ground truth boxes
    Returns mIoU and details about matches
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {
            'miou': 0,
            'precision': 0,
            'recall': 0,
            'matched_pred_indices': [],
            'matched_gt_indices': [],
            'matched_ious': [],
            'num_predictions': len(pred_boxes),
            'num_ground_truths': len(gt_boxes),
            'true_positives': 0,
            'false_positives': len(pred_boxes),
            'false_negatives': len(gt_boxes)
        }
    
    ious = torch.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            ious[i, j] = box_iou(pred_box, gt_box)
    
    # Match predictions to ground truths
    matched_gt_indices = []
    matched_pred_indices = []
    matched_ious = []
    
    # Greedy matching
    while True:
        # Find max IoU
        if ious.numel() == 0 or ious.max() < iou_threshold:
            break
            
        max_iou, indices = ious.max(dim=1)
        max_idx = max_iou.argmax()
        gt_idx = indices[max_idx]
        
        # Add match
        matched_gt_indices.append(gt_idx.item())
        matched_pred_indices.append(max_idx.item())
        matched_ious.append(max_iou[max_idx].item())
        
        # Remove matched pairs
        ious[max_idx, :] = 0
        ious[:, gt_idx] = 0
    
    # Calculate metrics
    true_positives = len(matched_ious)
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    miou = sum(matched_ious) / len(matched_ious) if matched_ious else 0
    
    return {
        'miou': miou,
        'precision': precision,
        'recall': recall,
        'matched_pred_indices': matched_pred_indices,
        'matched_gt_indices': matched_gt_indices,
        'matched_ious': matched_ious,
        'num_predictions': len(pred_boxes),
        'num_ground_truths': len(gt_boxes),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

# Function to run inference on a single image
def run_inference(model, image, device='cuda', confidence_threshold=0.2):
    """
    Run inference on a single image and return predictions
    """
    # Convert to tensor and move to device
    img = image.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model([img])
    
    # Process predictions
    pred_boxes = outputs[0]['boxes']
    pred_scores = outputs[0]['scores']
    
    # Keep only predictions with confidence above threshold
    keep = pred_scores > confidence_threshold
    pred_boxes = pred_boxes[keep]
    
    return pred_boxes.cpu().tolist()

def evaluate_model(model, dataloader, device='cuda', confidence_threshold=0.2, fixed_denominator=285):
    """
    Evaluate model on test dataset
    Returns mIoU and other metrics, including a mIoU calculated over a fixed denominator
    """
    model.to(device)
    model.eval()
   
    all_metrics = []
    frames_with_predictions = 0
    total_frames = 0
    sum_iou = 0  # Sum of all IoUs for fixed denominator calculation
   
    for batch in tqdm(dataloader, desc="Evaluating model"):
        batch_metrics = []
       
        # Process each image in batch
        for i in range(len(batch['image'])):
            image = batch['image'][i]
            gt_boxes = batch['gt_boxes'][i]
            image_id = batch['image_id'][i]
            total_frames += 1
           
            # Run inference
            pred_boxes = run_inference(model, image, device, confidence_threshold)
           
            # Count frames with predictions
            if len(pred_boxes) > 0:
                frames_with_predictions += 1
           
            # Calculate metrics
            metrics = calculate_miou(
                torch.tensor(pred_boxes) if pred_boxes else torch.zeros((0, 4)),
                gt_boxes
            )
            
            # Add to sum of IoUs for fixed denominator calculation
            sum_iou += metrics['miou']
           
            metrics['image_id'] = image_id
            metrics['has_prediction'] = len(pred_boxes) > 0
            metrics['has_gt'] = len(gt_boxes) > 0
            batch_metrics.append(metrics)
       
        all_metrics.extend(batch_metrics)
   
    # Calculate average metrics across dataset
    frames_with_valid_metrics = [m for m in all_metrics if m['has_gt'] or m['has_prediction']]
   
    if frames_with_valid_metrics:
        avg_miou = sum(m['miou'] for m in frames_with_valid_metrics) / len(frames_with_valid_metrics)
        avg_precision = sum(m['precision'] for m in frames_with_valid_metrics) / len(frames_with_valid_metrics)
        avg_recall = sum(m['recall'] for m in frames_with_valid_metrics) / len(frames_with_valid_metrics)
    else:
        avg_miou = 0
        avg_precision = 0
        avg_recall = 0
    
    # Calculate mIoU with fixed denominator (285)
    miou_fixed_denominator = sum_iou / fixed_denominator
   
    # Calculate mAP (mean Average Precision)
    # This is simplified, a more accurate version would calculate AP at different IoU thresholds
    map_score = avg_precision
   
    # Count frames with ground truth annotations in evaluation set
    frames_with_ground_truth = sum(1 for m in all_metrics if m['has_gt'])
   
    # Frame counts
    frame_counts = {
        'total_frames': total_frames,
        'frames_with_ground_truth': frames_with_ground_truth,
        'frames_with_predictions': frames_with_predictions,
        'ground_truth_percentage': (frames_with_ground_truth / total_frames * 100) if total_frames > 0 else 0,
        'prediction_percentage': (frames_with_predictions / total_frames * 100) if total_frames > 0 else 0
    }
   
    return {
        'miou': avg_miou,
        'miou_fixed_denominator': miou_fixed_denominator,  # New metric with fixed denominator
        'precision': avg_precision,
        'recall': avg_recall,
        'map': map_score,
        'frame_counts': frame_counts,
        'per_image_metrics': all_metrics
    }
# Visualize predictions for debugging
def visualize_predictions(image, pred_boxes, gt_boxes, save_path=None):
    """
    Visualize predictions vs ground truth
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Draw ground truth boxes (green)
    for box in gt_boxes:
        x1, y1, x2, y2 = box.tolist() if isinstance(box, torch.Tensor) else box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor='green', linewidth=2))
        plt.text(x1, y1, 'GT', bbox=dict(facecolor='green', alpha=0.5))
    
    # Draw predicted boxes (red)
    for box in pred_boxes:
        x1, y1, x2, y2 = box if isinstance(box, list) else box.tolist()
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor='red', linewidth=2))
        plt.text(x1, y1, 'Pred', bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Main function to run evaluation
def main():
    # Configuration - UPDATE THESE PATHS TO MATCH YOUR ENVIRONMENT
    model_path = //path to model
    coco_annotation_path = //path to annotation path
    test_image_dir = //path to test_images_folder
    results_dir = r"evaluation_results" 
    batch_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    confidence_threshold = 0.2  # Adjust based on your model's performance
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model using the more robust load_model function
    model = load_model(model_path, num_classes=2)  # Use the load_model function from the second snippet
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = COCOTestDataset(
        coco_annotation_path=coco_annotation_path,
        image_dir=test_image_dir
    )
    
    # Print annotation statistics
    annotation_counts = test_dataset.get_annotation_counts()
    print("\n===== Test Dataset Statistics =====")
    print(f"Total frames: {annotation_counts['total_frames']}")
    print(f"Frames with annotations: {annotation_counts['frames_with_annotations']} ({annotation_counts['annotation_percentage']:.6f}%)")
    print(f"Frames without annotations: {annotation_counts['frames_without_annotations']}")
    print("==================================\n")
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_dataloader, device, confidence_threshold, fixed_denominator=285)
    # Print results
    print("\n===== Model Evaluation Results =====")
    print(f"Mean IoU: {metrics['miou']:.6f}")
    print(f"Mean IoU (over 285): {metrics['miou_fixed_denominator']:.6f}")  # Add this line
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"mAP: {metrics['map']:.6f}")
    print("===================================\n")
    # Print frame counts
    frame_counts = metrics['frame_counts']
    print("\n===== Frame Prediction Counts =====")
    print(f"Total frames evaluated: {frame_counts['total_frames']}")
    print(f"Frames with ground truth annotations: {frame_counts['frames_with_ground_truth']} ({frame_counts['ground_truth_percentage']:.4f}%)")
    print(f"Frames with model predictions: {frame_counts['frames_with_predictions']} ({frame_counts['prediction_percentage']:.4f}%)")
    print("===================================\n")
    
    # Visualize some predictions
    # Create a results visualization directory
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize the first 10 samples or all if less than 10
    num_vis = min(10, len(test_dataset))
    for i in range(num_vis):
        sample_data = test_dataset[i]
        
        # Run inference
        pred_boxes = run_inference(model, sample_data['image'], device, confidence_threshold)
        
        # Visualize and save
        vis_path = os.path.join(vis_dir, f"sample_{i}.png")
        visualize_predictions(
            sample_data['original_image'],
            pred_boxes,
            sample_data['gt_boxes'],
            save_path=vis_path
        )
        print(f"Saved visualization to {vis_path}")
    
    # Save detailed results to file
    import json
    
    # Convert tensor data to serializable format
    serializable_metrics = {
        'miou': metrics['miou'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'map': metrics['map'],
        'frame_counts': metrics['frame_counts'],
        'per_image_metrics': [
            {k: v.tolist() if isinstance(v, torch.Tensor) else 
               ([x.tolist() if isinstance(x, torch.Tensor) else x for x in v] if isinstance(v, list) else v)
             for k, v in m.items()}
            for m in metrics['per_image_metrics']
        ]
    }
    
    with open(os.path.join(results_dir, 'faster_rcnn_evaluation_results.json'), 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"Saved detailed results to {os.path.join(results_dir, 'faster_rcnn_evaluation_results.json')}")

if __name__ == "__main__":
    main()
