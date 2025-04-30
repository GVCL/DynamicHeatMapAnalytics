import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm
import logging
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class EvalDataset(Dataset):
    def __init__(self, image_paths, bbox_annotations, sequence_length=5):
        self.sequence_length = sequence_length
        self.sequences = []
        self.max_bbox_dim = 640  # Maximum dimension for normalization
        
        # Build sequences from data
        for i in range(len(image_paths) - sequence_length + 1):
            seq_img_paths = []
            seq_bboxes = []
            
            for j in range(sequence_length):
                img_path = image_paths[i + j]
                bbox = bbox_annotations[i + j]
                
                seq_img_paths.append(img_path)
                seq_bboxes.append(bbox)
            
            self.sequences.append((seq_img_paths, seq_bboxes))
        
        logger.info(f"Created evaluation dataset with {len(self.sequences)} sequences")
        
        # Count sequences with annotations
        self.sequences_with_annotations = 0
        for _, bboxes in self.sequences:
            if any(sum(bbox) > 0 for bbox in bboxes):
                self.sequences_with_annotations += 1
                
        logger.info(f"Sequences with annotations: {self.sequences_with_annotations} ({self.sequences_with_annotations/len(self.sequences)*100:.2f}%)")
    
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

def run_inference_sequence(yolo_model, lstm_model, images, device):
    """Run inference on a sequence of images using YOLO and LSTM models"""
    yolo_predictions = []
    
    # Process each image with YOLO
    for i in range(images.size(0)):
        # Add batch dimension and ensure the input is correctly shaped
        # Critical fix: Make sure image is properly reshaped for YOLO
        img = images[i].unsqueeze(0)  # Add batch dimension to make (1, 3, 640, 640)
        
        # Run YOLO
        results = yolo_model(img)
        
        # Process results
        if len(results) > 0 and len(results[0].boxes) > 0:
            conf = results[0].boxes.conf
            max_conf_idx = conf.argmax().item()
            box = results[0].boxes.xywh[max_conf_idx].cpu().numpy()
            # Normalize coordinates
            pred = box / 640.0
        else:
            pred = np.zeros(4)
            
        yolo_predictions.append(torch.tensor(pred, dtype=torch.float32).to(device))
    
    # Stack predictions and run through LSTM
    yolo_preds_tensor = torch.stack(yolo_predictions).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        lstm_output, _ = lstm_model(yolo_preds_tensor)
    
    return lstm_output[0].cpu().numpy()

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x, y, width, height] format"""
    # Convert xywh to xyxy for easier calculation
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate box areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_yolo_lstm(yolo_model, lstm_model, dataloader, device, iou_threshold=0.2, 
                      results_dir='evaluation_results'):
    """Evaluate YOLO+LSTM model on test data with enhanced metrics"""
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)
    
    yolo_model.to(device)
    lstm_model.to(device)
    lstm_model.eval()
    
    total_sequences = 0
    correct_predictions = 0
    total_iou = 0.0
    
    frames_with_gt = 0
    frames_with_predictions = 0
    
    # Extended metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Store per-sequence metrics for detailed analysis
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, (image_sequences, target_bboxes) in enumerate(tqdm(dataloader, desc="Evaluating YOLO+LSTM model")):
            batch_size = image_sequences.size(0)
            total_sequences += batch_size
            
            for i in range(batch_size):
                # Get this sequence's images and target
                sequence = image_sequences[i].to(device)
                target = target_bboxes[i, -1].cpu().numpy()  # Use final frame's bbox as target
                
                # Check if target has annotations
                has_gt = np.sum(target) > 0
                if has_gt:
                    frames_with_gt += 1
                
                # Run inference
                try:
                    pred_box = run_inference_sequence(yolo_model, lstm_model, sequence, device)
                    
                    # Check if prediction is non-zero
                    has_prediction = np.sum(pred_box) > 0
                    if has_prediction:
                        frames_with_predictions += 1
                    
                    # Calculate IoU - but only if we have ground truth
                    iou = 0
                    if has_gt:
                        iou = calculate_iou(pred_box, target)
                        total_iou += iou
                        
                        # Update TP, FP, FN
                        if iou > iou_threshold:
                            true_positives += 1
                            correct_predictions += 1
                        else:
                            false_negatives += 1
                    elif has_prediction:
                        # No ground truth but we have a prediction
                        false_positives += 1
                    
                    # Store metrics for this sequence
                    sequence_metrics = {
                        'batch_idx': batch_idx,
                        'sequence_idx': i,
                        'has_gt': has_gt,
                        'has_prediction': has_prediction,
                        'iou': iou,
                        'correct': iou > iou_threshold if has_gt else False
                    }
                    all_metrics.append(sequence_metrics)
                    
                    # Visualize some predictions (first 10 batches)
                    if batch_idx < 5 and i == 0:
                        # Visualize the prediction
                        vis_path = os.path.join(results_dir, "visualizations", f"batch_{batch_idx}_seq_{i}.png")
                        visualize_prediction(
                            sequence[-1].cpu().numpy().transpose(1, 2, 0),  # Last frame
                            pred_box,
                            target,
                            save_path=vis_path
                        )
                
                except Exception as e:
                    logger.error(f"Error in sequence {batch_idx}_{i}: {str(e)}")
                    continue
                
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Calculate metrics
    accuracy = correct_predictions / frames_with_gt if frames_with_gt > 0 else 0
    avg_iou = total_iou / frames_with_gt if frames_with_gt > 0 else 0
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Use precision as a simplified mAP (mean Average Precision)
    map_score = precision
    
    metrics = {
        "total_sequences": total_sequences,
        "frames_with_gt": frames_with_gt,
        "frames_with_predictions": frames_with_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "average_iou": avg_iou,
        "mean_iou": avg_iou,  # Same as average_iou, just different name
        "iou_over_denominator": total_iou / 285 if 285 > 0 else 0,  # As requested
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "map": map_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "per_sequence_metrics": all_metrics
    }
    
    # Save metrics to file
    with open(os.path.join(results_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4, default=lambda x: x.item() if isinstance(x, (np.int64, np.float64)) else str(x))
    
    return metrics

def visualize_prediction(image, pred_box, gt_box, save_path=None):
    """Visualize prediction and ground truth for a single frame"""
    # Denormalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    
    # Denormalize boxes from [0,1] to pixel coordinates
    img_h, img_w = image.shape[:2]
    
    # Convert from center format to corner format for display
    if np.sum(gt_box) > 0:
        # Ground truth box (green)
        x, y, w, h = gt_box * np.array([img_w, img_h, img_w, img_h])
        x1, y1 = x - w/2, y - h/2
        plt.gca().add_patch(plt.Rectangle((x1, y1), w, h, 
                                         fill=False, edgecolor='green', linewidth=2))
        plt.text(x1, y1, 'GT', bbox=dict(facecolor='green', alpha=0.5))
    
    if np.sum(pred_box) > 0:
        # Prediction box (red)
        x, y, w, h = pred_box * np.array([img_w, img_h, img_w, img_h])
        x1, y1 = x - w/2, y - h/2
        plt.gca().add_patch(plt.Rectangle((x1, y1), w, h, 
                                         fill=False, edgecolor='red', linewidth=2))
        plt.text(x1, y1, 'Pred', bbox=dict(facecolor='red', alpha=0.5))
    
    plt.title(f"IoU: {calculate_iou(pred_box, gt_box):.4f}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Configuration - UPDATE THESE PATHS TO MATCH YOUR ENVIRONMENT
    model_path = //path to model
    test_annotation_file = //path to test annotation file
    test_image_dir = //path to test images folder
    results_dir = r"lstm_evaluation_results"  # Directory to save results
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load models
    try:
        # Load YOLOv8 model
        yolo_model = YOLO('yolov8n.pt')
        
        # Initialize and load LSTM model
        lstm_model = LSTMPredictor(input_size=4, hidden_size=128, num_layers=2)
        checkpoint = torch.load(model_path, map_location=device)
        lstm_model.load_state_dict(checkpoint['lstm_state_dict'])
        lstm_model.to(device)
        lstm_model.eval()
        
        logger.info(f"Models loaded successfully")
        
        # Load test data
        # This is a simplified example - replace with your actual test data loading logic
        import json
        import glob
        
        # Load annotations
        with open(test_annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Create image ID to filename mapping
        id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
        
        # Group annotations by image id
        annotations_by_frame = {}
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_frame:
                annotations_by_frame[img_id] = []
            annotations_by_frame[img_id].append(ann['bbox'])
        
        # Create sorted list of image paths and corresponding bbox annotations
        image_ids = sorted(id_to_filename.keys())
        image_paths = []
        bbox_annotations = []
        
        for img_id in image_ids:
            img_path = os.path.join(test_image_dir, id_to_filename[img_id])
            
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue
                
            image_paths.append(img_path)
            
            # Get bbox or use zeros if no annotation
            if img_id in annotations_by_frame:
                # Use the first bbox if multiple exist
                bbox = annotations_by_frame[img_id][0]
                # Convert from [x, y, w, h] to [x_center, y_center, w, h]
                x, y, w, h = bbox
                x_center = x + w/2
                y_center = y + h/2
                # Normalize bbox
                bbox = [
                    x_center / 640,
                    y_center / 640,
                    w / 640,
                    h / 640
                ]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            
            bbox_annotations.append(bbox)
        
        logger.info(f"Loaded {len(image_paths)} images")
        logger.info(f"Frames with annotations: {len([b for b in bbox_annotations if sum(b) > 0])}")
        logger.info(f"Frames without annotations: {len([b for b in bbox_annotations if sum(b) == 0])}")
        
        # Create dataset and dataloader
        sequence_length = 5
        test_dataset = EvalDataset(image_paths, bbox_annotations, sequence_length=sequence_length)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        logger.info(f"Total sequences: {len(test_dataset)}")
        
        # Print separator for clarity
        logger.info("=" * 50)
        
        # Evaluate model
        metrics = evaluate_yolo_lstm(yolo_model, lstm_model, test_dataloader, device, results_dir=results_dir)
        
        # Print results
        logger.info("\n===== Evaluation Results =====")
        logger.info(f"Total sequences evaluated: {metrics['total_sequences']}")
        logger.info(f"Frames with ground truth: {metrics['frames_with_gt']} ({metrics['frames_with_gt']/metrics['total_sequences']*100:.2f}%)")
        logger.info(f"Frames with predictions: {metrics['frames_with_predictions']} ({metrics['frames_with_predictions']/metrics['total_sequences']*100:.2f}%)")
        logger.info(f"Correct predictions (IoU > 0.1): {metrics['correct_predictions']}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Average IoU: {metrics['average_iou']:.4f}")
        logger.info(f"Mean IoU: {metrics['mean_iou']:.4f}")
        logger.info(f"IoU over denominator 285: {metrics['iou_over_denominator']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"mAP: {metrics['map']:.4f}")
        logger.info(f"True Positives: {metrics['true_positives']}")
        logger.info(f"False Positives: {metrics['false_positives']}")
        logger.info(f"False Negatives: {metrics['false_negatives']}")
        
        logger.info(f"Detailed metrics saved to {os.path.join(results_dir, 'detailed_metrics.json')}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
