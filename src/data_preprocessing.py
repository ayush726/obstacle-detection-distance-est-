
import json
import os
from tqdm import tqdm

# Config
DATASET_ROOT = r'c:\Users\KIIT\Documents\projects\model\dataset'
IMAGES_ROOT = os.path.join(DATASET_ROOT, 'bdd100k', 'bdd100k', 'images', '100k')
LABELS_JSON_DIR = os.path.join(DATASET_ROOT, 'bdd100k_labels_release', 'bdd100k', 'labels')

# Output
OUTPUT_LABELS_DIR = os.path.join(DATASET_ROOT, 'bdd100k', 'bdd100k', 'labels', '100k')

# BDD100K to YOLO Class Mapping
# We map similar categories to IDs.
classes = [
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]
# Map raw category strings to IDs
class_map = {
    'person': 0, 'pedestrian': 0,
    'rider': 1,
    'car': 2, 'taxi': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motor': 6, 'motorcycle': 6,
    'bike': 7, 'bicycle': 7,
    'light': 8, 'traffic light': 8,
    'sign': 9, 'traffic sign': 9
}

def convert(json_file, split):
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Converting {split} labels...")
    
    out_dir = os.path.join(OUTPUT_LABELS_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    
    count = 0
    # Create a set of available image names to avoid creating labels for missing images
    # Check for local images to sync labels
    local_img_dir = os.path.join(IMAGES_ROOT, split)
    if not os.path.exists(local_img_dir):
        print(f"Warning: Local image directory {local_img_dir} does not exist. Skipping.")
        return

    available_imgs = set(os.listdir(local_img_dir))
    
    for item in tqdm(data):
        name = item['name']
        if name not in available_imgs:
            continue
            
        txt_path = os.path.join(out_dir, name.replace('.jpg', '.txt'))
        
        with open(txt_path, 'w') as f_out:
            for label in item.get('labels', []):
                cat = label['category']
                if cat not in class_map:
                    continue
                
                cls_id = class_map[cat]
                
                # BDD Box: x1, y1, x2, y2
                if 'box2d' not in label:
                    continue
                    
                b = label['box2d']
                x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
                
                # Normalize
                # Normalize coordinates (BDD100k Standard: 1280x720)
                dw = 1.0 / 1280
                dh = 1.0 / 720
                
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                
                cx *= dw
                w *= dw
                cy *= dh
                h *= dh
                
                f_out.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        count += 1
        
    print(f"Converted {count} labels for {split}.")

def main():
    convert(os.path.join(LABELS_JSON_DIR, 'bdd100k_labels_images_train.json'), 'train')
    convert(os.path.join(LABELS_JSON_DIR, 'bdd100k_labels_images_val.json'), 'val')

if __name__ == '__main__':
    main()
