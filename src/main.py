
import cv2
import argparse
import os
from ultralytics import YOLO
from distance_estimation import DistanceEstimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='dataset/bdd100k/bdd100k/images/100k/val', help='Input image or directory')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Model weights path')
    parser.add_argument('--out', type=str, default='output', help='Output folder')
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Distance estimation init (Focal length estimated for 720p)
    estimator = DistanceEstimator(focal_length_px=1200)

    # Output setup
    os.makedirs(args.out, exist_ok=True)

    # Load images
    if os.path.isfile(args.source):
        files = [args.source]
    elif os.path.isdir(args.source):
        files = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:10] # Limit to 10 for demo
    else:
        print(f"Source not found: {args.source}")
        return

    print(f"Processing {len(files)} images...")

    for img_path in files:
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Run inference
        results = model.predict(img, conf=0.4, verbose=False)
        
        # Annotation
        annotated_img = img.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]

                # Estimate distance using bounding box height
                dist = estimator.estimate_distance([x1, y1, x2, y2], name)

                # Draw
                color = (0, 255, 0)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                label = f"{name} {conf:.2f}"
                if dist:
                    label += f" | {dist}m"
                
                cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save
        base_name = os.path.basename(img_path)
        out_path = os.path.join(args.out, base_name)
        cv2.imwrite(out_path, annotated_img)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
