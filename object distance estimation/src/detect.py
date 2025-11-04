from ultralytics import YOLO
import cv2
import os

#  Load YOLOv11 model
print("Loading YOLOv11 model...")
model = YOLO("yolo11n.pt")

#  Define constants
KNOWN_DISTANCE = 5.0   # meters
KNOWN_WIDTH = 1.8      # meters (average car width)
FOCAL_LENGTH = 800     # in pixels (adjust if calibrated)

#  Set folders
input_folder = "bdd100k/input_images"
output_folder = "bdd100k/output_images"
os.makedirs(output_folder, exist_ok=True)

# Define distance estimation
def estimate_distance(perceived_width):
    if perceived_width == 0:
        return None
    return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width

#   detection + distance estimation
print(f"Processing images from '{input_folder}'...")
for img_name in os.listdir(input_folder):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)
    results = model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            distance = estimate_distance(w)
            label = f"{r.names[int(box.cls[0])]}: {distance:.2f}m" if distance else "Unknown"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out_path = os.path.join(output_folder, img_name)
    cv2.imwrite(out_path, img)
    print(f"Processed: {img_name}, saved to {out_path}")

print(" Distance estimation completed! Check output_images folder.")

#  optimized ONNX model
print("\n Exporting YOLOv11 model to ONNX (optimized)...")
model.export(format="onnx", optimize=True)
print(" Model exported as 'yolo11n.onnx' successfully!")
