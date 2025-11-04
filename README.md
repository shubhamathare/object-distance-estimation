# Object Distance Estimation using YOLOv11

This project detects objects in images and estimates their distance from the camera using the YOLOv11 object detection model.  
It uses a simple geometric approach based on the perceived width of detected objects.

---

##  Features
- Real-time or image-based object detection using **YOLOv11**
- Distance estimation based on object bounding box width
- Automatically saves annotated images with distances
- Exports optimized **ONNX** model for deployment

---

## How It Works
The distance to an object is estimated using the formula:

\[
\text{Distance} = \frac{\text{Known Width} \times \text{Focal Length}}{\text{Perceived Width}}
\]

- **Known Width** → Approximate real-world width of the object (e.g., car = 1.8 m)  
- **Focal Length** → Pre-calibrated focal length of the camera in pixels  
- **Perceived Width** → Width of the detected bounding box in pixels  

---


Parameters (in detect.py)

KNOWN_DISTANCE – Actual distance of a reference object in meters

KNOWN_WIDTH – Real width of the object (e.g., car = 1.8 m)

FOCAL_LENGTH – Camera focal length in pixels (tune as needed)


Output Example

For each image in bdd100k/input_images/, an annotated version is saved in bdd100k/output_images/ showing:

Bounding boxes

Object labels

Estimated distance in meters


Contact Info
Name : Shubham Shankar Athare - Contact Number : 7666263925 - Email Address : shubhamathare701@gmail.com
