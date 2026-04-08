# Demo Script – Multi-Object Detection & Tracking Pipeline

> **Use this script as talking points for your 3-5 minute demo video recording.**

---

## Slide 1: Introduction (30 seconds)

"Hi, I'm Rudra. In this demo I'll walk you through my multi-object detection and tracking pipeline built for the Predusk AI assessment.

The goal is straightforward: take a public video with multiple people, detect every person in every frame, and assign each one a unique persistent ID that stays consistent throughout the entire video."

---

## Slide 2: Tech Stack (45 seconds)

"For detection, I'm using **YOLOv8 Nano** from Ultralytics. It's a single-stage anchor-free detector pre-trained on COCO — it gives us fast, accurate bounding boxes for 80 object classes. I filter for class 0, which is 'person'.

For tracking, I chose **BoT-SORT** — it combines a **Kalman filter** for motion prediction with **ReID features** for appearance matching. This means even when two people look similar or briefly overlap, the tracker can usually tell them apart and keep their IDs stable.

The entire pipeline runs through the Ultralytics API, which cleanly integrates detection and tracking into a single `model.track()` call — no glue code needed."

---

## Slide 3: Running the Pipeline (45 seconds)

"Let me show you how to run it. After cloning the repo and installing dependencies:

```bash
pip install -r requirements.txt
python main.py --video people-detection.mp4
```

The script processes every frame, prints progress updates, and produces several outputs:
- The annotated video with bounding boxes and IDs
- A trajectory map showing movement paths
- A movement heatmap
- An object count over time chart
- Sample screenshots
- An analytics JSON file with processing metrics"

---

## Slide 4: Results Walkthrough (60 seconds)

*Show the annotated video playing*

"Here you can see the output. Each detected person has:
- A **coloured bounding box** with rounded corners
- A **label** showing their unique ID and confidence score
- A **trajectory trail** showing their recent movement path

The HUD in the top corner shows the current frame number and how many people are actively being tracked.

Let me show some specific scenarios:
- **Here** — two people cross paths. Watch how their IDs stay consistent even as their bounding boxes overlap.
- **And here** — a person partially leaves the frame and comes back. The tracker holds onto their ID for a configurable timeout period."

---

## Slide 5: Analytics (30 seconds)

*Show trajectory map, heatmap, and count chart*

"Beyond the video, the pipeline generates three analytics visualisations:
1. **Trajectory map** — lines showing each person's full path, green dot for start, red for end
2. **Movement heatmap** — density of all tracked positions overlaid as a JET colourmap
3. **Count over time** — temporal bar chart showing how many people were visible in each frame"

---

## Slide 6: Challenges & Limitations (30 seconds)

"Some honest limitations:
- During prolonged full occlusion (30+ frames), the track times out and a new ID is assigned
- In very close proximity scenarios, brief ID swaps can still occur
- I used the nano model for speed — switching to yolov8s or yolov8m would improve accuracy

These could be addressed with pose estimation, track interpolation, or domain-specific ReID fine-tuning."

---

## Slide 7: Wrap Up (15 seconds)

"Thanks for watching. The full codebase is on GitHub with the README, technical report, and all generated outputs. The pipeline is modular — you can swap the video, model, or tracker via CLI arguments. Happy to discuss any questions!"

---

**Total estimated time: ~4 minutes**
