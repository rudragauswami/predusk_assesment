# Technical Report: Multi-Object Detection and Persistent ID Tracking

**Author**: Rudra Gauswami  
**Date**: April 2026  
**Assignment**: Predusk AI/Computer Vision Assessment

---

## 1. Introduction

This report describes the design, implementation, and evaluation of a multi-object detection and tracking pipeline applied to publicly available video footage. The goal is to detect all relevant subjects (people), assign each a unique persistent ID, and maintain that identity across the full video duration despite real-world challenges.

---

## 2. Model / Detector Used

### YOLOv8 Nano (`yolov8n.pt`)

**YOLOv8** (You Only Look Once, version 8) by Ultralytics is the latest iteration of the YOLO family of single-stage object detectors. Key characteristics:

- **Architecture**: CSPDarknet backbone + PANet neck + decoupled detection head
- **Anchor-free design**: Eliminates the need for pre-defined anchor boxes, simplifying training and improving generalization
- **Model variant**: We use the **nano** variant (`yolov8n.pt`, ~6.2M parameters) which provides the best speed-to-accuracy ratio for real-time applications
- **Pre-trained on COCO**: Supports 80 object classes; we filter for class 0 (person) only
- **Confidence threshold**: Set to 0.35 to balance recall (catch partially visible subjects) with precision (avoid false positives)

**Why YOLOv8 over alternatives?**  
Compared to older detectors (Faster R-CNN, SSD) and even YOLOv5/v7, YOLOv8 offers:
- Higher mAP on COCO benchmarks at comparable speeds
- Built-in tracking integration via the Ultralytics API
- Active maintenance and community support
- Simpler deployment with fewer external dependencies

---

## 3. Tracking Algorithm Used

### BoT-SORT (Robust Associations Multi-Pedestrian Tracking)

**BoT-SORT** extends the classic SORT/DeepSORT paradigm with several improvements:

1. **Kalman Filter with Camera Motion Compensation (CMC)**: Predicts object positions in the next frame while accounting for global camera movement (pan, zoom, tilt). This is critical for sports/event footage where the camera often moves.

2. **IoU + Re-identification (ReID) fusion**: Association between detections and existing tracks uses a weighted combination of:
   - **IoU distance**: Spatial overlap between predicted and detected bounding boxes
   - **ReID cosine distance**: Appearance-based similarity using deep features extracted from each detection

3. **Two-stage association**: Similar to ByteTrack, BoT-SORT performs a second matching round for low-confidence detections, recovering subjects that might otherwise be lost.

**Why BoT-SORT over ByteTrack?**  
While ByteTrack is faster, BoT-SORT provides:
- Better ID consistency through ReID features (crucial when subjects look similar)
- Camera motion compensation (essential for sports footage with panning cameras)
- Comparable speed with significantly better identity preservation

---

## 4. How ID Consistency Is Maintained

The pipeline maintains persistent IDs through a multi-layered approach:

| Layer | Mechanism | Handles |
|-------|-----------|---------|
| **Prediction** | Kalman filter extrapolates position | Smooth motion, brief occlusion |
| **Spatial matching** | IoU-based association | Normal frame-to-frame tracking |
| **Appearance matching** | ReID feature cosine similarity | Similar-looking subjects, re-entry |
| **Camera compensation** | Global motion estimation (GMC) | Camera pan/zoom/tilt |
| **Track management** | Track age & hit/miss counters | Lost/found state transitions |

**Track lifecycle:**
1. **New detection** without a match → create new track with fresh ID
2. **Matched detection** → update track position, appearance, and reset miss counter
3. **Unmatched track** → increment miss counter; continue predicting via Kalman filter
4. **Track timeout** → if miss counter exceeds threshold (~30 frames), delete track

The `persist=True` flag in the Ultralytics API ensures that the tracker state is carried across frames within a single `model.track()` session, preventing ID resets.

---

## 5. Challenges Faced

### 5.1 Occlusion
When one person walks behind another, the occluded person's detection may drop for several frames. The Kalman filter continues predicting the position, and when the person reappears, the ReID features help re-associate them with their original ID.

### 5.2 Similar Appearance
In the `people-detection` video, several subjects wear similar dark clothing. Pure IoU-based tracking would frequently swap IDs. The ReID component of BoT-SORT mitigates this but does not eliminate it entirely.

### 5.3 Scale Changes
As subjects walk toward or away from the camera, their bounding box size changes significantly. The Kalman filter's state vector includes width/height, allowing it to adapt to gradual scale changes.

### 5.4 Edge-of-frame Entry/Exit
Subjects entering or leaving the frame boundary can produce partial detections with low confidence. The 0.35 confidence threshold and two-stage association help capture these marginal detections.

---

## 6. Failure Cases Observed

1. **ID switch during prolonged full occlusion**: If a person is completely hidden for >30 frames, the track is deleted and a new ID is assigned upon re-emergence.
2. **Very close proximity**: When two people stand very close together facing the same direction, their bounding boxes overlap significantly, occasionally causing brief ID swaps.
3. **Fast motion blur**: Rapid movement can degrade both detection confidence and ReID feature quality, leading to missed frames.

---

## 7. Possible Improvements

| Improvement | Benefit | Complexity |
|-------------|---------|------------|
| Use `yolov8s.pt` or `yolov8m.pt` | Higher detection accuracy | Low (drop-in) |
| Fine-tune ReID model on domain data | Better appearance discrimination | Medium |
| Add pose estimation (YOLOv8-pose) | Track by skeleton keypoints | Medium |
| Implement track interpolation | Fill gaps during brief occlusions | Low |
| Multi-camera fusion | Handle handoffs between views | High |
| Online learning of appearance | Adapt to lighting/clothing changes | High |

---

## 8. Pipeline Architecture

```
Input Video
    │
    ▼
┌──────────────┐
│  YOLOv8 Nano │ ── Detection (per-frame bounding boxes + confidence)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   BoT-SORT   │ ── Tracking (Kalman + ReID + CMC → persistent IDs)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Annotator   │ ── Visualization (boxes, IDs, trails, HUD)
└──────┬───────┘
       │
       ▼
  Output Video + Analytics (heatmap, trajectory map, count chart, JSON)
```

---

## 9. Conclusion

The combination of YOLOv8 and BoT-SORT provides a robust, efficient, and easy-to-deploy solution for multi-object tracking. The pipeline successfully handles common real-world challenges including occlusion, scale changes, and similar appearance while maintaining persistent identity assignment. The modular design allows for easy upgrades (e.g., swapping the detector or tracker) without architectural changes.
