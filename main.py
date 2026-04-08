"""
Multi-Object Detection and Persistent ID Tracking Pipeline
===========================================================

This script implements a computer vision pipeline that:
1. Detects all persons in each frame using YOLOv8
2. Assigns consistent, unique IDs via BoT-SORT multi-object tracking
3. Produces an annotated output video with bounding boxes and IDs
4. Generates trajectory visualizations and analytics

Author : Rudra Gauswami
Video  : Intel IoT DevKit - people-detection.mp4
"""

import argparse
import os
import sys
import time
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: 'ultralytics' package not found. Install via: pip install ultralytics")
    sys.exit(1)


# ─────────────────────────── colour palette ───────────────────────────────────
def _generate_colors(n: int = 64) -> list:
    """Generate a visually distinct colour palette using HSV spacing."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))
    return colors


COLORS = _generate_colors(64)


def get_color(track_id: int) -> tuple:
    """Return a deterministic colour for a given track ID."""
    return COLORS[track_id % len(COLORS)]


# ─────────────────────── annotation helpers ───────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=8):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    # Four corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Four edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)


def draw_label(img, text, origin, color, font_scale=0.55, thickness=2):
    """Draw a text label with a filled background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    # background
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 6, y + 4), color, -1)
    # text (white on coloured background)
    cv2.putText(img, text, (x + 3, y - 4), font, font_scale, (255, 255, 255), thickness)


def annotate_frame(frame, boxes, track_ids, confidences, trails, frame_count):
    """Draw bounding boxes, IDs, confidence scores, and trajectory trails."""
    overlay = frame.copy()

    for box, tid, conf in zip(boxes, track_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = get_color(tid)

        # bounding box
        draw_rounded_rect(overlay, (x1, y1), (x2, y2), color, 2)

        # label
        label = f"ID {tid}  {conf:.0%}"
        draw_label(overlay, label, (x1, y1 - 2), color)

        # centre dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(overlay, (cx, cy), 3, color, -1)

    # draw trajectory trails
    for tid, points in trails.items():
        if len(points) < 2:
            continue
        color = get_color(tid)
        pts = np.array(points, dtype=np.int32)
        # fade older points
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thick = max(1, int(2 * alpha))
            cv2.line(overlay, tuple(pts[i - 1]), tuple(pts[i]), color, thick)

    # HUD – frame counter + active tracks
    hud_text = f"Frame {frame_count}  |  Active: {len(track_ids)}"
    cv2.putText(overlay, hud_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)

    # blend for a subtle overlay effect
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


# ───────────────────── analytics & visualisations ─────────────────────────────
def draw_trajectory_map(frame_shape, all_trails, output_path):
    """Generate a full-video trajectory visualisation image."""
    canvas = np.zeros(frame_shape, dtype=np.uint8)
    for tid, points in all_trails.items():
        if len(points) < 2:
            continue
        color = get_color(tid)
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
        # start and end markers
        cv2.circle(canvas, tuple(points[0]), 5, (0, 255, 0), -1)
        cv2.circle(canvas, tuple(points[-1]), 5, (0, 0, 255), -1)

    cv2.putText(canvas, "Trajectory Map  (green=start, red=end)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.imwrite(output_path, canvas)
    print(f"  [+] Trajectory map saved -> {output_path}")


def draw_heatmap(frame_shape, all_trails, output_path):
    """Generate a movement density heatmap."""
    heat = np.zeros(frame_shape[:2], dtype=np.float32)
    for points in all_trails.values():
        for (x, y) in points:
            if 0 <= y < heat.shape[0] and 0 <= x < heat.shape[1]:
                cv2.circle(heat, (x, y), 18, 1, -1)

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=25)
    heat = (heat / (heat.max() + 1e-6) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    cv2.putText(heatmap_color, "Movement Heatmap", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(output_path, heatmap_color)
    print(f"  [+] Heatmap saved -> {output_path}")


def save_count_over_time(count_per_frame, output_path):
    """Save a simple object-count-over-time chart as an image."""
    if not count_per_frame:
        return
    max_count = max(count_per_frame.values()) + 1
    n_frames = max(count_per_frame.keys()) + 1
    h, w = 300, max(600, n_frames)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for frame_idx, count in sorted(count_per_frame.items()):
        x = int(frame_idx * w / n_frames)
        bar_h = int(count * (h - 40) / max_count)
        cv2.line(canvas, (x, h - 20), (x, h - 20 - bar_h), (0, 200, 255), 1)

    cv2.putText(canvas, f"Object Count Over Time  (max={max_count - 1})", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imwrite(output_path, canvas)
    print(f"  [+] Count-over-time chart saved -> {output_path}")


# ─────────────────────────── main pipeline ────────────────────────────────────
def run_pipeline(
    video_path: str,
    output_video: str = "output/annotated_video.mp4",
    model_name: str = "yolov8n.pt",
    tracker_config: str = "botsort.yaml",
    confidence: float = 0.35,
    target_class: int = 0,        # COCO class 0 = person
    trail_length: int = 40,
    skip_frames: int = 0,
):
    """
    End-to-end detection + tracking pipeline.

    Parameters
    ----------
    video_path      : Path to the input video file.
    output_video    : Path for the annotated output video.
    model_name      : YOLOv8 model weights (e.g. yolov8n.pt, yolov8s.pt).
    tracker_config  : Tracker config shipped with ultralytics (botsort.yaml | bytetrack.yaml).
    confidence      : Minimum detection confidence threshold.
    target_class    : COCO class index to track (0 = person).
    trail_length    : Number of past positions to keep for trajectory trails.
    skip_frames     : Process every N-th frame (0 = process all).
    """
    # ── validate input ──
    if not os.path.isfile(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    # ── load model ──
    print(f"\n{'='*60}")
    print("  Multi-Object Detection & Tracking Pipeline")
    print(f"{'='*60}")
    print(f"  Model       : {model_name}")
    print(f"  Tracker     : {tracker_config}")
    print(f"  Confidence  : {confidence}")
    print(f"  Target class: {target_class} (person)")
    print(f"  Input video : {video_path}")
    print(f"  Output video: {output_video}")
    print(f"{'='*60}\n")

    model = YOLO(model_name)
    print("[✓] Model loaded successfully.\n")

    # ── open video ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video info: {w}x{h} @ {fps} FPS, {total_frames} frames\n")

    # ── prepare output ──
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # ── tracking state ──
    trails = defaultdict(list)          # tid -> [(x, y), ...]
    all_trails = defaultdict(list)      # full history for analytics
    count_per_frame = {}
    unique_ids = set()
    frame_idx = 0
    t_start = time.time()

    # ── frame loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # optional frame skipping
        if skip_frames and (frame_idx % (skip_frames + 1) != 1):
            writer.write(frame)
            continue

        # run detection + tracking
        results = model.track(
            frame,
            persist=True,
            conf=confidence,
            tracker=tracker_config,
            classes=[target_class],
            verbose=False,
        )

        boxes, track_ids, confidences = [], [], []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().tolist()
            confidences = r.boxes.conf.cpu().tolist()

            # update trails
            for box, tid in zip(boxes, track_ids):
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                trails[tid].append((cx, cy))
                if len(trails[tid]) > trail_length:
                    trails[tid] = trails[tid][-trail_length:]
                all_trails[tid].append((cx, cy))
                unique_ids.add(tid)

        count_per_frame[frame_idx] = len(track_ids)

        # annotate
        annotated = annotate_frame(frame, boxes, track_ids, confidences, trails, frame_idx)
        writer.write(annotated)

        # progress
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            pct = frame_idx / total_frames * 100
            fps_proc = frame_idx / (elapsed + 1e-6)
            print(f"  [{pct:5.1f}%]  frame {frame_idx}/{total_frames}  "
                  f"({fps_proc:.1f} FPS)  active={len(track_ids)}  "
                  f"unique_ids={len(unique_ids)}")

    cap.release()
    writer.release()
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Total frames processed : {frame_idx}")
    print(f"  Unique IDs assigned    : {len(unique_ids)}")
    print(f"  Output video           : {output_video}")
    print(f"{'='*60}\n")

    # ── generate analytics ──
    output_dir = os.path.dirname(output_video) or "."
    frame_shape = (h, w, 3)
    draw_trajectory_map(frame_shape, all_trails, os.path.join(output_dir, "trajectory_map.png"))
    draw_heatmap(frame_shape, all_trails, os.path.join(output_dir, "heatmap.png"))
    save_count_over_time(count_per_frame, os.path.join(output_dir, "count_over_time.png"))

    # ── save analytics JSON ──
    analytics = {
        "total_frames": frame_idx,
        "unique_ids": len(unique_ids),
        "processing_time_s": round(elapsed, 2),
        "avg_fps": round(frame_idx / (elapsed + 1e-6), 2),
        "model": model_name,
        "tracker": tracker_config,
        "confidence_threshold": confidence,
    }
    analytics_path = os.path.join(output_dir, "analytics.json")
    with open(analytics_path, "w") as f:
        json.dump(analytics, f, indent=2)
    print(f"  [+] Analytics saved -> {analytics_path}")

    # ── extract sample screenshots ──
    _extract_screenshots(video_path=output_video, output_dir=output_dir, n=3)

    print("\n  All done! ✓\n")


def _extract_screenshots(video_path: str, output_dir: str, n: int = 3):
    """Extract N evenly spaced screenshots from the annotated video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return
    step = max(1, total // (n + 1))
    screenshots_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    for i in range(1, n + 1):
        target_frame = step * i
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(screenshots_dir, f"screenshot_{i}.png")
            cv2.imwrite(path, frame)
            print(f"  [+] Screenshot saved -> {path}")
    cap.release()


# ─────────────────────────── CLI entry point ──────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection & Persistent ID Tracking Pipeline"
    )
    parser.add_argument(
        "--video", type=str, default="people-detection.mp4",
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output", type=str, default="output/annotated_video.mp4",
        help="Path for the annotated output video."
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="YOLOv8 model weights (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)."
    )
    parser.add_argument(
        "--tracker", type=str, default="botsort.yaml",
        help="Tracker config: 'botsort.yaml' or 'bytetrack.yaml'."
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection confidence threshold."
    )
    parser.add_argument(
        "--target-class", type=int, default=0,
        help="COCO class index to track (0=person, 32=sports ball, etc.)."
    )
    parser.add_argument(
        "--trail-length", type=int, default=40,
        help="Number of past positions for trajectory trails."
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Process every N-th frame (0 = all frames)."
    )
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_video=args.output,
        model_name=args.model,
        tracker_config=args.tracker,
        confidence=args.conf,
        target_class=args.target_class,
        trail_length=args.trail_length,
        skip_frames=args.skip_frames,
    )


if __name__ == "__main__":
    main()
