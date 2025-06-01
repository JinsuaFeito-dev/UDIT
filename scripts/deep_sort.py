#!/usr/bin/env python3
"""
yolo_deepsort_runN_skip1.py

YOLO + DeepSORT with custom frame skipping:
Runs YOLO for N frames, then skips 1 (e.g. skip_frames=4 → run 4 frames, skip 1)

Usage:
  python yolo_deepsort_runN_skip1.py \
    --yolo_model ./yolo.pt \
    --video_path ./input.mp4 \
    --output_path ./output.mp4 \
    --conf_thres 0.3 \
    --max_age 30 \
    --n_init 2 \
    --max_cosine_distance 0.2 \
    --skip_frames 4 \
    --display
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--conf_thres", type=float, default=0.3)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=2)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--skip_frames", type=int, default=1e10)
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def main():
    args = parse_args()

    # Load YOLO model
    yolo = YOLO(args.yolo_model)

    # Initialize DeepSORT
    tracker = DeepSort(
        max_age=args.max_age,
        n_init=1,  # confirm tracks immediately, use external logic
        max_cosine_distance=args.max_cosine_distance
    )

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    # Output video writer
    writer = None
    if args.output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output_path, fourcc, fps, (w, h))

    frame_idx = 0
    detection_counts = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run N frames, skip 1
        N = args.skip_frames 
        cycle_len = N + 1
        run_detection = (frame_idx % cycle_len) < N

        if run_detection:
            results = yolo(frame)
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            keep = confs >= args.conf_thres
            boxes_xyxy = boxes_xyxy[keep]
            confs = confs[keep]
            class_ids = class_ids[keep]

            detections = []
            for box, conf, cls in zip(boxes_xyxy, confs, class_ids):
                xywh = xyxy_to_xywh(box)
                detections.append((xywh, float(conf), int(cls)))

            tracks = tracker.update_tracks(detections, frame=frame)

            for t in tracks:
                if t.is_confirmed() and t.time_since_update == 0:
                    tid = t.track_id
                    detection_counts[tid] = detection_counts.get(tid, 0) + 1

        else:
            # Predict only
            tracker.tracker.predict()
            tracks = tracker.tracker.tracks

        # Draw tracks that have enough detections
        for t in tracks:
            tid = t.track_id
            if detection_counts.get(tid, 0) < args.n_init:
                continue

            l, t_, r, b = map(int, t.to_ltrb())
            cv2.rectangle(frame, (l, t_), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid}", (l, t_ - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if writer:
            writer.write(frame)
        if args.display:
            cv2.imshow("YOLO + DeepSORT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
