#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video people-tracking + congestion, line-cross counting, occupancy overlay
────────────────────────────────────────────────────────────────────────────
* Congestion   : 0-100 % = (min(occupancy / max_people, 1) × 100)
* People count : forward / backward crossings over a user-defined line
* Occupancy    : current # of tracked persons in frame
* Heat-map     : cumulative person locations, shown as minimap (top-right)
* Colours      : Supervision TRACK palette (same as masks / labels)
* Requirements : supervision ≥ 0.21, ultralytics (YOLOE fork) + BoT-SORT
"""
import argparse
import os
import random
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor


# ────────────────────────────── CLI ──────────────────────────────────── #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--source", type=str, default="/DL_data_super_hdd/video_label_sandbox/efg_cargil2025_test1.mp4",
                        help="Input video path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (optional, defaults to input filename without extension)")
    # Logging
    parser.add_argument("--log-detections", type=bool, default=True,
                        help="Log detection results to label.txt")
    # Model
    parser.add_argument("--checkpoint", type=str,
                        default="/works/samsung_prj/pretrain/yoloe-11l-seg.pt",
                        help="YOLOE checkpoint (detection + seg)")
    parser.add_argument("--names", nargs="+",
                        default=["fish", "disco ball", "pig"],
                        help="Custom class names list (index order matters)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Inference device")
    # Inference / tracking
    parser.add_argument("--image-size", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--conf-thresh", type=float, default=0.1,
                        help="Detection confidence threshold")
    parser.add_argument("--vp-thresh", type=float, default=0.1,
                        help="visual prompt confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.45,
                        help="Detection NMS IoU threshold")
    parser.add_argument("--track-history", type=int, default=100,
                        help="Frames kept in trajectory history")
    # Congestion / counting
    parser.add_argument("--max-people", type=int, default=100,
                        help="People count that corresponds to 100 % congestion")
    parser.add_argument("--line-start", nargs=2, type=float,
                        default=None,
                        help="Counting line start (norm. x y, 0~1)")
    parser.add_argument("--line-end", nargs=2, type=float,
                        default=None,
                        help="Counting line end   (norm. x y, 0~1)")
    # Snapshot
    parser.add_argument("--cross-vp", type=bool, default=False,
                        help="Enable cross visual prompt mode")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Interval for saving intermediate frames")
    # Image preprocessing
    parser.add_argument("--sharpen", type=float, default=0.0,
                        help="Image sharpening factor (0.0-1.0)")
    parser.add_argument("--contrast", type=float, default=1.1,
                        help="Image contrast factor (0.5-2.0)")
    parser.add_argument("--brightness", type=float, default=0.0,
                        help="Image brightness adjustment (-50 to 50)")
    parser.add_argument("--denoise", type=float, default=0.1,
                        help="Image denoising strength (0.0-1.0)")
    return parser.parse_args()


# ─────────────────────── Geometry helpers ────────────────────────────── #
def point_side(p, a, b) -> int:
    """
    Returns sign of point p relative to oriented line a->b.
    +1 : left side, -1 : right side, 0 : on line
    Using 2-D cross-product.
    """
    x, y = p
    x1, y1 = a
    x2, y2 = b
    val = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return 0 if val == 0 else (1 if val > 0 else -1)


# ───────────────────────────── Main ───────────────────────────────────── #
def preprocess_image(image, args):
    """이미지 전처리 함수"""
    # 이미지를 float32로 변환
    img = image.astype(np.float32)
    
    # 선명도 향상
    if args.sharpen > 0:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) * args.sharpen
        img = cv2.filter2D(img, -1, kernel)
    
    # 대비 조정
    if args.contrast != 1.0:
        img = cv2.convertScaleAbs(img, alpha=args.contrast, beta=0)
    
    # 밝기 조정
    if args.brightness != 0:
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=args.brightness)
    
    # 노이즈 제거
    if args.denoise > 0:
        img = cv2.fastNlMeansDenoisingColored(img, None, 
                                             h=args.denoise*10, 
                                             hColor=args.denoise*10, 
                                             templateWindowSize=7, 
                                             searchWindowSize=21)
    
    # 값 범위를 0-255로 제한
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def main() -> None:
    args = parse_args()
    # Hybrid VP 파라미터 (기본값 10, 10)
    KEY_INT = getattr(args, 'key_int', 10)
    HOLD_FR = getattr(args, 'hold', 10)
    drift_max = HOLD_FR  # 연속 검출 실패시 리셋
    drift_count = 0

    # 입력 파일 이름에서 확장자를 제외한 기본 이름 추출
    base_name = os.path.splitext(os.path.basename(args.source))[0]
    
    # 출력 디렉토리 설정
    if args.output is None:
        output_dir = os.path.join(os.path.dirname(args.source), base_name)
    else:
        output_dir = args.output
    
    # 기존 출력 디렉토리가 있다면 삭제
    if os.path.exists(output_dir):
        import shutil
        print(f"기존 출력 디렉토리 삭제 중: {output_dir}")
        shutil.rmtree(output_dir)
    
    # 출력 디렉토리 생성
    print(f"새로운 출력 디렉토리 생성: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 출력 비디오 경로 설정
    output_video = os.path.join(output_dir, f"{base_name}_output.mp4")

    # 로그 파일 설정
    log_file = None
    if args.log_detections:
        log_path = os.path.join(output_dir, "label.txt")
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write("Frame_Time,Class,TrackID,Class_Count\n")

    # ─────────────── I/O setup ──────────────── #
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {args.source}")

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select frames to process
    frames_to_process = range(total_frames)

    # ─────────────── Model & palette ─────────── #
    model = YOLOE(args.checkpoint)
    model.eval()
    model.to(args.device)
    model.set_classes(args.names, model.get_text_pe(args.names))

    # VP 모델 별도 로드
    model_vp = YOLOE(args.checkpoint)
    model_vp.eval()
    model_vp.to(args.device)
    model_vp.set_classes(args.names, model_vp.get_text_pe(args.names))

    person_class_id = args.names.index("fish") if "fish" in args.names else 0
    palette = sv.ColorPalette.DEFAULT

    # ─────────────── Counting line (pixel) ───── #
    if args.line_start is not None and args.line_end is not None:
        p1 = (int(args.line_end[0]   * width),  int(args.line_end[1]   * height))
        p2 = (int(args.line_start[0] * width),  int(args.line_start[1] * height))
        
        # 선분 벡터 및 길이² 계산
        seg_dx, seg_dy = p2[0] - p1[0], p2[1] - p1[1]
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy or 1
    else:
        p1 = p2 = None
        seg_dx = seg_dy = seg_len2 = 0

    # ─────────────── Runtime state ───────────── #
    track_history = defaultdict(list)
    track_side   = {}
    track_color  = {}
    forward_cnt = backward_cnt = 0

    # 트래킹 라인 관련 변수
    max_track_history = 30  # 최대 트래킹 히스토리 길이
    line_thickness = 2     # 라인 두께
    line_opacity = 0.5     # 라인 투명도

    # --- 프롬프트 반복 로직용 상태 변수 --- #
    prev_prompt = None
    prompt_frame = None
    last_vp_frame = None
    last_vp_results = None
    
    # VPE moving average 상태 변수
    vpe_avg = None
    vpe_alpha = 0.9  # moving average 계수 (0.9 = 이전 값의 90% + 현재 값의 10%)

    # Heat-map 누적 버퍼
    heatmap = np.zeros((height, width), dtype=np.float32)

    # 라인 깜박임 상태
    line_flash = False

    # 1초마다 갱신되는 값들
    last_update_time = 0
    current_occupancy = 0
    current_congestion = 0
    frame_count = 0

    # 중간 저장 관련 변수
    last_save_frame = 0
    log_buffer = []  # 로그 버퍼

    # 비디오 출력 설정
    out = cv2.VideoWriter(output_video,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (width, height))

    pbar = tqdm(total=len(frames_to_process), desc="Processing")

    # ────────────────── Frame loop ───────────────── #
    for frame_idx in frames_to_process:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_count += 1
        current_time = frame_idx / fps  # 실제 영상의 시간 계산

        # 현재 프레임 시간 계산 (시간:분:초:밀리초)
        hours = int(current_time // 3600)
        minutes = int((current_time % 3600) // 60)
        seconds = int(current_time % 60)
        milliseconds = int((current_time % 1) * 1000)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 이미지 전처리 적용
        frame_rgb = preprocess_image(frame_rgb, args)
        
        image_pil = Image.fromarray(frame_rgb)

        # ── Detection (track 대신 predict만 사용) ── #
        if prev_prompt is not None and prompt_frame is not None and args.cross_vp:
            # 이전 프롬프트로 Cross-Image VP
            model_vp.predictor = YOLOEVPSegPredictor
            model_vp.predictor = None
            model_vp.predict(
                source=prompt_frame,
                imgsz=args.image_size,
                conf=args.vp_thresh,
                iou=args.iou_thresh,
                return_vpe=True,
                prompts=prev_prompt,
                predictor=YOLOEVPSegPredictor
            )
            if hasattr(model_vp, 'predictor') and hasattr(model_vp.predictor, 'vpe'):
                current_vpe = model_vp.predictor.vpe
                if vpe_avg is None:
                    vpe_avg = current_vpe
                else:
                    vpe_avg = vpe_alpha * vpe_avg + (1 - vpe_alpha) * current_vpe
                model_vp.set_classes(args.names, vpe_avg)
                model_vp.predictor = None
            else:
                print("[DEBUG] VPE 생성 실패!")

            results = model_vp.predict(
                source=image_pil,
                imgsz=args.image_size,
                conf=args.conf_thresh,
                iou=args.iou_thresh,
                verbose=False
            )
            result = results[0] if isinstance(results, list) else results
        else:
            results = model.predict(
                source=image_pil,
                imgsz=args.image_size,
                conf=0.05,
                iou=args.iou_thresh,
                verbose=False
            )
            result = results[0] if isinstance(results, list) else results

        # 1차 프롬프트 갱신
        if len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            high_conf_mask = confidences >= args.vp_thresh
            if np.any(high_conf_mask):
                # 각 box에 해당하는 class ID 사용
                class_ids = result.boxes.cls.cpu().numpy()[high_conf_mask]
                prev_prompt = {
                    "bboxes": [boxes[high_conf_mask]],
                    "cls": [class_ids]
                }
                
                # 모든 클래스가 포함되도록 추가
                for cls_idx in range(len(args.names)):
                    if cls_idx not in class_ids:
                        prev_prompt["bboxes"][0] = np.append(prev_prompt["bboxes"][0], [[0, 0, 0, 0]], axis=0)
                        prev_prompt["cls"][0] = np.append(prev_prompt["cls"][0], [cls_idx])
                
                prompt_frame = image_pil.copy()
            else:
                prev_prompt = None
                prompt_frame = None
        else:
            prev_prompt = None
            prompt_frame = None

        # Convert results to supervision format
        detections = sv.Detections.from_ultralytics(result)
        
        # Add tracker_id to detections if available
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            unique_ids = np.unique(track_ids)
            id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
            detections.tracker_id = np.array([id_map[tid] for tid in track_ids])
        else:
            detections.tracker_id = np.array([0] * len(detections))

        boxes_xywh = result.boxes.xywh.cpu().numpy() if len(detections) else np.empty((0, 4))
        class_ids  = result.boxes.cls.cpu().numpy().astype(int) if len(detections) else np.empty(0, int)
        track_ids  = detections.tracker_id

        # 로그 파일에 검출 정보 기록
        if log_file is not None:
            class_counts = defaultdict(int)
            for cid in class_ids:
                class_counts[cid] += 1

            # 클래스별 검출 숫자 출력
            print(f"Frame {frame_idx} - Class counts:")
            for cid, count in class_counts.items():
                print(f"  {args.names[cid]}: {count}")

            for cid, tid in zip(class_ids, track_ids):
                class_name = args.names[cid]
                class_count = class_counts[cid]
                log_entry = f"{time_str},{class_name},{tid},{class_count}\n"
                if log_file is not None:
                    log_buffer.append(log_entry)
                    if len(log_buffer) >= 100:
                        log_file.writelines(log_buffer)
                        log_file.flush()
                        log_buffer.clear()

        annotated = frame_rgb.copy()
        if len(detections):
            mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.4)
            label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS,
                                         smart_position=True,
                                         text_scale=sv.calculate_optimal_text_scale(resolution_wh=image_pil.size))
            annotated = mask_annot.annotate(annotated, detections)
            box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
            annotated = box_annot.annotate(annotated, detections)
            annotated = label_annot.annotate(annotated, detections,
                                          labels=[f"{n} {c:.2f}" for n, c in zip(detections['class_name'],
                                                                                 detections.confidence)])

        # ── People occupancy - 1초마다 갱신
        raw_occupancy = int(np.sum(class_ids == person_class_id))

        if current_time - last_update_time >= 1.0:
            last_update_time = current_time
            current_occupancy = raw_occupancy
            current_congestion = int(min(current_occupancy / max(args.max_people, 1), 1.0) * 100)

        line_crossed_this_frame = False

        for (cx, cy, w, h), cid, tid in zip(boxes_xywh, class_ids, track_ids):
            if tid == -1 or cid != person_class_id:
                continue

            ix, iy = int(cx), int(cy)
            if 0 <= iy < height and 0 <= ix < width:
                radius = 10
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius*radius:
                            ny, nx = iy + dy, ix + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                intensity = 1.0 * (1.0 - ((dx*dx + dy*dy) / (radius*radius)))
                                heatmap[ny, nx] += intensity

            if tid not in track_color:
                track_color[tid] = palette.by_idx(tid).as_bgr()

            if p1 is not None and p2 is not None:
                side_now = point_side((cx, cy), p1, p2)
                t = ((cx - p1[0]) * seg_dx + (cy - p1[1]) * seg_dy) / seg_len2
                inside = 0.0 <= t <= 1.0

                side_prev = track_side.get(tid, side_now)
                if inside and (side_prev * side_now < 0):
                    line_crossed_this_frame = True
                    if side_prev < 0 < side_now:
                        forward_cnt += 1
                    elif side_prev > 0 > side_now:
                        backward_cnt += 1
                if inside and side_now != 0:
                    track_side[tid] = side_now

        if p1 is not None and p2 is not None:
            if line_crossed_this_frame:
                cv2.line(annotated, p1, p2, (255, 0, 0), 8)
            else:
                cv2.line(annotated, p1, p2, (255, 255, 255), 8)

            mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            perp_x, perp_y = -dy, dx
            length = 50
            mag = (perp_x**2 + perp_y**2) ** 0.5 or 1
            perp_x, perp_y = int(perp_x / mag * length), int(perp_y / mag * length)
            arrow_start, arrow_end = (mid_x, mid_y), (mid_x + perp_x, mid_y + perp_y)
            cv2.arrowedLine(annotated, arrow_start, arrow_end, (0, 255, 0), 8, tipLength=0.6)

        overlay = [
            f"[1] Congestion : {current_congestion:3d} %",
            f"[2] Crossing   : forward {forward_cnt} | backward {backward_cnt}",
            f"[3] Occupancy  : {current_occupancy}",
            f"[4] Time      : {time_str}"
        ]
        x0, y0, dy = 30, 60, 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        padding = 10
        max_width = 0
        for txt in overlay:
            (tw, th), _ = cv2.getTextSize(txt, font, font_scale, font_thickness)
            max_width = max(max_width, tw)
        total_height = th * len(overlay) + dy * (len(overlay) - 1)
        rect_x1, rect_y1 = x0 - padding, y0 - th - padding
        rect_x2, rect_y2 = x0 + max_width + padding, y0 + (len(overlay) - 1) * dy + padding
        overlay_rect = annotated.copy()
        cv2.rectangle(overlay_rect, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay_rect, 0.4, annotated, 0.6, 0)
        for i, txt in enumerate(overlay):
            cv2.putText(annotated, txt, (x0, y0 + i * dy),
                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        blur = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_hm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        color_hm = cv2.cvtColor(color_hm, cv2.COLOR_BGR2RGB)

        mini_w = 400
        mini_h = int(height * mini_w / width)
        mini_map = cv2.resize(color_hm, (mini_w, mini_h))

        margin = 20
        x_start = width - mini_w - margin
        y_start = margin
        roi = annotated[y_start:y_start + mini_h, x_start:x_start + mini_w]
        blended = cv2.addWeighted(mini_map, 0.6, roi, 0.4, 0)
        annotated[y_start:y_start + mini_h, x_start:x_start + mini_w] = blended

        if frame_idx - last_save_frame >= args.save_interval:
            snapshot_path = os.path.join(output_dir, f"intermediate_{frame_idx:06d}_{time_str.replace(':', '-')}.jpg")
            cv2.imwrite(snapshot_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            last_save_frame = frame_idx
            print(f"\n중간 저장 완료: {snapshot_path}")

        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        pbar.update(1)

    pbar.close()
    cap.release()
    if log_file is not None:
        if log_buffer:
            log_file.writelines(log_buffer)
        log_file.close()
    out.release()
    print(f"✔ 완료. 저장 위치: {output_dir}")


if __name__ == "__main__":
    main() 
