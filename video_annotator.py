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
import torch

# ────────────────────────────── Tracker ──────────────────────────────────── #
from tracker.boostTrack.GBI import GBInterpolation
from tracker.boostTrack.boost_track import BoostTrack


# ────────────────────────────── ObjectMeta Class ──────────────────────────── #
class ObjectMeta:
    """
    개체의 모든 메타데이터를 담는 클래스
    Detection부터 Tracking, Overlay까지 일관되게 사용
    """
    def __init__(self, box=None, mask=None, confidence=None, class_id=None, 
                 class_name=None, track_id=None):
        self.box = box              # [x1, y1, x2, y2] or [cx, cy, w, h]
        self.mask = mask            # segmentation mask array
        self.confidence = confidence # detection confidence score
        self.class_id = class_id    # class index
        self.class_name = class_name # class name string
        self.track_id = track_id    # tracking ID (None initially)
    
    def __repr__(self):
        return f"ObjectMeta(class={self.class_name}, track_id={self.track_id}, conf={self.confidence:.2f})"


# ────────────────────────────── CLI ──────────────────────────────────── #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--source", type=str, default="/DL_data_super_hdd/video_label_sandbox/Heungchuk_videos/day.mp4",
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
                        default=["cow", "cattle"],
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
    parser.add_argument("--track-det-thresh", type=float, default=0.1,
                        help="Detection confidence threshold for tracking")
    parser.add_argument("--track-iou-thresh", type=float, default=0.3,
                        help="Tracking IoU threshold")
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
    parser.add_argument("--cross-vp", type=bool, default=True,
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


# ─────────────────────── ObjectMeta Conversion ────────────────────────────── #
def convert_results_to_objects(result, class_names) -> list[ObjectMeta]:
    """
    YOLO 결과를 ObjectMeta 리스트로 변환 (Supervision 방식의 마스크 처리)
    """
    objects = []
    
    if len(result.boxes) == 0:
        return objects
    
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    
    # 마스크 정보 (있을 경우) - 정교한 변환 로직 사용
    masks = None
    if hasattr(result, 'masks') and result.masks is not None:
        try:
            # Supervision 방식: masks.xy 사용 (이미 원본 좌표계로 변환됨)
            if hasattr(result.masks, 'xy') and result.masks.xy is not None:
                # masks.xy는 이미 원본 이미지 좌표계로 변환된 polygon 좌표들
                orig_height, orig_width = result.orig_shape
                masks_list = []
                
                for mask_coords in result.masks.xy:
                    # polygon을 마스크로 변환
                    mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
                    if len(mask_coords) > 0:
                        # polygon 좌표를 integer로 변환
                        coords = mask_coords.astype(np.int32)
                        # fillPoly로 마스크 생성
                        cv2.fillPoly(mask, [coords], 1)
                    masks_list.append(mask)
                
                masks = np.array(masks_list)
                
            # xy가 없으면 data를 사용하되 더 정교한 변환 적용
            elif hasattr(result.masks, 'data'):
                orig_height, orig_width = result.orig_shape
                mask_data = result.masks.data.cpu().numpy()  # [N, H, W]
                
                # YOLO의 이미지 전처리 정보 계산
                input_height, input_width = mask_data.shape[1], mask_data.shape[2]
                
                # aspect ratio 계산
                scale = min(input_width / orig_width, input_height / orig_height)
                scaled_width = int(orig_width * scale)
                scaled_height = int(orig_height * scale)
                
                # 패딩 계산
                pad_x = (input_width - scaled_width) // 2
                pad_y = (input_height - scaled_height) // 2
                
                masks_list = []
                for mask in mask_data:
                    # 패딩 제거
                    if pad_y > 0 and pad_x > 0:
                        mask = mask[pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width]
                    
                    # 원본 크기로 리사이즈
                    resized_mask = cv2.resize(
                        mask.astype(np.float32), 
                        (orig_width, orig_height), 
                        interpolation=cv2.INTER_LINEAR  # 더 부드러운 보간
                    )
                    masks_list.append(resized_mask)
                
                masks = np.array(masks_list)
                
        except Exception as e:
            print(f"⚠️ 마스크 변환 오류: {e}")
            masks = None
    
    for i in range(len(boxes)):
        obj = ObjectMeta(
            box=boxes[i],
            mask=masks[i] if masks is not None else None,
            confidence=confidences[i],
            class_id=class_ids[i],
            class_name=class_names[class_ids[i]],
            track_id=None  # 트래커에서 부여받을 예정
        )
        objects.append(obj)
    
    return objects


# ─────────────────────── Direct Overlay Functions ────────────────────────────── #
def draw_box(image, box, track_id, class_id, color_palette):
    """박스 그리기"""
    x1, y1, x2, y2 = map(int, box)
    color = color_palette.by_idx(track_id).as_bgr()
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

def draw_mask(image, mask, track_id, class_id, color_palette, opacity=0.4):
    """Supervision 방식의 마스크 그리기 (정교한 크기 변환 적용)"""
    if mask is None:
        return image
    
    color = color_palette.by_idx(track_id).as_bgr()
    
    # 마스크 크기 검증 및 상세 디버깅
    if mask.shape[:2] != image.shape[:2]:
        print(f"🔍 마스크 크기 디버깅:")
        print(f"   Image: {image.shape[:2]} (H, W)")
        print(f"   Mask:  {mask.shape[:2]} (H, W)")
        print(f"   Track: {track_id}, Class: {class_id}")
        
        # 정확한 리사이즈
        mask = cv2.resize(
            mask.astype(np.float32), 
            (image.shape[1], image.shape[0]),  # (width, height) 순서 주의
            interpolation=cv2.INTER_LINEAR
        )
        print(f"   Resized: {mask.shape[:2]}")
    
    # boolean 마스크로 변환 (threshold 조정)
    mask = mask > 0.3  # threshold를 낮춰서 더 많은 영역 포함
    
    # 마스크 영역이 있는지 확인
    mask_area = np.sum(mask)
    if mask_area == 0:
        print(f"⚠️ 빈 마스크: track_id={track_id}")
        return image
    
    # Supervision 방식: colored_mask 생성 후 addWeighted 사용
    colored_mask = np.array(image, copy=True, dtype=np.uint8)
    colored_mask[mask] = color  # 마스크 영역에만 색상 적용
    
    # addWeighted로 블렌딩 (원본 이미지에 직접 적용)
    cv2.addWeighted(colored_mask, opacity, image, 1 - opacity, 0, dst=image)
    
    return image

def draw_label(image, box, track_id, class_name, confidence, color_palette):
    """라벨 그리기"""
    x1, y1, x2, y2 = map(int, box)
    color = color_palette.by_idx(track_id).as_bgr()
    
    # 라벨 텍스트
    label = f"{class_name} {track_id} {confidence:.2f}"
    
    # 텍스트 크기 계산
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # 라벨 배경 그리기
    cv2.rectangle(image, (x1, y1 - text_h - baseline - 10), 
                  (x1 + text_w, y1), color, -1)
    
    # 텍스트 그리기
    cv2.putText(image, label, (x1, y1 - baseline - 5), 
                font, font_scale, (255, 255, 255), thickness)
    
    return image

def draw_objects_overlay(image, objects, color_palette):
    """모든 개체의 overlay 그리기 (Supervision 방식: 큰 객체부터 작은 객체 순)"""
    # area 계산하여 큰 것부터 정렬 (Supervision과 동일)
    valid_objects = [obj for obj in objects if obj.track_id is not None]
    
    if not valid_objects:
        return image
    
    # 박스 area 계산
    areas = []
    for obj in valid_objects:
        x1, y1, x2, y2 = obj.box
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
    
    # area 기준으로 큰 것부터 정렬 (flip으로 내림차순)
    sorted_indices = np.flip(np.argsort(areas))
    
    # 마스크부터 먼저 그리기 (supervision 방식)
    for idx in sorted_indices:
        obj = valid_objects[idx]
        # 마스크 그리기
        image = draw_mask(image, obj.mask, obj.track_id, obj.class_id, color_palette)
    
    # 그 다음 박스와 라벨 그리기 (원래 순서대로)
    for obj in valid_objects:
        # 박스 그리기
        image = draw_box(image, obj.box, obj.track_id, obj.class_id, color_palette)
        
        # 라벨 그리기
        image = draw_label(image, obj.box, obj.track_id, obj.class_name, 
                          obj.confidence, color_palette)
    
    return image


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
    
    # ─────────────── Tracker ─────────── #
    tracker = BoostTrack(
        video_name=args.source,
        det_thresh=args.track_det_thresh,
        iou_threshold=args.track_iou_thresh
    )

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

        # ── ObjectMeta 중심 처리 시작 ── #
        # 1. Model 결과를 ObjectMeta 리스트로 변환
        detected_objects = convert_results_to_objects(result, args.names)
        
        # 2. 새로운 ObjectMeta 기반 트래커 사용
        tracked_objects = tracker.update(detected_objects, result.orig_img, "None")
        
        # 3. 기존 변수들을 ObjectMeta에서 추출 (기존 로직 호환성을 위해)
        if tracked_objects:
            # tracked_objects에서 xywh 형태로 변환 (기존 로직 호환)
            boxes_xywh = []
            for obj in tracked_objects:
                x1, y1, x2, y2 = obj.box  # xyxy
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                boxes_xywh.append([cx, cy, w, h])
            boxes_xywh = np.array(boxes_xywh)
            
            class_ids = np.array([obj.class_id for obj in tracked_objects])
            track_ids = np.array([obj.track_id for obj in tracked_objects])
        else:
            boxes_xywh = np.empty((0, 4))
            class_ids = np.empty(0, int)
            track_ids = np.empty(0, int)

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

        # ── 새로운 ObjectMeta 기반 Overlay 시스템 ── #
        annotated = frame_rgb.copy()
        if tracked_objects:
            # 직접 구현한 overlay 함수 사용
            annotated = draw_objects_overlay(annotated, tracked_objects, palette)

        # ── People occupancy - 1초마다 갱신
        raw_occupancy = int(np.sum(class_ids == person_class_id))

        if current_time - last_update_time >= 1.0:
            last_update_time = current_time
            current_occupancy = raw_occupancy
            current_congestion = int(min(current_occupancy / max(args.max_people, 1), 1.0) * 100)

        line_crossed_this_frame = False

        # ── ObjectMeta 기반 히트맵 및 라인 크로싱 처리 ── #
        for obj in tracked_objects:
            if obj.track_id == -1 or obj.class_id != person_class_id:
                continue

            # 박스 중심점 계산 (xyxy → center)
            x1, y1, x2, y2 = obj.box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ix, iy = int(cx), int(cy)
            
            # 히트맵 업데이트
            if 0 <= iy < height and 0 <= ix < width:
                radius = 10
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius*radius:
                            ny, nx = iy + dy, ix + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                intensity = 1.0 * (1.0 - ((dx*dx + dy*dy) / (radius*radius)))
                                heatmap[ny, nx] += intensity

            # 트래킹 컬러 설정
            if obj.track_id not in track_color:
                track_color[obj.track_id] = palette.by_idx(obj.track_id).as_bgr()

            # 라인 크로싱 체크
            if p1 is not None and p2 is not None:
                side_now = point_side((cx, cy), p1, p2)
                t = ((cx - p1[0]) * seg_dx + (cy - p1[1]) * seg_dy) / seg_len2
                inside = 0.0 <= t <= 1.0

                side_prev = track_side.get(obj.track_id, side_now)
                if inside and (side_prev * side_now < 0):
                    line_crossed_this_frame = True
                    if side_prev < 0 < side_now:
                        forward_cnt += 1
                    elif side_prev > 0 > side_now:
                        backward_cnt += 1
                if inside and side_now != 0:
                    track_side[obj.track_id] = side_now

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
