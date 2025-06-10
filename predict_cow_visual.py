#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-sequence people detection + segmentation with Cross-Image Visual Prompting (VPE)
Adapted directly from video-based pipeline: replaces frame loop with image list loop
────────────────────────────────────────────────────────────────────────────
* Processes single images, directories, or glob patterns as "frames"
* Applies YOLOE detection + segmentation
* Builds and accumulates VPE across images exactly as for video frames
* Draws masks, boxes, and labels on each image
* No tracking, no occupancy/congestion/counting
* Requirements: supervision ≥ 0.21, ultralytics (YOLOE fork)
"""
import argparse
import os
import glob
import cv2
import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-sequence detection+VPE")
    parser.add_argument("--source", type=str, default="/DL_data_super_ssd/new_EFID2023/yolo_oadseg_dataset/250605cowNEW/test_images/day",
                        help="Image file, directory, or glob (e.g. ./images/*.jpg)")
    parser.add_argument("--output", type=str, default="/works/yoloe/output/day",
                        help="Directory to save annotated images")
    parser.add_argument("--checkpoint", type=str, default="pretrain/yoloe-11l-seg.pt",
                        help="YOLOE checkpoint path")
    parser.add_argument("--names", nargs='+', default=["cow"],
                        help="Class names in index order")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Inference device")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--conf-thresh", type=float, default=0.2, ## 추론 임계치
                        help="Detection confidence threshold")
    parser.add_argument("--vp-thresh", type=float, default=0.4, ## vpe updated 여부 임계치
                        help="VPE generation confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--no-vpe", type=bool, default=False,
                        help="Disable Cross-Image Visual Prompting; use only text prompt")
    return parser.parse_args()

def get_image_paths(src: str):
    if os.path.isdir(src):
        exts = ("*.jpg","*.jpeg","*.png","*.bmp")
        files = []
        for e in exts:
            files += glob.glob(os.path.join(src, e))
        return sorted(files)
    if '*' in src:
        return sorted(glob.glob(src))
    return [src]

def preprocess_image(image: np.ndarray) -> np.ndarray:
    # identical to video pipeline; override if needed
    return image

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ─────────────── Model & VP setup ──────────────── #
    base_model = YOLOE(args.checkpoint)
    base_model.eval().to(args.device)
    base_model.set_classes(args.names, base_model.get_text_pe(args.names))

    model_vp = YOLOE(args.checkpoint)
    model_vp.eval().to(args.device)
    model_vp.set_classes(args.names, model_vp.get_text_pe(args.names))

    # VPE state
    prev_prompt = None
    vpe_avg = None
    vpe_alpha = 0.7

    image_paths = get_image_paths(args.source)
    total = len(image_paths)

    for idx, path in enumerate(image_paths, 1):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"Warning: cannot read {path}, skipping.")
            continue
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_proc = preprocess_image(img_rgb)
        img_pil  = Image.fromarray(img_proc)

        # ── Detection (video 코드 그대로) ── #
        if not args.no_vpe and prev_prompt is not None:
            # 이전 프롬프트로 Cross-Image VP
            model_vp.predictor = YOLOEVPSegPredictor    # ➊ 클래스 지정
            model_vp.predictor = None                  # ➋ 초기화
            model_vp.predict(                           # ➌ VPE 생성
                source=prompt_frame,
                imgsz=args.image_size,
                conf=args.vp_thresh,
                iou=args.iou_thresh,
                return_vpe=True,
                prompts=prev_prompt,
                predictor=YOLOEVPSegPredictor
            )
            # 생성된 predictor에서 vpe 추출
            current_vpe = model_vp.predictor.vpe
            if vpe_avg is None:
                vpe_avg = current_vpe
            else:
                vpe_avg = vpe_alpha * vpe_avg + (1 - vpe_alpha) * current_vpe
            model_vp.set_classes(args.names, vpe_avg) # ➍ 누적 반영
            model_vp.predictor = None                 # ➍ predictor 클리어

            # ➎ VPE 반영된 모델로 최종 추론
            results = model_vp.predict(
                source=img_pil,
                imgsz=args.image_size,
                conf=args.conf_thresh,
                iou=args.iou_thresh,
                verbose=False
            )
            res = results[0]
        else:
            # 첫 이미지 or 프롬프트 없을 땐 기본 모델
            results = base_model.predict(
                source=img_pil,
                imgsz=args.image_size,
                conf=args.conf_thresh,
                iou=args.iou_thresh,
                verbose=False
            )
            res = results[0]

        # ── 프롬프트 갱신 ── #
        if len(res.boxes):
            confidences = res.boxes.conf.cpu().numpy()
            boxes       = res.boxes.xyxy.cpu().numpy()
            cls_ids     = res.boxes.cls.cpu().numpy().astype(int)
            mask = confidences >= args.vp_thresh
            if mask.any():
                prev_prompt  = {"bboxes":[boxes[mask]], "cls":[cls_ids[mask]]}
                prompt_frame = img_pil.copy()
            else:
                prev_prompt  = None
                prompt_frame = None
        else:
            prev_prompt  = None
            prompt_frame = None

        # ── 시각화 & 저장 ── #
        dets = sv.Detections.from_ultralytics(res)
        out  = img_proc.copy()
        out  = sv.MaskAnnotator(opacity=0.4).annotate(out, dets)
        out  = sv.BoxAnnotator().annotate(out, dets)
        names  = [args.names[c] for c in dets.class_id]
        labels = [f"{n} {conf:.2f}" for n,conf in zip(names, dets.confidence)]
        out  = sv.LabelAnnotator().annotate(out, dets, labels=labels)

        save_path = os.path.join(args.output, os.path.basename(path))
        cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"[{idx}/{total}] Saved: {save_path}")

if __name__ == "__main__":
    main()
