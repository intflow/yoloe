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
import threading
import queue
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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

# ─────────────────────── ROI (Region of Interest) Utilities ────────────────────────────── #
import json
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import ast


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
# ANCHOR argparse
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # I/O
    # parser.add_argument("--source", type=str, default="/DL_data_super_hdd/video_label_sandbox/efg_cargil2025_test1.mp4",
    parser.add_argument("--source", type=str, default="../10s_test.mp4",
                        help="Input video path")
    parser.add_argument("--output", type=str, default="output",
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
    parser.add_argument("--track-det-thresh", type=float, default=0.1,
                        help="Detection confidence threshold for tracking")
    parser.add_argument("--track-iou-thresh", type=float, default=0.3,
                        help="Tracking IoU threshold")
    # Congestion / counting
    parser.add_argument("--max-people", type=int, default=100,
                        help="People count that corresponds to 100 percent congestion")
    parser.add_argument("--line-start", nargs=2, type=float,
                        default=None,
                        help="Counting line start (norm. x y, 0~1)")
    parser.add_argument("--line-end", nargs=2, type=float,
                        default=None,
                        help="Counting line end   (norm. x y, 0~1)")
    # Snapshot
    parser.add_argument("--cross-vp", type=bool, default=True,
                        help="Enable cross visual prompt mode")
    parser.add_argument("--save-interval", type=int, default=150,
                        help="Interval for saving intermediate frames")
    # NOTE [args] Batch processing
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference processing")
    parser.add_argument("--frame-loading-threads", type=int, default=32,
                        help="Number of threads for frame loading")
    parser.add_argument("--vpe-momentum", type=float, default=0.1,
                        help="VPE moving average momentum")    
    # Image preprocessing
    parser.add_argument("--sharpen", type=float, default=0.0,
                        help="Image sharpening factor (0.0-1.0)")
    parser.add_argument("--contrast", type=float, default=1.1,
                        help="Image contrast factor (0.5-2.0)")
    parser.add_argument("--brightness", type=float, default=0.0,
                        help="Image brightness adjustment (-50 to 50)")
    parser.add_argument("--denoise", type=float, default=0.1,
                        help="Image denoising strength (0.0-1.0)")
    # NOTE [args] ROI Access Detection
    parser.add_argument("--roi-zones", type=str, 
                        # default=None,
                        default="[[214,392,286,392,290,478,214,478],[864,387,945,383,947,465,869,468],[478,14,735,7,736,225,491,221]]",
                        help="ROI zones as nested list of pixel coordinates")
    parser.add_argument("--roi-names", type=str, 
                        # default=None,
                        default="water_area1,water_area2,feeding_area",
                        help="ROI zone names separated by comma")
    parser.add_argument("--roi-detection-method", choices=['bbox', 'mask', 'hybrid'], 
                        default='bbox', help="ROI detection method: bbox overlap, mask overlap, or hybrid")
    parser.add_argument("--roi-dwell-time", type=float, default=0.5,
                        help="Minimum dwell time in seconds for ROI access detection")
    parser.add_argument("--roi-bbox-threshold", type=float, default=0.1,
                        help="Minimum bbox overlap ratio for ROI detection")
    parser.add_argument("--roi-mask-threshold", type=float, default=0.1,
                        help="Minimum mask overlap ratio for ROI detection")
    parser.add_argument("--roi-exit-grace-time", type=float, default=2.0,
                        help="Grace time in seconds before removing objects that left ROI")
    # Output resolution
    parser.add_argument("--output-width", type=int, default=1920,
                        help="Output video width (if not set, uses input width)")
    parser.add_argument("--output-height", type=int, default=1080,
                        help="Output video height (if not set, uses input height)")
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


def parse_roi_zones(roi_zones_str):
    """ROI zones 문자열을 파싱하여 polygon 리스트로 변환"""
    if not roi_zones_str:
        return []
    
    try:
        # 문자열을 리스트로 파싱
        zones_data = ast.literal_eval(roi_zones_str)
        
        polygons = []
        for zone_coords in zones_data:
            if len(zone_coords) < 6 or len(zone_coords) % 2 != 0:
                print(f"⚠️ ROI zone 좌표가 잘못됨: {zone_coords} (최소 3개 점 필요)")
                continue
            
            # [x1,y1,x2,y2,x3,y3,...] → [(x1,y1), (x2,y2), (x3,y3), ...]
            points = [(zone_coords[i], zone_coords[i+1]) for i in range(0, len(zone_coords), 2)]
            
            try:
                polygon = Polygon(points)
                if polygon.is_valid:
                    polygons.append(polygon)
                else:
                    print(f"⚠️ 유효하지 않은 polygon: {points}")
            except Exception as e:
                print(f"⚠️ Polygon 생성 실패: {points}, 오류: {e}")
        
        return polygons
    
    except Exception as e:
        print(f"💥 ROI zones 파싱 오류: {e}")
        return []

def parse_roi_names(roi_names_str, num_zones):
    """ROI names 문자열을 파싱하여 이름 리스트로 변환"""
    if not roi_names_str:
        # 이름이 없으면 자동으로 Zone_0, Zone_1, ... 생성
        return [f"Zone_{i}" for i in range(num_zones)]
    
    names = [name.strip() for name in roi_names_str.split(',')]
    
    # 이름 개수가 부족하면 자동 생성으로 채움
    while len(names) < num_zones:
        names.append(f"Zone_{len(names)}")
    
    return names[:num_zones]  # 초과하는 이름은 제거

def calculate_bbox_polygon_overlap(bbox, polygon):
    """bbox와 polygon의 겹침 비율 계산"""
    try:
        x1, y1, x2, y2 = bbox
        bbox_polygon = box(x1, y1, x2, y2)
        
        if not bbox_polygon.is_valid or not polygon.is_valid:
            return 0.0
        
        intersection = bbox_polygon.intersection(polygon)
        if intersection.is_empty:
            return 0.0
        
        bbox_area = bbox_polygon.area
        if bbox_area == 0:
            return 0.0
        
        overlap_ratio = intersection.area / bbox_area
        return overlap_ratio
    
    except Exception as e:
        print(f"⚠️ bbox-polygon 겹침 계산 오류: {e}")
        return 0.0

def calculate_mask_polygon_overlap(mask, polygon, image_shape):
    """mask와 polygon의 겹침 비율 계산"""
    try:
        if mask is None:
            return 0.0
        
        height, width = image_shape[:2]
        
        # mask를 이미지 크기에 맞게 조정
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        
        # mask를 boolean으로 변환
        mask_bool = mask > 0.5
        
        # polygon을 mask로 변환
        polygon_coords = np.array(polygon.exterior.coords, dtype=np.int32)
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon_coords], 1)
        polygon_bool = polygon_mask.astype(bool)
        
        # 교집합 계산
        intersection = np.logical_and(mask_bool, polygon_bool)
        intersection_area = np.sum(intersection)
        
        # mask 전체 면적
        mask_area = np.sum(mask_bool)
        if mask_area == 0:
            return 0.0
        
        overlap_ratio = intersection_area / mask_area
        return overlap_ratio
    
    except Exception as e:
        print(f"⚠️ mask-polygon 겹침 계산 오류: {e}")
        return 0.0

def check_roi_access(obj, polygon, method, bbox_threshold, mask_threshold, image_shape):
    """객체가 ROI에 접근했는지 확인"""
    if method == 'bbox':
        overlap_ratio = calculate_bbox_polygon_overlap(obj.box, polygon)
        return overlap_ratio >= bbox_threshold
    
    elif method == 'mask':
        if obj.mask is None:
            # mask가 없으면 bbox로 fallback
            overlap_ratio = calculate_bbox_polygon_overlap(obj.box, polygon)
            return overlap_ratio >= bbox_threshold
        else:
            overlap_ratio = calculate_mask_polygon_overlap(obj.mask, polygon, image_shape)
            return overlap_ratio >= mask_threshold
    
    elif method == 'hybrid':
        # 1단계: bbox 빠른 필터링
        bbox_overlap = calculate_bbox_polygon_overlap(obj.box, polygon)
        if bbox_overlap < bbox_threshold:
            return False
        
        # 2단계: mask 정밀 검사 (bbox 겹침이 있을 때만)
        if obj.mask is not None:
            mask_overlap = calculate_mask_polygon_overlap(obj.mask, polygon, image_shape)
            return mask_overlap >= mask_threshold
        else:
            # mask가 없으면 bbox 결과 사용
            return True
    
    return False


# ─────────────────────── ROI Access Manager Class ────────────────────────────── #
class ROIAccessManager:
    """ROI 접근 감지 및 통계 관리 클래스"""
    
    def __init__(self, roi_polygons, roi_names, detection_method, dwell_time, 
                 bbox_threshold, mask_threshold, fps, exit_grace_time):
        self.roi_polygons = roi_polygons
        self.roi_names = roi_names
        self.detection_method = detection_method
        self.dwell_time = dwell_time
        self.bbox_threshold = bbox_threshold
        self.mask_threshold = mask_threshold
        self.fps = fps
        self.exit_grace_time = exit_grace_time
        
        # 접근에 필요한 프레임 수 계산
        self.required_frames = int(dwell_time * fps)
        # 퇴장 유예 프레임 수 계산
        self.exit_grace_frames = int(exit_grace_time * fps)
        
        # 각 ROI별 통계
        self.roi_stats = {}
        for i, name in enumerate(roi_names):
            self.roi_stats[name] = {
                'total_access': 0,
                'accessed_labels': set(),
                'track_access_count': {},  # {track_id: access_count}
                'current_tracks': {},  # {track_id: {'enter_frame': frame, 'label': label, 'consecutive_frames': count}}
                'exit_pending_tracks': {}  # {track_id: {'exit_frame': frame, 'grace_frames_left': count}}
            }
        
        print(f"\n🎯 ROI Access Manager 초기화:")
        print(f"   - ROI 개수: {len(roi_polygons)}")
        print(f"   - ROI 이름: {roi_names}")
        print(f"   - 감지 방법: {detection_method}")
        print(f"   - 체류 시간: {dwell_time}초 ({self.required_frames} 프레임)")
        print(f"   - 퇴장 유예 시간: {exit_grace_time}초 ({self.exit_grace_frames} 프레임)")
        print(f"   - bbox 임계값: {bbox_threshold}")
        print(f"   - mask 임계값: {mask_threshold}")
    
    def update(self, tracked_objects, frame_idx, image_shape):
        """매 프레임마다 ROI 접근 상태 업데이트"""
        current_frame_tracks = set()
        
        for roi_idx, (polygon, roi_name) in enumerate(zip(self.roi_polygons, self.roi_names)):
            roi_stat = self.roi_stats[roi_name]
            current_roi_tracks = set()
            
            # 현재 프레임에서 이 ROI에 있는 객체들 확인
            for obj in tracked_objects:
                if obj.track_id is None or obj.track_id == -1:
                    continue
                
                # ROI 접근 확인
                is_in_roi = check_roi_access(
                    obj, polygon, self.detection_method,
                    self.bbox_threshold, self.mask_threshold, image_shape
                )
                
                if is_in_roi:
                    current_roi_tracks.add(obj.track_id)
                    current_frame_tracks.add(obj.track_id)
                    
                    if obj.track_id in roi_stat['current_tracks']:
                        # 이미 ROI에 있던 객체 → 연속 프레임 수 증가
                        roi_stat['current_tracks'][obj.track_id]['consecutive_frames'] += 1
                        
                        # 체류 시간 조건 만족 시 접근으로 카운트
                        if (roi_stat['current_tracks'][obj.track_id]['consecutive_frames'] >= self.required_frames and
                            not roi_stat['current_tracks'][obj.track_id].get('counted', False)):
                            
                            # 접근 카운트 증가
                            roi_stat['total_access'] += 1
                            roi_stat['accessed_labels'].add(obj.class_name)
                            
                            if obj.track_id not in roi_stat['track_access_count']:
                                roi_stat['track_access_count'][obj.track_id] = 0
                            roi_stat['track_access_count'][obj.track_id] += 1
                            
                            # 중복 카운트 방지
                            roi_stat['current_tracks'][obj.track_id]['counted'] = True
                            
                            print(f"🎯 ROI 접근 감지: {roi_name} - track_id:{obj.track_id} ({obj.class_name})")
                    
                    else:
                        # 새로 ROI에 진입한 객체
                        roi_stat['current_tracks'][obj.track_id] = {
                            'enter_frame': frame_idx,
                            'label': obj.class_name,
                            'consecutive_frames': 1,
                            'counted': False
                        }
            
            # ROI에서 나간 객체들 처리 (유예 시간 적용)
            tracks_to_remove = []
            tracks_to_exit_pending = []
            
            # 1. 현재 ROI에 없는 객체들 확인
            for track_id in roi_stat['current_tracks']:
                if track_id not in current_roi_tracks:
                    # ROI에서 나간 객체 → 퇴장 대기 상태로 이동
                    if track_id not in roi_stat['exit_pending_tracks']:
                        tracks_to_exit_pending.append(track_id)
            
            # 2. 퇴장 대기 상태로 이동
            for track_id in tracks_to_exit_pending:
                roi_stat['exit_pending_tracks'][track_id] = {
                    'exit_frame': frame_idx,
                    'grace_frames_left': self.exit_grace_frames
                }
                # current_tracks에서는 아직 제거하지 않음 (유예 기간 동안 유지)
            
            # 3. 퇴장 대기 중인 객체들의 유예 시간 감소
            exit_pending_to_remove = []
            for track_id in list(roi_stat['exit_pending_tracks'].keys()):
                if track_id in current_roi_tracks:
                    # 다시 ROI에 들어온 경우 → 퇴장 대기 취소
                    del roi_stat['exit_pending_tracks'][track_id]
                    # print(f"🔄 ROI 재진입: {roi_name} - track_id:{track_id}")
                else:
                    # 여전히 ROI 밖에 있는 경우 → 유예 시간 감소
                    roi_stat['exit_pending_tracks'][track_id]['grace_frames_left'] -= 1
                    
                    if roi_stat['exit_pending_tracks'][track_id]['grace_frames_left'] <= 0:
                        # 유예 시간 만료 → 완전 제거
                        exit_pending_to_remove.append(track_id)
                        tracks_to_remove.append(track_id)
            
            # 4. 유예 시간이 만료된 객체들 완전 제거
            for track_id in exit_pending_to_remove:
                del roi_stat['exit_pending_tracks'][track_id]
            
            for track_id in tracks_to_remove:
                if track_id in roi_stat['current_tracks']:
                    del roi_stat['current_tracks'][track_id]
                    # print(f"🚪 ROI 완전 퇴장: {roi_name} - track_id:{track_id} (유예 시간 만료)")
    
    def get_current_roi_tracks(self):
        """현재 각 ROI에 있는 track_id들 반환 (퇴장 대기 중인 객체 포함)"""
        current_tracks = {}
        for roi_name, roi_stat in self.roi_stats.items():
            # 현재 ROI에 있는 객체들 + 퇴장 대기 중인 객체들 (유예 기간 동안은 여전히 ROI에 있는 것으로 간주)
            active_tracks = set(roi_stat['current_tracks'].keys())
            # 퇴장 대기 중인 객체들도 포함 (시각화 목적)
            pending_tracks = set(roi_stat['exit_pending_tracks'].keys())
            all_tracks = active_tracks.union(pending_tracks)
            current_tracks[roi_name] = list(all_tracks)
        return current_tracks
    
    def get_statistics(self):
        """최종 통계 반환"""
        stats = {}
        for roi_name, roi_stat in self.roi_stats.items():
            stats[roi_name] = {
                'total_access': roi_stat['total_access'],
                'accessed_labels': list(roi_stat['accessed_labels']),
                'track_access_count': dict(roi_stat['track_access_count'])
            }
        return stats
    
    def save_statistics(self, output_path):
        """통계를 JSON 파일로 저장"""
        stats = self.get_statistics()
        
        # 추가 메타데이터
        metadata = {
            'roi_detection_method': self.detection_method,
            'dwell_time_seconds': self.dwell_time,
            'required_frames': self.required_frames,
            'exit_grace_time_seconds': self.exit_grace_time,
            'exit_grace_frames': self.exit_grace_frames,
            'bbox_threshold': self.bbox_threshold,
            'mask_threshold': self.mask_threshold,
            'roi_names': self.roi_names
        }
        
        result = {
            'metadata': metadata,
            'roi_statistics': stats
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"📊 ROI 통계 저장 완료: {output_path}")
        except Exception as e:
            print(f"💥 ROI 통계 저장 실패: {e}")
    
    def print_final_statistics(self):
        """최종 통계를 콘솔에 출력"""
        print(f"\n🎯 ROI 접근 감지 최종 통계:")
        print(f"{'='*50}")
        
        for roi_name, roi_stat in self.roi_stats.items():
            print(f"\n📍 {roi_name}: {roi_stat['total_access']}회 접근")
            
            if roi_stat['accessed_labels']:
                print(f"   접근한 클래스: {', '.join(sorted(roi_stat['accessed_labels']))}")
            
            if roi_stat['track_access_count']:
                print(f"   Track ID별 접근 횟수:")
                for track_id, count in sorted(roi_stat['track_access_count'].items()):
                    print(f"     - Track {track_id}: {count}회")
            
            if not roi_stat['total_access']:
                print(f"   접근 기록 없음")
        
        print(f"{'='*50}")


# ─────────────────────── ROI Visualization Functions ────────────────────────────── #
def draw_roi_polygons(image, roi_polygons, roi_names, roi_stats):
    """ROI polygon들을 이미지에 그리기"""
    for polygon, roi_name in zip(roi_polygons, roi_names):
        try:
            # polygon 좌표 추출
            coords = np.array(polygon.exterior.coords, dtype=np.int32)
            
            # 빨간색 테두리로 polygon 그리기
            cv2.polylines(image, [coords], isClosed=True, color=(255, 0, 0), thickness=6)
            
            # ROI 이름과 접근 횟수 표시
            if roi_name in roi_stats:
                total_access = roi_stats[roi_name]['total_access']
                
                # polygon의 중심점 계산
                centroid = polygon.centroid
                text_x, text_y = int(centroid.x), int(centroid.y)
                
                # 텍스트 배경
                text = f"{roi_name}: {total_access}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # 배경 사각형
                cv2.rectangle(image, (text_x - 5, text_y - text_h - 10), 
                             (text_x + text_w + 5, text_y + 5), (255, 0, 0), -1)
                
                # 텍스트
                cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        except Exception as e:
            print(f"⚠️ ROI polygon 그리기 오류: {e}")
    
    return image

def highlight_roi_objects(image, tracked_objects, current_roi_tracks, color_palette):
    """현재 ROI에 있는 객체들을 하이라이트"""
    # 모든 ROI에 있는 track_id들 수집
    all_roi_track_ids = set()
    for roi_tracks in current_roi_tracks.values():
        all_roi_track_ids.update(roi_tracks)
    
    # ROI에 있는 객체들에 빨간색 하이라이트 추가
    for obj in tracked_objects:
        if obj.track_id in all_roi_track_ids:
            x1, y1, x2, y2 = map(int, obj.box)
            
            # 빨간색 굵은 테두리 추가
            cv2.rectangle(image, (x1-2, y1-2), (x2+2, y2+2), (255, 0, 0), 4)
    
    return image


# ─────────────────────── ObjectMeta Conversion ────────────────────────────── #
def convert_results_to_objects(cpu_result, class_names) -> list[ObjectMeta]:
    """
    CPU로 변환된 YOLO 결과를 ObjectMeta 리스트로 변환
    """
    objects = []
    
    if not cpu_result['has_boxes']:
        return objects
    
    boxes = cpu_result['boxes_xyxy']  # 이미 CPU numpy array
    confidences = cpu_result['boxes_conf']  # 이미 CPU numpy array
    class_ids = cpu_result['boxes_cls']  # 이미 CPU numpy array
    
    # 마스크 정보 처리 (정교한 변환 로직 사용)
    masks = None
    if cpu_result['masks_type'] is not None:
        try:
            orig_height, orig_width = cpu_result['orig_shape']
            
            if cpu_result['masks_type'] == 'xy':
                # masks.xy 사용 (이미 원본 좌표계로 변환됨)
                masks_list = []
                
                for mask_coords in cpu_result['masks_xy']:
                    # polygon을 마스크로 변환
                    mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
                    if len(mask_coords) > 0:
                        # polygon 좌표를 integer로 변환
                        coords = mask_coords.astype(np.int32)
                        # fillPoly로 마스크 생성
                        cv2.fillPoly(mask, [coords], 1)
                    masks_list.append(mask)
                
                masks = np.array(masks_list)
                
            elif cpu_result['masks_type'] == 'data':
                # masks.data 사용
                mask_data = cpu_result['masks_data']  # 이미 CPU numpy array
                
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


# ─────────────────────── Inference Thread Class ────────────────────────────── #
class InferenceThread(threading.Thread):
    """
    🧠 GPU 추론 및 VPE 업데이트를 전담하는 스레드
    Frame Loading Thread → Inference Thread → Main Thread 파이프라인의 중간 단계
    """
    
    def __init__(self, frame_queue, result_queue, model, model_vp, args):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model = model
        self.model_vp = model_vp
        self.args = args
        self.stop_event = threading.Event()
        self.prev_vpe = None  # VPE 상태 관리
        self.stats = {
            'total_batches_processed': 0,
            'total_frames_processed': 0,
            'vpe_updates': 0,
            'failed_batches': 0
        }
        
        print(f"\n🧠 Inference Thread 초기화:")
        print(f"   - GPU 디바이스: {args.device}")
        print(f"   - Cross-VP 모드: {'활성화' if args.cross_vp else '비활성화'}")
        print(f"   - VPE 모멘텀: {args.vpe_momentum}")
    
    def stop(self):
        """스레드 종료 요청"""
        self.stop_event.set()
        print("🛑 Inference Thread 종료 요청됨")
    
    # ANCHOR Batch Inference    
    def run(self):
        """메인 GPU 추론 루프"""
        print("\n🚀 Inference Thread 시작!")
        
        try:
            batch_idx = 0
            while not self.stop_event.is_set():
                try:
                    # Frame Queue에서 배치 데이터 가져오기 (타임아웃 3초)
                    batch_data = self.frame_queue.get(timeout=3.0)
                    
                    # 종료 신호 확인
                    if batch_data is None:
                        print("🏁 Inference Thread: Frame Loading 종료 신호 수신")
                        # Main Thread에도 종료 신호 전달
                        self.result_queue.put(None)
                        break
                    
                    # 배치 데이터 언패킹
                    batch_frames = batch_data['batch_frames']
                    batch_indices = batch_data['batch_indices']
                    batch_original_frames = batch_data['batch_original_frames']
                    loaded_count = batch_data['loaded_count']
                    
                    if not batch_frames:
                        print("⚠️ Inference Thread: 빈 batch 수신, 건너뛰기")
                        continue
                    
                    # print(f"🧠 Batch {batch_idx + 1} GPU 추론 시작: {loaded_count} 프레임")
                    
                    # ──────── GPU INFERENCE ──────── #
                    batch_results = self._inference_batch(batch_frames)
                    
                    if batch_results is None:
                        print(f"❌ Batch {batch_idx + 1} 추론 실패")
                        self.stats['failed_batches'] += 1
                        continue
                    
                    # ──────── VPE UPDATE ──────── #
                    vpe_updated = False
                    if self.args.cross_vp:
                        vpe_updated = self._update_vpe(batch_results)
                    
                    # 결과 패키징
                    result_data = {
                        'batch_results': batch_results,
                        'batch_indices': batch_indices,
                        'batch_original_frames': batch_original_frames,
                        'loaded_count': loaded_count,
                        'batch_idx': batch_idx,
                        'vpe_updated': vpe_updated,
                        'prev_vpe_status': self.prev_vpe is not None
                    }
                    
                    # Result Queue에 전달
                    self.result_queue.put(result_data)
                    
                    # 통계 업데이트
                    self.stats['total_batches_processed'] += 1
                    self.stats['total_frames_processed'] += loaded_count
                    if vpe_updated:
                        self.stats['vpe_updates'] += 1
                    
                    # print(f"✅ Batch {batch_idx + 1} GPU 처리 완료: {loaded_count} 프레임 → Result Queue")
                    batch_idx += 1
                    
                except queue.Empty:
                    # Frame Queue가 비어있으면 잠시 대기
                    continue
                    
                except Exception as e:
                    print(f"💥 Inference Thread 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stats['failed_batches'] += 1
                    
        except Exception as e:
            print(f"💥 Inference Thread 치명적 오류: {e}")
        finally:
            # 🚀 GPU 메모리 완전 정리
            self.model_vp.predictor = None
            if hasattr(self, 'prev_vpe') and self.prev_vpe is not None:
                del self.prev_vpe
                self.prev_vpe = None
            
            import torch
            torch.cuda.empty_cache()
            
            # 종료 신호를 Result Queue에 반드시 전달 (Queue가 빌 때까지 대기)
            print("📤 Inference Thread 종료 신호 전송 준비 중...")
            queue_full_warning_shown = False
            while True:
                try:
                    self.result_queue.put(None, timeout=1.0)
                    print("🏁 Inference Thread 종료 신호 Result Queue에 전달 완료!")
                    break
                except queue.Full:
                    if not queue_full_warning_shown:
                        print("⏳ Result Queue 가득함, 종료 신호 전송을 위해 대기 중...")
                        queue_full_warning_shown = True
                    time.sleep(0.1)
            
            print(f"📈 Inference Thread 최종 통계:")
            print(f"   - 처리된 배치 수: {self.stats['total_batches_processed']}")
            print(f"   - 처리된 프레임 수: {self.stats['total_frames_processed']}")
            print(f"   - VPE 업데이트 횟수: {self.stats['vpe_updates']}")
            print(f"   - 실패한 배치 수: {self.stats['failed_batches']}")
            print("🚀 GPU 메모리 완전 정리 완료!")
    
    def _inference_batch(self, batch_frames):
        """배치 추론 실행 및 GPU → CPU 변환"""
        try:
            batch_size = len(batch_frames)
            
            if self.prev_vpe is not None and self.args.cross_vp:
                # VPE 모드로 추론
                self.model_vp.set_classes(self.args.names, self.prev_vpe)
                self.model_vp.predictor = None
                
                results = self.model_vp.predict(
                    source=batch_frames,
                    imgsz=self.args.image_size,
                    conf=self.args.conf_thresh,
                    iou=self.args.iou_thresh,
                    verbose=False
                )
            else:
                # 기본 모드로 추론
                results = self.model.predict(
                    source=batch_frames,
                    imgsz=self.args.image_size,
                    conf=0.05,
                    iou=self.args.iou_thresh,
                    verbose=False
                )
            
            # 🔥 GPU → CPU 변환을 여기서 수행하여 메모리 효율성 확보!
            cpu_results = []
            for result in results:
                cpu_result = {
                    'boxes_xyxy': result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else np.empty((0, 4)),
                    'boxes_conf': result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else np.empty(0),
                    'boxes_cls': result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) > 0 else np.empty(0, dtype=int),
                    'orig_img': result.orig_img,
                    'orig_shape': result.orig_shape,
                    'has_boxes': len(result.boxes) > 0
                }
                
                # 마스크 정보 처리 (있을 경우)
                if hasattr(result, 'masks') and result.masks is not None:
                    try:
                        # masks.xy 사용 (이미 원본 좌표계로 변환됨)
                        if hasattr(result.masks, 'xy') and result.masks.xy is not None:
                            masks_xy_list = []
                            for mask_coords in result.masks.xy:
                                # GPU tensor인지 numpy array인지 확인 후 변환
                                if hasattr(mask_coords, 'cpu'):
                                    masks_xy_list.append(mask_coords.cpu().numpy())
                                else:
                                    masks_xy_list.append(mask_coords)  # 이미 numpy array
                            cpu_result['masks_xy'] = masks_xy_list
                            cpu_result['masks_type'] = 'xy'
                        # xy가 없으면 data 사용
                        elif hasattr(result.masks, 'data'):
                            # GPU tensor인지 numpy array인지 확인 후 변환
                            if hasattr(result.masks.data, 'cpu'):
                                cpu_result['masks_data'] = result.masks.data.cpu().numpy()
                            else:
                                cpu_result['masks_data'] = result.masks.data  # 이미 numpy array
                            cpu_result['masks_type'] = 'data'
                        else:
                            cpu_result['masks_type'] = None
                    except Exception as e:
                        print(f"⚠️ 마스크 CPU 변환 오류: {e}")
                        cpu_result['masks_type'] = None
                else:
                    cpu_result['masks_type'] = None
                
                cpu_results.append(cpu_result)
            
            # 🚀 GPU 메모리 즉시 정리
            del results
            import torch
            torch.cuda.empty_cache()
            
            return cpu_results
            
        except Exception as e:
            print(f"💥 배치 추론 오류: {e}")
            # GPU 메모리 정리
            import torch
            torch.cuda.empty_cache()
            return None
    
    def _update_vpe(self, batch_results):
        """VPE 업데이트 (기존 로직 활용)"""
        try:
            current_vpe = self._update_batch_vpe(batch_results)
            
            if current_vpe is not None:
                if self.prev_vpe is None:
                    # 첫 번째 VPE
                    self.prev_vpe = current_vpe
                    # print(f"🎯 첫 번째 VPE 설정 완료")
                else:
                    # VPE Moving Average
                    momentum = self.args.vpe_momentum
                    self.prev_vpe = momentum * self.prev_vpe + (1 - momentum) * current_vpe
                    # print(f"🔄 VPE Moving Average 업데이트 완료")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"💥 VPE 업데이트 오류: {e}")
            return False
    
    def _update_batch_vpe(self, batch_results):
        """배치 결과에서 VPE 생성 (CPU 데이터 기반)"""
        high_conf_prompts = []
        
        # Batch 내 모든 프레임에서 high-confidence detection 수집
        for i, cpu_result in enumerate(batch_results):
            if cpu_result['has_boxes']:
                confidences = cpu_result['boxes_conf']  # 이미 CPU numpy array
                boxes = cpu_result['boxes_xyxy']  # 이미 CPU numpy array
                class_ids = cpu_result['boxes_cls']  # 이미 CPU numpy array
                
                high_conf_mask = confidences >= self.args.vp_thresh
                
                if np.any(high_conf_mask):
                    # High confidence detection을 프롬프트로 사용
                    prompt_data = {
                        "bboxes": boxes[high_conf_mask],
                        "cls": class_ids[high_conf_mask],
                        "frame": cpu_result['orig_img'],  # 원본 이미지 정보
                        "frame_idx": i
                    }
                    
                    # 모든 클래스가 포함되도록 추가
                    for cls_idx in range(len(self.args.names)):
                        if cls_idx not in class_ids:
                            prompt_data["bboxes"] = np.append(prompt_data["bboxes"], [[0, 0, 0, 0]], axis=0)
                            prompt_data["cls"] = np.append(prompt_data["cls"], [cls_idx])                
                            
                    high_conf_prompts.append(prompt_data)
                    
        if high_conf_prompts:
            # Batch 내 프롬프트들로 VPE 생성
            batch_vpe = self._generate_batch_vpe(high_conf_prompts)
            return batch_vpe
        else:
            return None
    
    def _generate_batch_vpe(self, prompts_list):
        """여러 프롬프트에서 VPE를 배치로 생성 후 평균화 (기존 함수 로직 활용)"""
        
        # 배치용 이미지와 프롬프트 준비
        batch_images = []
        batch_prompts = {"bboxes": [], "cls": []}
        
        for i, prompt_data in enumerate(prompts_list):
            # 이미지 준비
            frame_rgb = cv2.cvtColor(prompt_data["frame"], cv2.COLOR_BGR2RGB)
            batch_images.append(Image.fromarray(frame_rgb))
            
            # 프롬프트 준비
            batch_prompts["bboxes"].append(prompt_data["bboxes"])
            batch_prompts["cls"].append(prompt_data["cls"])
        
        try:
            # 배치 VPE 생성
            self.model_vp.predictor = None  # VPE 생성 전 초기화
            self.model_vp.predict(
                source=batch_images,
                prompts=batch_prompts,
                predictor=YOLOEVPSegPredictor,
                return_vpe=True,
                imgsz=self.args.image_size,
                conf=self.args.vp_thresh,
                iou=self.args.iou_thresh,
                verbose=False
            )
            
            # VPE 추출
            if hasattr(self.model_vp, 'predictor') and hasattr(self.model_vp.predictor, 'vpe'):
                batch_vpe = self.model_vp.predictor.vpe
                
                # 배치 차원을 평균화하여 단일 VPE로 변환
                if len(batch_vpe.shape) > 2:  # (batch_size, classes, features) 형태인 경우
                    averaged_vpe = batch_vpe.mean(dim=0, keepdim=True)  # (1, classes, features)
                else:
                    averaged_vpe = batch_vpe
                
                # 🚀 예측기 정리 및 GPU 메모리 해제
                self.model_vp.predictor = None
                del batch_vpe
                import torch
                torch.cuda.empty_cache()
                
                return averaged_vpe
            else:
                self.model_vp.predictor = None
                import torch
                torch.cuda.empty_cache()
                return None
                
        except Exception as e:
            print(f"💥 배치 VPE 생성 중 오류: {e}")
            self.model_vp.predictor = None
            import torch
            torch.cuda.empty_cache()
            return None


# ─────────────────────── Frame Loading Thread Class ────────────────────────────── #
class FrameLoadingThread(threading.Thread):
    """
    🚀 Producer-Consumer 패턴으로 개선된 프레임 로딩 스레드
    
    구조:
    1. VideoCapture는 초기화 시 한 번만 생성
    2. 내부 읽기 스레드가 지속적으로 프레임 읽기
    3. 프레임 버퍼에 frame_id와 함께 저장
    4. 배치 구성 시 버퍼에서 꺼내서 사용 (즉시 삭제)
    5. 최대 버퍼 크기로 메모리 제어
    """
    
    def __init__(self, video_source, total_frames, batch_size, args, batch_queue, 
                 max_queue_size=3, max_buffer_size=200):
        super().__init__(daemon=True)
        self.video_source = video_source
        self.total_frames = total_frames
        self.batch_size = batch_size
        self.args = args
        self.batch_queue = batch_queue
        self.max_queue_size = max_queue_size
        self.max_buffer_size = max_buffer_size  # 프레임 버퍼 최대 크기
        
        # 상태 관리
        self.current_frame = 0
        self.stop_event = threading.Event()
        self.queue_full_print = False
        
        # 🎯 핵심: 프레임 버퍼 (Producer-Consumer)
        self.frame_buffer = {}  # {frame_id: frame_data}
        self.buffer_lock = threading.Lock()
        self.buffer_not_full = threading.Condition(self.buffer_lock)
        self.frames_available = threading.Condition(self.buffer_lock)
        
        # 통계
        self.stats = {
            'total_batches_loaded': 0,
            'total_frames_loaded': 0,
            'failed_frames': 0,
            'buffer_waits': 0,
            'max_buffer_used': 0
        }
        
        # VideoCapture 초기화 (한 번만!)
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"VideoCapture 열기 실패: {self.video_source}")
        
        # 내부 프레임 읽기 스레드 생성
        self.reader_thread = threading.Thread(target=self._continuous_frame_reader, daemon=True)
        self.reader_started = False
        
        print(f"\n🎬 Frame Loading Thread 초기화:")
        print(f"   - 총 프레임: {total_frames}")
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - Queue 최대 크기: {max_queue_size}")
        print(f"   - 프레임 버퍼 최대 크기: {max_buffer_size}")
        print(f"   - VideoCapture: 초기화 완료 ✅")
    
    def stop(self):
        """스레드 종료 요청"""
        self.stop_event.set()
        # 모든 대기 중인 스레드 깨우기
        with self.buffer_lock:
            self.buffer_not_full.notify_all()
            self.frames_available.notify_all()
        print("🛑 Frame Loading Thread 종료 요청됨")
    
    def _continuous_frame_reader(self):
        """
        🔄 Producer: 지속적으로 프레임을 읽어서 버퍼에 저장
        별도 스레드에서 실행되며 VideoCapture를 독점 사용
        """
        print("📖 Start Continuous Frame Reader")
        
        frame_idx = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        try:
            while not self.stop_event.is_set() and frame_idx < self.total_frames:
                # 버퍼 크기 체크 및 대기
                with self.buffer_lock:
                    while (len(self.frame_buffer) >= self.max_buffer_size and 
                           not self.stop_event.is_set()):
                        self.stats['buffer_waits'] += 1
                        # print(f"⏳ 프레임 버퍼 가득참 ({len(self.frame_buffer)}/{self.max_buffer_size}), 대기 중...")
                        self.buffer_not_full.wait(timeout=1.0)
                
                if self.stop_event.is_set():
                    break
                
                # 프레임 읽기
                ok, frame_bgr = self.cap.read()
                
                if ok:
                    # 성공적으로 읽음
                    consecutive_failures = 0
                    
                    # 버퍼에 저장
                    with self.buffer_lock:
                        self.frame_buffer[frame_idx] = {
                            'frame': frame_bgr.copy(),
                            'frame_idx': frame_idx
                        }
                        
                        # 통계 업데이트
                        current_buffer_size = len(self.frame_buffer)
                        self.stats['max_buffer_used'] = max(self.stats['max_buffer_used'], 
                                                          current_buffer_size)
                        
                        # Consumer에게 프레임 준비됨 알림
                        self.frames_available.notify_all()
                    
                    frame_idx += 1
                    
                    # # 주기적 상태 출력 (1000 프레임마다)
                    # if frame_idx % 1000 == 0:
                    #     with self.buffer_lock:
                    #         buffer_size = len(self.frame_buffer)
                    #     print(f"📖 프레임 읽기 진행: {frame_idx}/{self.total_frames} "
                    #           f"(버퍼: {buffer_size}/{self.max_buffer_size})")
                
                else:
                    # 읽기 실패
                    consecutive_failures += 1
                    self.stats['failed_frames'] += 1
                    
                    # if consecutive_failures >= max_consecutive_failures:
                    #     print(f"💥 연속 {consecutive_failures}회 프레임 읽기 실패, 중단")
                    #     break
                    
                    print(f"⚠️ 프레임 {frame_idx} 읽기 실패 ({consecutive_failures}/{max_consecutive_failures})")
                    frame_idx += 1  # 다음 프레임으로 계속
                
        except Exception as e:
            print(f"💥 연속 프레임 읽기 오류: {e}")
        finally:
            print(f"📖 연속 프레임 읽기 종료: {frame_idx} 프레임 처리 완료")
            # Consumer들에게 종료 알림
            with self.buffer_lock:
                self.frames_available.notify_all()
    
    def _load_batch_frames(self, start_frame, end_frame):
        """
        🛒 Consumer: 버퍼에서 필요한 프레임들을 꺼내서 배치 구성
        사용된 프레임은 즉시 삭제하여 메모리 효율성 확보
        """
        batch_size = end_frame - start_frame
        raw_frames = {}
        
        # 1단계: 버퍼에서 필요한 프레임들 수집
        for frame_idx in range(start_frame, end_frame):
            frame_data = None
            max_wait_attempts = 50  # 최대 5초 대기 (100ms × 50)
            
            for attempt in range(max_wait_attempts):
                with self.buffer_lock:
                    if frame_idx in self.frame_buffer:
                        # 프레임 찾음 → 즉시 꺼내서 삭제 (핵심!)
                        frame_data = self.frame_buffer.pop(frame_idx)
                        
                        # Producer에게 버퍼 공간 생김 알림
                        self.buffer_not_full.notify()
                        break
                    else:
                        # 프레임 아직 없음 → 잠시 대기
                        self.frames_available.wait(timeout=0.1)
                
                if self.stop_event.is_set():
                    break
            
            if frame_data is not None:
                raw_frames[frame_idx] = frame_data['frame']
            else:
                print(f"⚠️ 프레임 {frame_idx} 버퍼에서 찾을 수 없음 (타임아웃)")
                self.stats['failed_frames'] += 1
        
        if not raw_frames:
            print("💥 버퍼에서 읽은 프레임이 없음")
            return None
        
        # print(f"   🛒 버퍼에서 수집: {len(raw_frames)}/{batch_size} 프레임")
        
        # 2단계: 병렬 전처리 (기존 로직 유지)
        processed_frames = {}
        lock = threading.Lock()
        
        def preprocess_single_frame(frame_idx, frame_bgr):
            """단일 프레임 전처리 (스레드에서 실행)"""
            try:
                # RGB 변환 및 전처리 (CPU 집약적)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = preprocess_image(frame_rgb, self.args)
                
                # PIL Image로 변환
                pil_frame = Image.fromarray(frame_rgb)
                
                # 스레드 세이프하게 저장
                with lock:
                    processed_frames[frame_idx] = {
                        'pil_frame': pil_frame,
                        'original_frame': frame_bgr,
                        'success': True
                    }
                    
            except Exception as e:
                with lock:
                    processed_frames[frame_idx] = {'success': False, 'error': str(e)}
                print(f"💥 프레임 {frame_idx} 전처리 실패: {e}")
        
        # 병렬 전처리 실행
        max_workers = min(self.args.frame_loading_threads, len(raw_frames))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(preprocess_single_frame, frame_idx, frame_bgr) 
                for frame_idx, frame_bgr in raw_frames.items()
            ]
            concurrent.futures.wait(futures)
        
        # 3단계: 결과 정리 (인덱스 순서대로)
        batch_frames = []
        batch_indices = []
        batch_original_frames = []
        
        loaded_count = 0
        for frame_idx in range(start_frame, end_frame):
            if frame_idx in processed_frames and processed_frames[frame_idx]['success']:
                batch_frames.append(processed_frames[frame_idx]['pil_frame'])
                batch_indices.append(frame_idx)
                batch_original_frames.append(processed_frames[frame_idx]['original_frame'])
                loaded_count += 1
        
        if loaded_count == 0:
            return None
        
        return {
            'batch_frames': batch_frames,
            'batch_indices': batch_indices,
            'batch_original_frames': batch_original_frames,
            'loaded_count': loaded_count,
            'total_count': batch_size
        }
    
    # ANCHOR Frame Loading Thread Run
    def run(self):
        """메인 배치 로딩 루프 (Consumer)"""
        print("\n🚀 Frame Loading Thread 시작!")
        
        # 내부 프레임 읽기 스레드 시작
        if not self.reader_started:
            self.reader_thread.start()
            self.reader_started = True
        
        try:
            while not self.stop_event.is_set() and self.current_frame < self.total_frames:
                # Queue가 가득 찬 경우 대기
                if self.batch_queue.qsize() >= self.max_queue_size:
                    if not self.queue_full_print:
                        # print(f"⏳ Queue 가득참 ({self.batch_queue.qsize()}/{self.max_queue_size}), 잠시 대기...")
                        self.queue_full_print = True
                    time.sleep(0.1)
                    continue
                
                # 배치 프레임 로드
                batch_start = self.current_frame
                batch_end = min(batch_start + self.batch_size, self.total_frames)
                
                batch_data = self._load_batch_frames(batch_start, batch_end)
                
                if batch_data is not None:
                    # Queue에 배치 데이터 저장
                    self.batch_queue.put(batch_data, timeout=5.0)
                    self.stats['total_batches_loaded'] += 1
                    self.stats['total_frames_loaded'] += len(batch_data['batch_indices'])
                    
                    # print(f"✅ 배치 로딩 완료: {len(batch_data['batch_indices'])} 프레임 → Queue에 저장")
                else:
                    print(f"❌ 배치 로딩 실패: 프레임 {batch_start}-{batch_end-1}")
                
                self.current_frame = batch_end
                self.queue_full_print = False
                
            print(f"🔄 현재 프레임: {self.current_frame}/{self.total_frames}")
                
        except Exception as e:
            print(f"💥 Frame Loading Thread 오류: {e}")
        finally:
            # 종료 신호를 Queue에 반드시 전달 (Queue가 빌 때까지 대기)
            print("📤 종료 신호 전송 준비 중...")
            queue_full_warning_shown = False
            while True:
                try:
                    self.batch_queue.put(None, timeout=1.0)  # None은 종료 신호
                    print("🏁 Frame Loading Thread 종료 신호 전달 완료!")
                    break
                except queue.Full:
                    if not queue_full_warning_shown:
                        print("⏳ Queue 가득참, 종료 신호 전송을 위해 대기 중...")
                        queue_full_warning_shown = True
                    time.sleep(0.1)
            
            # 내부 스레드 정리
            if self.reader_started and self.reader_thread.is_alive():
                print("🛑 내부 프레임 읽기 스레드 종료 대기...")
                self.reader_thread.join(timeout=3.0)
                if self.reader_thread.is_alive():
                    print("⚠️ 내부 프레임 읽기 스레드 강제 종료 타임아웃")
                else:
                    print("✅ 내부 프레임 읽기 스레드 정상 종료")
            
            # VideoCapture 정리
            if self.cap.isOpened():
                self.cap.release()
                print("🎬 VideoCapture 해제 완료")
            
            # 최종 통계 출력
            print(f"📈 Frame Loading Thread 최종 통계:")
            print(f"   - 로드된 배치 수: {self.stats['total_batches_loaded']}")
            print(f"   - 로드된 프레임 수: {self.stats['total_frames_loaded']}")
            print(f"   - 실패한 프레임 수: {self.stats['failed_frames']}")
            print(f"   - 버퍼 대기 횟수: {self.stats['buffer_waits']}")
            print(f"   - 최대 버퍼 사용량: {self.stats['max_buffer_used']}/{self.max_buffer_size}")
            
            # 버퍼 효율성 계산
            if self.stats['total_frames_loaded'] > 0:
                success_rate = (self.stats['total_frames_loaded'] / 
                              (self.stats['total_frames_loaded'] + self.stats['failed_frames'])) * 100
                buffer_efficiency = (self.stats['max_buffer_used'] / self.max_buffer_size) * 100
                print(f"   - 프레임 로딩 성공률: {success_rate:.1f}%")
                print(f"   - 버퍼 사용 효율성: {buffer_efficiency:.1f}%")
    



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

# ANCHOR Detection Result Processing
def process_batch_results(batch_results, batch_indices, batch_original_frames, 
                         tracker, args, fps, palette, person_class_id,
                         # 상태 변수들
                         track_history, track_side, track_color, 
                         forward_cnt, backward_cnt, current_occupancy, current_congestion,
                         last_update_time, heatmap, last_save_frame, log_buffer,
                         # I/O 관련
                         output_dir, out, log_file,
                         # 라인 크로싱 관련
                         p1, p2, p1_input, p2_input, seg_dx, seg_dy, seg_len2,
                         width, height, output_width, output_height, scale_x, scale_y,
                         # Progress bar & Queue monitoring
                         pbar=None, frame_queue=None, result_queue=None, pipeline_stats=None,
                         # ROI Access Detection
                         roi_manager=None):
    """Batch 결과를 개별 프레임으로 처리 (CPU 데이터 기반)"""
    
    updated_state = {
        'forward_cnt': forward_cnt,
        'backward_cnt': backward_cnt,
        'current_occupancy': current_occupancy,
        'current_congestion': current_congestion,
        'last_update_time': last_update_time,
        'last_save_frame': last_save_frame
    }
    
    for cpu_result, frame_idx, original_frame in zip(batch_results, batch_indices, batch_original_frames):
        # 시간 계산
        current_time = frame_idx / fps
        hours = int(current_time // 3600)
        minutes = int((current_time % 3600) // 60)
        seconds = int(current_time % 60)
        milliseconds = int((current_time % 1) * 1000)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        
        # ObjectMeta 변환 (CPU 데이터 기반)
        detected_objects = convert_results_to_objects(cpu_result, args.names)
        
        # Tracker 업데이트 (기존 로직 유지)
        tracked_objects = tracker.update(detected_objects, cpu_result['orig_img'], "None")
        
        # ROI Access Detection 업데이트
        current_roi_tracks = {}
        if roi_manager is not None:
            roi_manager.update(tracked_objects, frame_idx, cpu_result['orig_img'].shape)
            current_roi_tracks = roi_manager.get_current_roi_tracks()
        
        # 기존 변수들 추출 (호환성 유지)
        if tracked_objects:
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

        # 로깅 처리
        if log_file is not None:
            class_counts = defaultdict(int)
            for cid in class_ids:
                class_counts[cid] += 1

            for cid, tid in zip(class_ids, track_ids):
                class_name = args.names[cid]
                class_count = class_counts[cid]
                log_entry = f"{time_str},{class_name},{tid},{class_count}\n"
                log_buffer.append(log_entry)
                if len(log_buffer) >= 100:
                    log_file.writelines(log_buffer)
                    log_file.flush()
                    log_buffer.clear()

        # 시각화 - 출력 해상도로 리사이즈
        frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        
        # 출력 해상도와 입력 해상도가 다르면 리사이즈
        if output_width != width or output_height != height:
            frame_rgb = cv2.resize(frame_rgb, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        
        annotated = frame_rgb.copy()
        
        # 객체 오버레이 (스케일링된 좌표로)
        scaled_objects = []
        if tracked_objects:
            # 객체 좌표를 출력 해상도에 맞게 스케일링
            for obj in tracked_objects:
                scaled_obj = ObjectMeta(
                    box=[obj.box[0] * scale_x, obj.box[1] * scale_y, 
                         obj.box[2] * scale_x, obj.box[3] * scale_y],
                    mask=None,  # 마스크는 별도 처리
                    confidence=obj.confidence,
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    track_id=obj.track_id
                )
                
                # 마스크가 있으면 출력 해상도로 리사이즈
                if obj.mask is not None:
                    scaled_obj.mask = cv2.resize(obj.mask.astype(np.float32), 
                                               (output_width, output_height), 
                                               interpolation=cv2.INTER_LINEAR)
                
                scaled_objects.append(scaled_obj)
            
            annotated = draw_objects_overlay(annotated, scaled_objects, palette)
        
        # ROI 시각화 (출력 해상도에 맞게 스케일링)
        if roi_manager is not None:
            # ROI polygon을 출력 해상도에 맞게 스케일링
            scaled_roi_polygons = []
            for polygon in roi_manager.roi_polygons:
                coords = np.array(polygon.exterior.coords)
                scaled_coords = coords * [scale_x, scale_y]
                from shapely.geometry import Polygon
                scaled_polygon = Polygon(scaled_coords)
                scaled_roi_polygons.append(scaled_polygon)
            
            # ROI polygon 그리기 (스케일링된 좌표로)
            annotated = draw_roi_polygons(annotated, scaled_roi_polygons, 
                                        roi_manager.roi_names, roi_manager.roi_stats)
            
            # ROI에 있는 객체들 하이라이트 (스케일링된 객체로)
            if tracked_objects:
                annotated = highlight_roi_objects(annotated, scaled_objects, 
                                                current_roi_tracks, palette)

        # Occupancy 업데이트
        raw_occupancy = int(np.sum(class_ids == person_class_id))
        if current_time - updated_state['last_update_time'] >= 1.0:
            updated_state['last_update_time'] = current_time
            updated_state['current_occupancy'] = raw_occupancy
            updated_state['current_congestion'] = int(min(raw_occupancy / max(args.max_people, 1), 1.0) * 100)

        line_crossed_this_frame = False

        # 히트맵 및 라인 크로싱 처리
        for obj in tracked_objects:
            if obj.track_id == -1 or obj.class_id != person_class_id:
                continue

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

            # 라인 크로싱 체크 (입력 해상도 기준 좌표 사용)
            if p1_input is not None and p2_input is not None:
                side_now = point_side((cx, cy), p1_input, p2_input)
                t = ((cx - p1_input[0]) * seg_dx + (cy - p1_input[1]) * seg_dy) / seg_len2
                inside = 0.0 <= t <= 1.0

                side_prev = track_side.get(obj.track_id, side_now)
                if inside and (side_prev * side_now < 0):
                    line_crossed_this_frame = True
                    if side_prev < 0 < side_now:
                        updated_state['forward_cnt'] += 1
                    elif side_prev > 0 > side_now:
                        updated_state['backward_cnt'] += 1
                if inside and side_now != 0:
                    track_side[obj.track_id] = side_now

        # 라인 그리기 (출력 해상도에 맞게 스케일링)
        if p1 is not None and p2 is not None:
            line_thickness = 8
            
            if line_crossed_this_frame:
                cv2.line(annotated, p1, p2, (255, 0, 0), line_thickness)
            else:
                cv2.line(annotated, p1, p2, (255, 255, 255), line_thickness)

            mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            perp_x, perp_y = -dy, dx
            length = 50
            mag = (perp_x**2 + perp_y**2) ** 0.5 or 1
            perp_x, perp_y = int(perp_x / mag * length), int(perp_y / mag * length)
            arrow_start, arrow_end = (mid_x, mid_y), (mid_x + perp_x, mid_y + perp_y)
            cv2.arrowedLine(annotated, arrow_start, arrow_end, (0, 255, 0), line_thickness, tipLength=0.6)

        # Overlay 정보 그리기 (ROI 정보 추가)
        overlay = [
            f"[1] Congestion : {updated_state['current_congestion']:3d} %",
            f"[2] Crossing   : forward {updated_state['forward_cnt']} | backward {updated_state['backward_cnt']}",
            f"[3] Occupancy  : {updated_state['current_occupancy']}",
            f"[4] Time      : {time_str}"
        ]
        
        # ROI 접근 횟수 정보 추가
        if roi_manager is not None:
            for roi_name, roi_stat in roi_manager.roi_stats.items():
                overlay.append(f"[ROI] {roi_name}: {roi_stat['total_access']} accesses")
        
        # 오버레이 텍스트 크기
        x0, y0, dy = 30, 60, 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        padding = 10
        
        max_width = 0
        for txt in overlay:
            (tw, th), _ = cv2.getTextSize(txt, font, font_scale, font_thickness)
            max_width = max(max_width, tw)
        rect_x1, rect_y1 = x0 - padding, y0 - th - padding
        rect_x2, rect_y2 = x0 + max_width + padding, y0 + (len(overlay) - 1) * dy + padding
        overlay_rect = annotated.copy()
        cv2.rectangle(overlay_rect, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay_rect, 0.4, annotated, 0.6, 0)
        for i, txt in enumerate(overlay):
            cv2.putText(annotated, txt, (x0, y0 + i * dy),
                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # 히트맵 minimap (출력 해상도에 맞게 조정)
        blur = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_hm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        color_hm = cv2.cvtColor(color_hm, cv2.COLOR_BGR2RGB)

        # 미니맵 크기를 출력 해상도에 맞게 조정
        mini_w = 400
        mini_h = int(output_height * mini_w / output_width)
        mini_map = cv2.resize(color_hm, (mini_w, mini_h))

        margin = 20
        x_start = output_width - mini_w - margin
        y_start = margin
        
        # 경계 체크
        if x_start >= 0 and y_start >= 0 and x_start + mini_w <= output_width and y_start + mini_h <= output_height:
            roi = annotated[y_start:y_start + mini_h, x_start:x_start + mini_w]
            blended = cv2.addWeighted(mini_map, 0.6, roi, 0.4, 0)
            annotated[y_start:y_start + mini_h, x_start:x_start + mini_w] = blended

        # 중간 저장
        if frame_idx - updated_state['last_save_frame'] >= args.save_interval:
            snapshot_path = os.path.join(output_dir, f"intermediate_{frame_idx:06d}_{time_str.replace(':', '-')}.jpg")
            cv2.imwrite(snapshot_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            updated_state['last_save_frame'] = frame_idx
            print(f"\n중간 저장 완료: {snapshot_path}")

        # 비디오 출력
        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        
        # Progress bar 즉시 업데이트 (프레임 단위) + 실시간 메트릭 반영
        if pbar is not None:
            pbar.update(1)
            
            # 실시간 메트릭을 postfix로 표시
            if frame_queue is not None and result_queue is not None and pipeline_stats is not None:
                frame_queue_size = frame_queue.qsize()
                result_queue_size = result_queue.qsize()
                
                # FPS 계산
                elapsed_time = time.time() - pipeline_stats['start_time']
                avg_fps = pipeline_stats['processed_frames'] / elapsed_time if elapsed_time > 0 else 0
                
                # 실시간 메트릭을 postfix로 업데이트
                pbar.set_postfix({
                    'FPS': f'{avg_fps:.1f}',
                    'Frame_Q': frame_queue_size,
                    'Result_Q': result_queue_size
                    # 'Occupancy': updated_state['current_occupancy'],
                    # 'Congestion': f"{updated_state['current_congestion']}%",
                    # 'Forward': updated_state['forward_cnt'],
                    # 'Backward': updated_state['backward_cnt']
                })
    
    return updated_state

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

    # 입력 해상도
    input_width, input_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 출력 해상도 설정 (args에서 지정하지 않으면 입력 해상도 사용)
    output_width = args.output_width if args.output_width is not None else input_width
    output_height = args.output_height if args.output_height is not None else input_height
    
    # 스케일링 비율 계산
    scale_x = output_width / input_width
    scale_y = output_height / input_height
    
    print(f"\n📺 해상도 설정:")
    print(f"   - 입력 해상도: {input_width} x {input_height}")
    print(f"   - 출력 해상도: {output_width} x {output_height}")
    print(f"   - 스케일링 비율: x={scale_x:.3f}, y={scale_y:.3f}")
    
    # 기존 변수명 호환성을 위해 width, height는 입력 해상도로 유지 (내부 로직용)
    width, height = input_width, input_height

    # Select frames to process
    frames_to_process = range(total_frames)
    
    # 🚀 Pipeline 처리 설정
    batch_size = args.batch_size
    total_batches = (total_frames + batch_size - 1) // batch_size
    max_queue_size = 3  # Queue 최대 크기 (메모리 제어)
    
    print(f"\n🚀 Pipeline 처리 시작: {total_frames} 프레임을 {batch_size} 단위로 {total_batches} batch 처리")
    # print(f"📦 Queue 기반 병렬 처리: Frame Loading Thread + Inference Pipeline")

    # ─────────────── 🚀 3단계 Pipeline Queue 및 Thread 설정 ─────────── #
    frame_queue = queue.Queue(maxsize=max_queue_size)    # Frame Loading → Inference
    result_queue = queue.Queue(maxsize=max_queue_size)   # Inference → Main
    
    print(f"\n🔗 3단계 Pipeline 구성:")
    print(f"   Stage 1: Frame Loading Thread → Frame Queue")
    print(f"   Stage 2: Inference Thread → Result Queue") 
    print(f"   Stage 3: Main Thread (Process Results)")
    
    # Frame Loading Thread 시작
    frame_loader = FrameLoadingThread(
        video_source=args.source,
        total_frames=total_frames,
        batch_size=batch_size,
        args=args,
        batch_queue=frame_queue,  # frame_queue로 변경
        max_queue_size=max_queue_size
    )
    frame_loader.start()
    # print(f"🎬 Stage 1 시작: Frame Loading Thread (PID: {frame_loader.ident})")

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

    # Inference Thread 시작
    inference_thread = InferenceThread(
        frame_queue=frame_queue,
        result_queue=result_queue,
        model=model,
        model_vp=model_vp,
        args=args
    )
    inference_thread.start()
    # print(f"🧠 Stage 2 시작: Inference Thread (PID: {inference_thread.ident})")

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
        # 입력 해상도 기준으로 라인 좌표 계산 (내부 로직용)
        p1_input = (int(args.line_end[0]   * width),  int(args.line_end[1]   * height))
        p2_input = (int(args.line_start[0] * width),  int(args.line_start[1] * height))
        
        # 출력 해상도 기준으로 라인 좌표 계산 (시각화용)
        p1 = (int(args.line_end[0]   * output_width),  int(args.line_end[1]   * output_height))
        p2 = (int(args.line_start[0] * output_width),  int(args.line_start[1] * output_height))
        
        # 선분 벡터 및 길이² 계산 (입력 해상도 기준, 내부 로직용)
        seg_dx, seg_dy = p2_input[0] - p1_input[0], p2_input[1] - p1_input[1]
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy or 1
    else:
        p1 = p2 = None
        p1_input = p2_input = None
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

    # ─────────────── ROI Access Detection 설정 ──────────────── #
    roi_manager = None
    if args.roi_zones:
        # ROI zones 파싱
        roi_polygons = parse_roi_zones(args.roi_zones)
        if roi_polygons:
            # ROI names 파싱
            roi_names = parse_roi_names(args.roi_names, len(roi_polygons))
            
            # ROI Access Manager 초기화
            roi_manager = ROIAccessManager(
                roi_polygons=roi_polygons,
                roi_names=roi_names,
                detection_method=args.roi_detection_method,
                dwell_time=args.roi_dwell_time,
                bbox_threshold=args.roi_bbox_threshold,
                mask_threshold=args.roi_mask_threshold,
                fps=fps,
                exit_grace_time=args.roi_exit_grace_time
            )
        else:
            print("⚠️ ROI zones 파싱 실패, ROI 기능 비활성화")
    else:
        print("📍 ROI zones 미설정, ROI 기능 비활성화")

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

    # 비디오 출력 설정 (출력 해상도 사용)
    out = cv2.VideoWriter(output_video,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (output_width, output_height))

    # Pipeline 통계
    pipeline_stats = {
        'processed_batches': 0,
        'processed_frames': 0,
        'start_time': time.time(),
        'empty_queue_waits': 0
    }

    print(f"\n🎬 Process Results 시작\n")
    
    pbar = tqdm(total=total_frames, desc="🔥 Pipeline Processing")

    # ────────────────── 🚀 MAIN THREAD RESULT PROCESSING LOOP 🚀 ───────────────── #
    try:
        batch_idx = 0
        while True:
            try:
                # Result Queue에서 처리 결과 가져오기 (타임아웃 3초)
                result_data = result_queue.get(timeout=3.0)
                
                # 종료 신호 확인
                if result_data is None:
                    print("🏁 Main Thread: Inference Thread 종료 신호 수신")
                    break
                
                # 결과 데이터 언패킹
                batch_results = result_data['batch_results']
                batch_indices = result_data['batch_indices']
                batch_original_frames = result_data['batch_original_frames']
                loaded_count = result_data['loaded_count']
                inference_batch_idx = result_data['batch_idx']
                vpe_updated = result_data['vpe_updated']
                prev_vpe_status = result_data['prev_vpe_status']
                
                # print(f"📊 Main Thread: Batch {batch_idx + 1} 처리 시작 ({loaded_count} 프레임)")
                
                if not batch_results:
                    print("⚠️ 빈 결과 수신, 건너뛰기")
                    batch_idx += 1
                    continue
                
                # ──────── 3. Batch 결과 처리 (CPU 작업) ──────── #
                updated_state = process_batch_results(
                    batch_results, batch_indices, batch_original_frames,
                    tracker, args, fps, palette, person_class_id,
                    # 상태 변수들
                    track_history, track_side, track_color, 
                    forward_cnt, backward_cnt, current_occupancy, current_congestion,
                    last_update_time, heatmap, last_save_frame, log_buffer,
                    # I/O 관련
                    output_dir, out, log_file,
                    # 라인 크로싱 관련
                    p1, p2, p1_input, p2_input, seg_dx, seg_dy, seg_len2,
                    width, height, output_width, output_height, scale_x, scale_y,
                    # Progress bar & Queue monitoring
                    pbar, frame_queue, result_queue, pipeline_stats,
                    # ROI Access Detection
                    roi_manager
                )
                
                # 상태 변수 업데이트
                forward_cnt = updated_state['forward_cnt']
                backward_cnt = updated_state['backward_cnt']
                current_occupancy = updated_state['current_occupancy'] 
                current_congestion = updated_state['current_congestion']
                last_update_time = updated_state['last_update_time']
                last_save_frame = updated_state['last_save_frame']
                
                # 통계 업데이트
                pipeline_stats['processed_batches'] += 1
                pipeline_stats['processed_frames'] += loaded_count
                
                # Progress bar postfix 업데이트 (실제 update는 process_batch_results 내부에서 프레임 단위로)
                elapsed_time = time.time() - pipeline_stats['start_time']
                avg_fps = pipeline_stats['processed_frames'] / elapsed_time if elapsed_time > 0 else 0
                
                # 배치 완료 후에는 별도 업데이트 불필요 (process_batch_results에서 프레임별로 실시간 업데이트됨)
                
                # print(f"✅ Main Thread: Batch {batch_idx + 1} 완료 ({loaded_count} 프레임, VPE: {'Updated' if vpe_updated else 'Kept'})")
                batch_idx += 1
                
            except queue.Empty:
                pipeline_stats['empty_queue_waits'] += 1
                # print(f"⏳ Result Queue 비어있음, 대기 중... ({pipeline_stats['empty_queue_waits']}회)")
                
                # Inference Thread가 아직 살아있는지 확인
                if not inference_thread.is_alive():
                    print("💀 Inference Thread 종료됨, Main Thread 종료")
                    break
                    
                continue
            
            except Exception as e:
                print(f"💥 Main Thread 처리 중 오류: {e}")
                import traceback
                traceback.print_exc()
                break
    
    finally:
        # ──────── 🧹 3-Stage Pipeline 정리 ──────── #
        print(f"\n🧹 3-Stage Pipeline 정리 중...")
        
        # Inference Thread 종료
        if inference_thread.is_alive():
            print("🛑 Inference Thread 종료 요청...")
            inference_thread.stop()
            inference_thread.join(timeout=5.0)
            if inference_thread.is_alive():
                print("⚠️ Inference Thread 강제 종료 타임아웃")
            else:
                print("✅ Inference Thread 정상 종료")
        
        # Frame Loading Thread 종료
        if frame_loader.is_alive():
            print("🛑 Frame Loading Thread 종료 요청...")
            frame_loader.stop()
            frame_loader.join(timeout=5.0)
            if frame_loader.is_alive():
                print("⚠️ Frame Loading Thread 강제 종료 타임아웃")
            else:
                print("✅ Frame Loading Thread 정상 종료")
        
        # Queue 정리
        print("🗑️ Queue 정리 중...")
        queue_cleanup_count = 0
        
        # Frame Queue 정리
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
                queue_cleanup_count += 1
            except queue.Empty:
                break
        
        # Result Queue 정리
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
                queue_cleanup_count += 1
            except queue.Empty:
                break
        
        if queue_cleanup_count > 0:
            print(f"🗑️ Queue에서 {queue_cleanup_count}개 미처리 데이터 정리됨")

    # ────────────────── 🎉 Pipeline 처리 완료 & 최종 통계 🎉 ───────────────── #
    total_elapsed_time = time.time() - pipeline_stats['start_time']
    final_avg_fps = pipeline_stats['processed_frames'] / total_elapsed_time if total_elapsed_time > 0 else 0
    
    print(f"\n🎉 Pipeline 처리 완료!")
    print(f"📊 최종 통계:")
    print(f"   - 총 처리된 batch: {pipeline_stats['processed_batches']}/{total_batches}")
    print(f"   - 총 처리된 프레임: {pipeline_stats['processed_frames']}/{total_frames}")
    print(f"   - 총 처리 시간: {total_elapsed_time:.1f}초")
    print(f"   - 평균 FPS: {final_avg_fps:.1f}")
    print(f"   - Queue 빈 대기 횟수: {pipeline_stats['empty_queue_waits']}")
    # print(f"   - VPE 상태: {'활성화' if prev_vpe is not None else '비활성화'}")
    print(f"   - Batch 크기: {batch_size}")
    print(f"   - Queue 크기: {max_queue_size}")
    print(f"   - Frame Loading 스레드 수: {args.frame_loading_threads}")
    
    print(f"📈 분석 결과:")
    print(f"   - 최종 occupancy: {current_occupancy}")
    print(f"   - 최종 congestion: {current_congestion}%")
    print(f"   - 라인 크로싱: forward {forward_cnt}, backward {backward_cnt}")
    
    # ROI 최종 통계 출력 및 저장
    if roi_manager is not None:
        roi_manager.print_final_statistics()
        
        # ROI 통계 JSON 파일로 저장
        roi_stats_path = os.path.join(output_dir, "roi_statistics.json")
        roi_manager.save_statistics(roi_stats_path)
    
    # if args.cross_vp and prev_vpe is not None:
    #     print(f"🧠 VPE 정보:")
    #     print(f"   - VPE shape: {prev_vpe.shape}")
    #     print(f"   - VPE dtype: {prev_vpe.dtype}")
    #     print(f"   - Cross-VP 모드: 활성화")
    
# ──────── 스레드별 상세 통계 ──────── #
    
    # Frame Loading Thread 통계 출력
    if hasattr(frame_loader, 'stats'):
        loader_stats = frame_loader.stats
        print(f"📦 Stage 1 - Frame Loading Thread 통계:")
        print(f"   - 로드된 배치 수: {loader_stats['total_batches_loaded']}")
        print(f"   - 로드된 프레임 수: {loader_stats['total_frames_loaded']}")
        print(f"   - 실패한 프레임 수: {loader_stats['failed_frames']}")
        
        # 효율성 계산
        if loader_stats['total_frames_loaded'] > 0:
            success_rate = (loader_stats['total_frames_loaded'] / 
                          (loader_stats['total_frames_loaded'] + loader_stats['failed_frames'])) * 100
            print(f"   - 프레임 로딩 성공률: {success_rate:.1f}%")
    
    # Inference Thread 통계 출력
    if hasattr(inference_thread, 'stats'):
        inference_stats = inference_thread.stats
        print(f"🧠 Stage 2 - Inference Thread 통계:")
        print(f"   - 처리된 배치 수: {inference_stats['total_batches_processed']}")
        print(f"   - 처리된 프레임 수: {inference_stats['total_frames_processed']}")
        print(f"   - VPE 업데이트 횟수: {inference_stats['vpe_updates']}")
        print(f"   - 실패한 배치 수: {inference_stats['failed_batches']}")
        
        # GPU 효율성 계산
        if inference_stats['total_batches_processed'] > 0:
            gpu_success_rate = ((inference_stats['total_batches_processed'] - inference_stats['failed_batches']) / 
                              inference_stats['total_batches_processed']) * 100
            vpe_update_rate = (inference_stats['vpe_updates'] / inference_stats['total_batches_processed']) * 100
            print(f"   - GPU 추론 성공률: {gpu_success_rate:.1f}%")
            print(f"   - VPE 업데이트 비율: {vpe_update_rate:.1f}%")
    
    print(f"📊 Stage 3 - Main Thread 통계:")
    print(f"   - 처리 완료된 배치 수: {pipeline_stats['processed_batches']}")
    print(f"   - 처리 완료된 프레임 수: {pipeline_stats['processed_frames']}")
    print(f"   - Queue 대기 횟수: {pipeline_stats['empty_queue_waits']}")
    
    # 전체 Pipeline 효율성
    if pipeline_stats['processed_frames'] > 0 and hasattr(loader_stats, 'total_frames_loaded'):
        pipeline_efficiency = (pipeline_stats['processed_frames'] / loader_stats['total_frames_loaded']) * 100
        print(f"   - Pipeline 전체 효율성: {pipeline_efficiency:.1f}%")

    pbar.close()
    cap.release()
    if log_file is not None:
        if log_buffer:
            log_file.writelines(log_buffer)
        log_file.close()
    out.release()
    
    print(f"\n✔ Pipeline 처리 완료! 저장 위치: {output_dir}")


if __name__ == "__main__":
    main() 
