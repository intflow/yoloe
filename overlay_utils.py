#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay Utilities for Video Annotation
────────────────────────────────────────────────────────────────────────────
Contains all visualization and overlay functions for video annotation:
- Object detection overlays (boxes, masks, labels)
- ROI visualization 
- Detection area visualization
- Low confidence object visualization
"""

import cv2
import numpy as np
from shapely.geometry import Polygon


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
        self.class_id = class_id    # class index (confidence에 따라 조정된 클래스)
        self.class_name = class_name # class name string
        self.track_id = track_id    # tracking ID (None initially)
        self.original_class_id = class_id  # 원본 클래스 ID (VPE 업데이트용)
    
    def __repr__(self):
        return f"ObjectMeta(class={self.class_name}, track_id={self.track_id}, conf={self.confidence:.2f})"


# ─────────────────────── Direct Overlay Functions ────────────────────────────── #
def draw_box(image, box, track_id, class_id, color_palette):
    """박스 그리기"""
    x1, y1, x2, y2 = map(int, box)
    # 음수 track_id를 양수로 변환
    safe_track_id = abs(track_id) if track_id < 0 else track_id
    color = color_palette.by_idx(safe_track_id).as_bgr()
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


def draw_mask(image, mask, track_id, class_id, color_palette, opacity=0.4):
    """Supervision 방식의 마스크 그리기 (정교한 크기 변환 적용)"""
    if mask is None:
        return image
    
    # 음수 track_id를 양수로 변환
    safe_track_id = abs(track_id) if track_id < 0 else track_id
    color = color_palette.by_idx(safe_track_id).as_bgr()
    
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
    """🏷️ 라벨 그리기 (이미지 경계 고려)"""
    x1, y1, x2, y2 = map(int, box)
    # 음수 track_id를 양수로 변환
    safe_track_id = abs(track_id) if track_id < 0 else track_id
    color = color_palette.by_idx(safe_track_id).as_bgr()
    
    # 라벨 텍스트
    label = f"{class_name} {track_id} {confidence:.2f}"
    
    # 텍스트 크기 계산
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # 이미지 크기 가져오기
    img_height, img_width = image.shape[:2]
    
    # 라벨 높이 계산 (패딩 포함)
    label_height = text_h + baseline + 10
    
    # 🎯 라벨 위치 결정: bbox 위쪽에 공간이 있는지 확인
    if y1 - label_height >= 0:
        # bbox 위쪽에 충분한 공간이 있음 → 위쪽에 그리기 (기존 방식)
        label_y_top = y1 - text_h - baseline - 10
        label_y_bottom = y1
        text_y = y1 - baseline - 5
    else:
        label_y_top = y1
        label_y_bottom = y1 + text_h + baseline + 10
        text_y = y1 + text_h + 5
    
    # 라벨이 이미지 우측을 벗어나는 경우 x 좌표 조정
    if x1 + text_w >= img_width:
        x1 = max(0, img_width - text_w)
    
    # 라벨 배경 그리기
    cv2.rectangle(image, (x1, label_y_top), (x1 + text_w, label_y_bottom), color, -1)
    
    # 텍스트 그리기
    cv2.putText(image, label, (x1, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return image


def draw_low_conf_box(image, box, track_id, class_id, color_palette):
    """저신뢰도 박스 그리기 (점선 스타일)"""
    x1, y1, x2, y2 = map(int, box)
    # 음수 track_id를 양수로 변환
    safe_track_id = abs(track_id) if track_id < 0 else track_id
    color = color_palette.by_idx(safe_track_id).as_bgr()
    
    # 점선 효과를 위한 파라미터
    dash_length = 10
    gap_length = 5
    thickness = 2
    
    # 상단 선 (점선)
    x = x1
    while x < x2:
        end_x = min(x + dash_length, x2)
        cv2.line(image, (x, y1), (end_x, y1), color, thickness)
        x += dash_length + gap_length
    
    # 하단 선 (점선)
    x = x1
    while x < x2:
        end_x = min(x + dash_length, x2)
        cv2.line(image, (x, y2), (end_x, y2), color, thickness)
        x += dash_length + gap_length
    
    # 좌측 선 (점선)
    y = y1
    while y < y2:
        end_y = min(y + dash_length, y2)
        cv2.line(image, (x1, y), (x1, end_y), color, thickness)
        y += dash_length + gap_length
    
    # 우측 선 (점선)
    y = y1
    while y < y2:
        end_y = min(y + dash_length, y2)
        cv2.line(image, (x2, y), (x2, end_y), color, thickness)
        y += dash_length + gap_length
    
    return image


def draw_low_conf_label(image, box, track_id, class_name, confidence, color_palette):
    """저신뢰도 라벨 그리기 (반투명 배경과 특별한 형식)"""
    x1, y1, x2, y2 = map(int, box)
    # 음수 track_id를 양수로 변환
    safe_track_id = abs(track_id) if track_id < 0 else track_id
    color = color_palette.by_idx(safe_track_id).as_bgr()
    
    # 🎯 라벨 텍스트 (저신뢰도 표시 + track ID 포함)
    if track_id is not None and track_id >= 0:
        # track ID가 부여된 경우: [LOW] 클래스명 ID confidence
        label = f"[LOW] {class_name} {track_id} {confidence:.3f}"
    else:
        # track ID가 없는 경우: [LOW] 클래스명 confidence
        label = f"[LOW] {class_name} {confidence:.3f}"
    
    
    # 텍스트 크기 계산
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # 반투명 배경을 위한 오버레이
    overlay = image.copy()
    
    # 라벨 배경 그리기 (더 어두운 색상으로)
    bg_color = tuple(int(c * 0.6) for c in color)  # 원래 색상의 60%로 어둡게
    cv2.rectangle(overlay, (x1, y1 - text_h - baseline - 10), 
                  (x1 + text_w + 10, y1), bg_color, -1)
    
    # 반투명 효과 적용
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 텍스트 그리기 (밝은 색으로)
    cv2.putText(image, label, (x1 + 5, y1 - baseline - 5), 
                font, font_scale, (255, 255, 255), thickness)
    
    return image


def draw_objects_overlay(image, objects, color_palette, args=None, debug_enabled=False):
    """모든 개체의 overlay 그리기 (Supervision 방식: 큰 객체부터 작은 객체 순)
    추적된 모든 객체 표시 (정상 신뢰도 + 저신뢰도 객체 모두)
    """
    if not objects:
        return image
    
    # 객체 타입별 분리 (confidence 기준으로 저신뢰도 구분)
    normal_objects = []
    low_conf_objects = []
    
    # Default thresholds if args is not provided
    very_low_thresh = args.very_low_conf_thresh if args else 0.01
    medium_thresh = args.medium_conf_thresh if args else 0.1
    
    for obj in objects:
        # 🎯 confidence 기준으로만 저신뢰도 구분 (클래스는 원본 유지)
        if very_low_thresh <= obj.confidence < medium_thresh:
            low_conf_objects.append(obj)
        else:
            normal_objects.append(obj)
    
    # 🔍 실시간 디버깅 (옵션으로 활성화)
    if debug_enabled and objects:
        print(f"🎨 실제 오버레이 표시:")
        print(f"   - 정상 객체: {len(normal_objects)}개")
        print(f"   - 저신뢰도 object: {len(low_conf_objects)}개")
        
        if low_conf_objects:
            conf_values = [f"{obj.confidence:.3f}" for obj in low_conf_objects]
            track_ids = [f"{obj.track_id}" for obj in low_conf_objects]
            class_names = [obj.class_name for obj in low_conf_objects]
            print(f"   - 저신뢰도 객체 confidence: {', '.join(conf_values)}")
            print(f"   - 저신뢰도 객체 클래스: {', '.join(class_names)}")
            print(f"   - 저신뢰도 객체 track_ids: {', '.join(track_ids)}")
    
    all_display_objects = normal_objects + low_conf_objects
    
    if not all_display_objects:
        return image
    
    # 박스 area 계산하여 큰 것부터 정렬 (Supervision과 동일)
    areas = []
    for obj in all_display_objects:
        x1, y1, x2, y2 = obj.box
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
    
    # area 기준으로 큰 것부터 정렬 (flip으로 내림차순)
    sorted_indices = np.flip(np.argsort(areas))
    
    # 마스크부터 먼저 그리기 (supervision 방식)
    for idx in sorted_indices:
        obj = all_display_objects[idx]
        display_track_id = obj.track_id if obj.track_id is not None else -999 - idx
        image = draw_mask(image, obj.mask, display_track_id, obj.class_id, color_palette)
    
    # 그 다음 박스와 라벨 그리기 (원래 순서대로)
    for obj in all_display_objects:
        display_track_id = obj.track_id if obj.track_id is not None else -999
        
        # 🎯 confidence 기준으로 저신뢰도 객체 특별 스타일 적용
        if very_low_thresh <= obj.confidence < medium_thresh:
            # 저신뢰도 객체: 점선 박스와 특별 라벨 (원본 클래스 유지)
            image = draw_low_conf_box(image, obj.box, display_track_id, obj.class_id, color_palette)
            image = draw_low_conf_label(image, obj.box, display_track_id, obj.class_name, 
                                      obj.confidence, color_palette)
        else:
            # 정상 객체: 일반 스타일
            image = draw_box(image, obj.box, display_track_id, obj.class_id, color_palette)
            image = draw_label(image, obj.box, display_track_id, obj.class_name, 
                              obj.confidence, color_palette)
    
    return image


# ─────────────────────── ROI Visualization Functions ────────────────────────────── #
def draw_roi_polygons(image, roi_polygons, roi_names, roi_stats):
    """ROI polygon들을 이미지에 그리기 (반투명 색칠 포함)"""
    
    # 반투명 내부 채우기를 위한 복사본 생성
    overlay = image.copy()
    color = (255, 228, 0)
    
    for i, (polygon, roi_name) in enumerate(zip(roi_polygons, roi_names)):
        try:
            # polygon 좌표 추출
            coords = np.array(polygon.exterior.coords, dtype=np.int32)
            
            # 1. polygon 내부를 색으로 채우기 (overlay에만 - 나중에 투명도 적용)
            cv2.fillPoly(overlay, [coords], color)
        
        except Exception as e:
            print(f"⚠️ ROI polygon 그리기 오류: {e}")
    
    # 2. 반투명 내부 채우기 블렌딩 (투명도 30%)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 3. 테두리 그리기 (투명도 없이 원본 image에 직접)
    for i, (polygon, roi_name) in enumerate(zip(roi_polygons, roi_names)):
        try:
            coords = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.polylines(image, [coords], isClosed=True, color=color, thickness=8)
            
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
                
                # 배경 사각형 (원본 image에 직접)
                cv2.rectangle(image, (text_x - 5, text_y - text_h - 10), 
                             (text_x + text_w + 5, text_y + 5), color, -1)
                
                # 텍스트 (원본 image에 직접)
                cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)            
        except Exception as e:
            print(f"⚠️ ROI polygon 테두리 그리기 오류: {e}")
    
    return image


def highlight_roi_objects(image, tracked_objects, roi_tracks_status, color_palette):
    """🎯 ROI 상태별로 객체들을 하이라이트 (3단계 구분, 복합 키 기반)"""
    
    # 모든 ROI의 상태별 복합 키들 수집
    confirmed_composite_keys = set()  # 접근 확인 (빨간색 실선)
    pending_composite_keys = set()    # 접근 확인 대기 + 퇴장 대기 (빨간색 점선)
    
    for roi_name, status_dict in roi_tracks_status.items():
        confirmed_composite_keys.update(status_dict['confirmed_access'])
        pending_composite_keys.update(status_dict['pending_access'])
        pending_composite_keys.update(status_dict['exit_pending'])
    
    # 빨간색 정의
    red_color = (255, 0, 0)  # BGR 형식
    thickness = 4
    
    for obj in tracked_objects:
        # 객체의 복합 키 생성
        obj_composite_key = (obj.class_id, obj.track_id)
        
        if obj_composite_key in confirmed_composite_keys:
            # 접근 확인된 객체 → 빨간색 실선 테두리
            x1, y1, x2, y2 = map(int, obj.box)
            cv2.rectangle(image, (x1-2, y1-2), (x2+2, y2+2), red_color, thickness)
            
        elif obj_composite_key in pending_composite_keys:
            # 접근 확인 대기 또는 퇴장 대기 객체 → 빨간색 점선 테두리
            x1, y1, x2, y2 = map(int, obj.box)
            draw_dashed_rectangle(image, (x1-2, y1-2), (x2+2, y2+2), red_color, thickness)
    
    return image


def draw_dashed_rectangle(image, pt1, pt2, color, thickness, dash_length=10):
    """🎨 점선으로 사각형 그리기"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 4개 변을 각각 점선으로 그리기
    # 상단 변
    draw_dashed_line(image, (x1, y1), (x2, y1), color, thickness, dash_length)
    # 하단 변  
    draw_dashed_line(image, (x1, y2), (x2, y2), color, thickness, dash_length)
    # 좌측 변
    draw_dashed_line(image, (x1, y1), (x1, y2), color, thickness, dash_length)
    # 우측 변
    draw_dashed_line(image, (x2, y1), (x2, y2), color, thickness, dash_length)


def draw_dashed_line(image, pt1, pt2, color, thickness, dash_length=10):
    """🎨 점선 그리기"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 선분의 총 길이 계산
    total_length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    
    if total_length == 0:
        return
    
    # 단위 벡터 계산
    dx = (x2 - x1) / total_length
    dy = (y2 - y1) / total_length
    
    # 점선 그리기
    current_length = 0
    while current_length < total_length:
        # 현재 점 계산
        start_x = int(x1 + dx * current_length)
        start_y = int(y1 + dy * current_length)
        
        # 다음 점 계산 (dash_length만큼 이동)
        end_length = min(current_length + dash_length, total_length)
        end_x = int(x1 + dx * end_length)
        end_y = int(y1 + dy * end_length)
        
        # 실선 구간 그리기
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
        
        # 다음 구간으로 이동 (공백 포함)
        current_length += dash_length * 2


def draw_detection_area(image, detection_area_polygon, scale_x=1.0, scale_y=1.0):
    """기본 감지 영역을 연두색 테두리로 그리기"""
    if detection_area_polygon is None:
        return image
    
    try:
        # polygon 좌표 추출 및 스케일링
        coords = np.array(detection_area_polygon.exterior.coords, dtype=np.float32)
        scaled_coords = coords * [scale_x, scale_y]
        scaled_coords = scaled_coords.astype(np.int32)
        
        # 연두색 (BGR: 0, 255, 0) 테두리로 그리기
        lime_green = (83, 255, 76)
        thickness = 8
        
        cv2.polylines(image, [scaled_coords], isClosed=True, color=lime_green, thickness=thickness)
        
    except Exception as e:
        print(f"💥 기본 감지 영역 그리기 오류: {e}")
    
    return image


# ─────────────────────── Advanced Overlay Functions ────────────────────────────── #
def update_heatmap(heatmap, tracked_objects, person_class_id, width, height):
    """히트맵 업데이트"""
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
    
    return heatmap


def check_line_crossing(tracked_objects, person_class_id, track_side, 
                       p1_input, p2_input, seg_dx, seg_dy, seg_len2):
    """라인 크로싱 체크 및 상태 업데이트"""
    line_crossed_this_frame = False
    forward_cnt_delta = 0
    backward_cnt_delta = 0
    
    if p1_input is None or p2_input is None:
        return line_crossed_this_frame, forward_cnt_delta, backward_cnt_delta
    
    # point_side 함수 정의 (로컬 함수)
    def point_side(p, a, b):
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
    
    for obj in tracked_objects:
        if obj.track_id == -1 or obj.class_id != person_class_id:
            continue
            
        x1, y1, x2, y2 = obj.box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # 라인 크로싱 체크 (입력 해상도 기준 좌표 사용)
        side_now = point_side((cx, cy), p1_input, p2_input)
        t = ((cx - p1_input[0]) * seg_dx + (cy - p1_input[1]) * seg_dy) / seg_len2
        inside = 0.0 <= t <= 1.0

        side_prev = track_side.get(obj.track_id, side_now)
        if inside and (side_prev * side_now < 0):
            line_crossed_this_frame = True
            if side_prev < 0 < side_now:
                forward_cnt_delta += 1
            elif side_prev > 0 > side_now:
                backward_cnt_delta += 1
        if inside and side_now != 0:
            track_side[obj.track_id] = side_now
    
    return line_crossed_this_frame, forward_cnt_delta, backward_cnt_delta


def draw_counting_line(image, p1, p2, line_crossed_this_frame):
    """라인 그리기 (출력 해상도에 맞게 스케일링)"""
    if p1 is None or p2 is None:
        return image
        
    line_thickness = 8
    
    if line_crossed_this_frame:
        cv2.line(image, p1, p2, (255, 0, 0), line_thickness)
    else:
        cv2.line(image, p1, p2, (255, 255, 255), line_thickness)

    mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    perp_x, perp_y = -dy, dx
    length = 50
    mag = (perp_x**2 + perp_y**2) ** 0.5 or 1
    perp_x, perp_y = int(perp_x / mag * length), int(perp_y / mag * length)
    arrow_start, arrow_end = (mid_x, mid_y), (mid_x + perp_x, mid_y + perp_y)
    cv2.arrowedLine(image, arrow_start, arrow_end, (0, 255, 0), line_thickness, tipLength=0.6)
    
    return image


def draw_overlay_info(image, congestion, forward_cnt, backward_cnt, occupancy, 
                     time_str, roi_manager, output_width, output_height):
    """오버레이 정보 그리기"""
    # Overlay 정보 그리기 (ROI 정보 추가)
    overlay = [
        f"[1] Congestion : {congestion:3d} %",
        f"[2] Crossing   : forward {forward_cnt} | backward {backward_cnt}",
        f"[3] Occupancy  : {occupancy}",
        f"[4] Time      : {time_str}"
    ]
    
    # ROI 접근 횟수 정보 추가
    if roi_manager is not None:
        for roi_name, roi_stat in roi_manager.roi_stats.items():
            overlay.append(f"[ROI] {roi_name}: {roi_stat['total_access']} accesses")
    
    # 오버레이 텍스트 크기 (0~1 정규화 좌표를 출력 해상도에 맞게 스케일링)
    # 1920x1080 기준: x0=30, y0=60, dy=50, padding=10
    # 정규화: x0=30/1920=0.0156, y0=60/1080=0.0556, dy=50/1080=0.0463, padding=10/1920=0.0052
    
    # 정규화된 좌표 (0~1 범위)
    norm_x0, norm_y0, norm_dy, norm_padding = 0.0156, 0.0556, 0.0463, 0.0052
    
    # 출력 해상도에 맞게 스케일링
    x0 = int(norm_x0 * output_width)
    y0 = int(norm_y0 * output_height)
    dy = int(norm_dy * output_height)
    padding = int(norm_padding * output_width)
    
    # 폰트 크기도 출력 해상도에 맞게 스케일링
    # 1920x1080 기준: font_scale=1, font_thickness=2
    # 정규화: font_scale=1, font_thickness=2 (기본값 유지, 필요시 조정 가능)
    base_font_scale = 1.0
    base_font_thickness = 2
    
    # 출력 해상도에 따른 폰트 크기 조정 (1920x1080 기준으로 정규화)
    scale_factor = min(output_width / 1920, output_height / 1080)
    font_scale = base_font_scale * scale_factor
    font_thickness = max(1, int(base_font_thickness * scale_factor))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    max_width = 0
    for txt in overlay:
        (tw, th), _ = cv2.getTextSize(txt, font, font_scale, font_thickness)
        max_width = max(max_width, tw)
    rect_x1, rect_y1 = x0 - padding, y0 - th - padding
    rect_x2, rect_y2 = x0 + max_width + padding, y0 + (len(overlay) - 1) * dy + padding
    overlay_rect = image.copy()
    cv2.rectangle(overlay_rect, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay_rect, 0.4, image, 0.6, 0)
    for i, txt in enumerate(overlay):
        cv2.putText(image, txt, (x0, y0 + i * dy),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return image


def draw_heatmap_minimap(image, heatmap, output_width, output_height):
    """히트맵 미니맵 그리기 (출력 해상도에 맞게 조정)"""
    blur = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_hm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color_hm = cv2.cvtColor(color_hm, cv2.COLOR_BGR2RGB)

    # 미니맵 크기를 출력 해상도에 맞게 조정 (0~1 정규화 좌표 사용)
    # 1920x1080 기준: mini_w=400, margin=20
    # 정규화: mini_w=400/1920=0.2083, margin=20/1920=0.0104
    
    # 정규화된 크기 (0~1 범위)
    norm_mini_width = 0.2083  # 화면 너비의 20.83%
    norm_margin = 0.0104      # 화면 너비의 1.04%
    
    # 출력 해상도에 맞게 스케일링
    mini_w = int(norm_mini_width * output_width)
    mini_h = int(output_height * mini_w / output_width)
    margin = int(norm_margin * output_width)
    
    mini_map = cv2.resize(color_hm, (mini_w, mini_h))

    # 미니맵 위치 계산 (우상단)
    x_start = output_width - mini_w - margin
    y_start = margin
    
    # 경계 체크
    if x_start >= 0 and y_start >= 0 and x_start + mini_w <= output_width and y_start + mini_h <= output_height:
        roi = image[y_start:y_start + mini_h, x_start:x_start + mini_w]
        blended = cv2.addWeighted(mini_map, 0.6, roi, 0.4, 0)
        image[y_start:y_start + mini_h, x_start:x_start + mini_w] = blended
    
    return image 