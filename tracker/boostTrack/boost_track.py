"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import os
from copy import deepcopy
from typing import Optional, List

import cv2
import numpy as np

from .default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings
from .assoc import associate, iou_batch, MhDist_similarity, shape_similarity, soft_biou_batch
from .ecc import ECC
from .kalmanfilter import KalmanFilter


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,h,r] where x,y is the centre of the box and h is the height and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0

    r = w / float(h + 1e-6)

    return np.array([x, y, h, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,h,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """

    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h

    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    def __init__(self, bbox, emb: Optional[np.ndarray] = None, is_tentative: bool = False, class_id: int = 0, id_counter_callback=None):
        """
        Initialises a tracker using initial bounding box.
        """

        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        self.class_id = class_id
        
        # ID 할당 - tentative와 confirmed 완전히 분리
        if id_counter_callback:
            self.id = id_counter_callback(class_id, is_tentative)
            self.is_tentative_id = is_tentative
        else:
            # 기본 전역 카운터 (하위 호환성)
            if is_tentative:
                self.id = -1  # 기본 tentative ID
                self.is_tentative_id = True
            else:
                self.id = 1  # 기본 confirmed ID
                self.is_tentative_id = False

        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.emb = emb
        self.hit_streak = 0
        self.age = 0
        
        # Tentative 모드를 위한 상태 관리 필드들
        self.start_frame = 0  # 시작 프레임
        self.state = "Tentative"  # "Tentative", "Tracked", "Lost"
        self.is_activated = False  # 정식 활성화 여부

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7

        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update-1)

    def update(self, bbox: np.ndarray, score: float = 0):
        """
        Updates the state vector with observed bbox.
        """

        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)

    def camera_update(self, transform: np.ndarray):
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = transform @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = transform @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h,  w / h]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        if self.emb is None:
            self.emb = emb
        else:
            self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb
    
    @property
    def end_frame(self):
        """현재 프레임 ID 반환 (C++ end_frame()과 동일)"""
        return self.age
    
    def mark_removed(self):
        """tracker를 removed 상태로 변경"""
        self.state = "Removed"
    
    def activate(self, frame_count, id_counter_callback=None):
        """tracker를 정식 활성화"""
        # tentative tracker였다면 새로운 confirmed ID 할당
        if self.is_tentative_id:
            old_id = self.id
            if id_counter_callback:
                self.id = id_counter_callback(self.class_id, False)  # confirmed ID 요청
            else:
                self.id = 1  # 기본값
            self.is_tentative_id = False
            # print(f"[ID 승격] Tentative ID {old_id} → Confirmed ID {self.id}")
        
        self.state = "Tracked"
        self.is_activated = True
        
    def mark_lost(self):
        """tracker를 lost 상태로 변경"""
        self.state = "Lost"
        
    def re_activate(self, bbox, score, frame_count):
        """lost tracker를 재활성화"""
        self.update(bbox, score)
        self.state = "Tracked"
        self.is_activated = True
        
    def re_match(self, new_tracker, frame_count):
        """lost tracker와 tentative tracker를 병합 (C++ re_match와 동일)"""
        # C++과 동일: Kalman filter 업데이트 대신 새 tracker의 상태를 직접 복사
        self.kf.x = new_tracker.kf.x.copy()
        self.kf.covariance = new_tracker.kf.covariance.copy()
        
        # 상태 정보 업데이트
        self.state = "Tracked"
        self.is_activated = True
        
        # 새 tracker의 속성들 복사 (C++과 동일)
        # score, emb 등의 정보도 새 tracker에서 가져옴
        if hasattr(new_tracker, 'emb') and new_tracker.emb is not None:
            self.emb = new_tracker.emb.copy()
        
        self.age = frame_count  # frame_id 업데이트
        
    def tentative_init(self, frame_count):
        """tentative 모드로 초기화"""
        self.start_frame = frame_count
        self.state = "Tentative"
        self.is_activated = False


class BoostTrack(object):
    def __init__(self, video_name: Optional[str] = None, det_thresh: Optional[float] = None, iou_threshold: Optional[float] = None, max_num_tracks: Optional[int] = None, max_age_sec: Optional[float] = None, fps: Optional[float] = None):
        self.frame_count = 0
        self.trackers: List[KalmanBoxTracker] = []
        self.trackers_by_class = {}  # Dictionary to store trackers by class
        
        # 클래스별 ID 카운터
        self.count_by_class = {}  # 정식 tracker용 counter (양수 ID)
        self.tentative_count_by_class = {}  # tentative tracker용 counter (음수 ID)
        
        # Tentative 모드를 위한 컨테이너들
        self.tentative_trackers_by_class = {}  # 수습 중인 tracker들
        self.lost_trackers_by_class = {}  # lost된 tracker들
        
        # Tentative 모드 설정
        self.probation_age = 30  # 수습 기간 (프레임)
        self.early_termination_age = 2  # 조기 제거 기간
        self.max_num_tracks = max_num_tracks  # 최대 추적 수 (None이면 무제한)

        # max_age 설정: 초 단위를 프레임 단위로 변환
        if max_age_sec is not None and fps is not None:
            self.max_age = int(max_age_sec * fps)  # 초 * fps = 프레임
        else:
            self.max_age = GeneralSettings.max_age(video_name)  # 기본값 사용
        # CLI 파라미터가 있으면 사용, 없으면 기본값 사용
        self.iou_threshold = iou_threshold if iou_threshold is not None else GeneralSettings['iou_threshold']
        self.det_thresh = det_thresh if det_thresh is not None else GeneralSettings['det_thresh']
        self.min_hits = GeneralSettings['min_hits']

        self.lambda_iou = BoostTrackSettings['lambda_iou']
        self.lambda_mhd = BoostTrackSettings['lambda_mhd']
        self.lambda_shape = BoostTrackSettings['lambda_shape']
        self.use_dlo_boost = BoostTrackSettings['use_dlo_boost']
        self.use_duo_boost = BoostTrackSettings['use_duo_boost']
        self.dlo_boost_coef = BoostTrackSettings['dlo_boost_coef']

        self.use_rich_s = BoostTrackPlusPlusSettings['use_rich_s']
        self.use_sb = BoostTrackPlusPlusSettings['use_sb']
        self.use_vt = BoostTrackPlusPlusSettings['use_vt']

        if GeneralSettings['use_embedding']:
            from .embedding import EmbeddingComputer
            self.embedder = EmbeddingComputer(GeneralSettings['dataset'], GeneralSettings['test_dataset'], True)
        else:
            self.embedder = None

        if GeneralSettings['use_ecc']:
            self.ecc = ECC(scale=350, video_name=video_name, use_cache=True)
        else:
            self.ecc = None

    def get_next_id(self, class_id: int, is_tentative: bool) -> int:
        """클래스별로 독립적인 ID 생성"""
        if is_tentative:
            if class_id not in self.tentative_count_by_class:
                self.tentative_count_by_class[class_id] = 0
            self.tentative_count_by_class[class_id] += 1
            return -self.tentative_count_by_class[class_id]  # 음수 ID
        else:
            if class_id not in self.count_by_class:
                self.count_by_class[class_id] = 0
            self.count_by_class[class_id] += 1
            return self.count_by_class[class_id]  # 양수 ID

    def joint_stracks(self, tlista, tlistb):
        """두 tracker 리스트를 합치되 중복 제거 (C++ joint_stracks와 동일)"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    def sub_stracks(self, tlista, tlistb):
        """tlista에서 tlistb에 있는 tracker들 제거 (C++ sub_stracks와 동일)"""
        stracks = {}
        for t in tlista:
            stracks[t.id] = t
        for t in tlistb:
            tid = t.id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())

    def update(self, objects, img_numpy, tag):
        """
        ObjectMeta 리스트를 받아서 tracking하고 track_id를 부여하여 반환
        C++ tracker_ref.cpp의 정확한 순서를 따름 (임시 리스트 시스템 사용)
        """
        if not objects:
            return []

        self.frame_count += 1

        # ═══════════════════════════════════════════════════════════════════
        # 임시 리스트들 선언 (C++의 activated_stracks, lost_stracks 등과 동일)
        # ═══════════════════════════════════════════════════════════════════
        activated_trackers_by_class = {}  # activated_stracks
        lost_trackers_by_class = {}       # lost_stracks  
        refind_trackers_by_class = {}     # refind_stracks
        rematched_trackers_by_class = {}  # rematched_stracks
        tentative_trackers_by_class = {}  # tentative_stracks
        removed_trackers_by_class = {}    # removed_stracks

        # ═══════════════════════════════════════════════════════════════════
        # ANCHOR Step 1: Get detections - 객체들 분리 및 numpy 배열 생성
        # ═══════════════════════════════════════════════════════════════════
        high_conf_objects = []  # >= 0.1 (track_thresh)
        low_conf_objects = []   # 0.01 ~ 0.1
        high_conf_dets = []
        
        for obj in objects:
            if obj.confidence >= 0.1:  # track_thresh
                high_conf_objects.append(obj)
                x1, y1, x2, y2 = obj.box
                det = [x1, y1, x2, y2, obj.confidence, obj.class_id]
                high_conf_dets.append(det)
            elif 0.01 <= obj.confidence < 0.1:
                low_conf_objects.append(obj)
        
        high_conf_dets = np.array(high_conf_dets) if high_conf_dets else np.empty((0, 6))

        if self.ecc is not None:
            transform = self.ecc(img_numpy, self.frame_count, tag)
            for cls_trackers in self.trackers_by_class.values():
                for trk in cls_trackers:
                    trk.camera_update(transform)

        # ═══════════════════════════════════════════════════════════════════
        # ANCHOR Step 2: First association - confirmed + lost vs high-confidence
        # ═══════════════════════════════════════════════════════════════════
        unmatched_high_conf_by_class = {}  # 매칭안된 high confidence detection들 저장
        unmatched_confirmed_by_class = {}  # 매칭안된 confirmed tracker들 저장
        
        if len(high_conf_objects) > 0:
            unique_classes = np.unique(high_conf_dets[:, 5])
            
            for cls in unique_classes:
                # 클래스별 분리
                cls_mask = high_conf_dets[:, 5] == cls
                cls_dets = high_conf_dets[cls_mask]
                cls_objects = [high_conf_objects[i] for i in range(len(high_conf_objects)) if cls_mask[i]]

                # C++과 동일: tracked만 predict, lost는 predict 안 함
                tracked_trackers = []
                lost_trackers = []
                
                if cls in self.trackers_by_class:
                    tracked_trackers.extend(self.trackers_by_class[cls])
                if cls in self.lost_trackers_by_class:
                    lost_trackers.extend(self.lost_trackers_by_class[cls])

                # tracked tracker들만 predict 호출 (C++과 동일)
                for tracker in tracked_trackers:
                    tracker.predict()
                
                # strack_pool = tracked + lost (C++의 joint_stracks와 동일)
                strack_pool = tracked_trackers + lost_trackers
                
                if len(strack_pool) == 0:
                    # tracker가 없으면 모든 detection이 unmatched
                    unmatched_high_conf_by_class[cls] = list(range(len(cls_dets)))
                    continue

                # tracker들의 현재 상태 추출 (predict는 이미 호출됨)
                trks = np.zeros((len(strack_pool), 5))
                confs = np.zeros((len(strack_pool), 1))
                
                for t in range(len(strack_pool)):
                    pos = strack_pool[t].get_state()[0]  # predict 대신 get_state 사용
                    confs[t] = strack_pool[t].get_confidence()
                    trks[t] = [pos[0], pos[1], pos[2], pos[3], confs[t, 0]]

                # confidence boost
                if self.use_dlo_boost:
                    cls_dets = self.dlo_confidence_boost(cls_dets, self.use_rich_s, self.use_sb, self.use_vt)
                if self.use_duo_boost:
                    cls_dets = self.duo_confidence_boost(cls_dets)

                # threshold filtering
                remain_inds = cls_dets[:, 4] >= self.det_thresh
                cls_dets = cls_dets[remain_inds]
                cls_objects = [cls_objects[i] for i, keep in enumerate(remain_inds) if keep]
                scores = cls_dets[:, 4]

                if len(cls_dets) == 0:
                    # 모든 detection이 threshold 미달
                    unmatched_confirmed_by_class[cls] = tracked_trackers
                    continue

                # embedding
                dets_embs = np.ones((cls_dets.shape[0], 1))
                emb_cost = None
                if self.embedder and cls_dets.size > 0:
                    dets_embs = self.embedder.compute_embedding(img_numpy, cls_dets[:, :4], tag)
                    trk_embs = []
                    for t in range(len(strack_pool)):
                        trk_embs.append(strack_pool[t].get_emb())
                    trk_embs = np.array(trk_embs)
                    if trk_embs.size > 0 and cls_dets.size > 0:
                        emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ trk_embs.reshape((trk_embs.shape[0], -1)).T

                # 첫 번째 매칭: strack_pool vs high-confidence detections
                matched, unmatched_dets, unmatched_trks, _ = associate(
                    cls_dets, trks, self.iou_threshold,
                    mahalanobis_distance=self.get_mh_dist_matrix_for_trackers(cls_dets, strack_pool),
                    track_confidence=confs, detection_confidence=scores, emb_cost=emb_cost,
                    lambda_iou=self.lambda_iou, lambda_mhd=self.lambda_mhd, lambda_shape=self.lambda_shape
                )

                trust = (cls_dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
                af = 0.95
                dets_alpha = af + (1 - af) * (1 - trust)

                # 매칭 처리 - 임시 리스트에 추가
                for m in matched:
                    det_idx, trk_idx = m[0], m[1]
                    tracker = strack_pool[trk_idx]
                    tracker.update(cls_dets[det_idx, :], scores[det_idx])
                    tracker.update_emb(dets_embs[det_idx], alpha=dets_alpha[det_idx])
                    
                    # track_id 할당
                    cls_objects[det_idx].track_id = tracker.id
                    
                    # 상태에 따른 처리 - 임시 리스트에 추가
                    if tracker.state == "Tracked":
                        # activated_stracks에 추가
                        if cls not in activated_trackers_by_class:
                            activated_trackers_by_class[cls] = []
                        activated_trackers_by_class[cls].append(tracker)
                    else:  # Lost 상태였던 tracker
                        # refind_stracks에 추가
                        tracker.re_activate(cls_dets[det_idx, :], scores[det_idx], self.frame_count)
                        if cls not in refind_trackers_by_class:
                            refind_trackers_by_class[cls] = []
                        refind_trackers_by_class[cls].append(tracker)

                # 매칭안된 것들 저장
                unmatched_high_conf_by_class[cls] = unmatched_dets
                
                # 매칭안된 confirmed tracker들만 분리 (C++의 r_tracked_stracks)
                unmatched_confirmed = []
                for trk_idx in unmatched_trks:
                    if strack_pool[trk_idx].state == "Tracked":
                        unmatched_confirmed.append(strack_pool[trk_idx])
                unmatched_confirmed_by_class[cls] = unmatched_confirmed

        # ════════════════════════════════════════════════════
        # ANCHOR Step 3: Second association - unmatched confirmed vs low-confidence
        # ═══════════════════════════════════════════════════════════════════
        if low_conf_objects:
            # 저신뢰도 객체들을 numpy 배열로 변환
            low_conf_dets = []
            for obj in low_conf_objects:
                x1, y1, x2, y2 = obj.box
                det = [x1, y1, x2, y2, obj.confidence, obj.class_id]
                low_conf_dets.append(det)
            low_conf_dets = np.array(low_conf_dets)

            low_unique_classes = np.unique(low_conf_dets[:, 5])
            
            for cls in low_unique_classes:
                if cls not in unmatched_confirmed_by_class or not unmatched_confirmed_by_class[cls]:
                    continue
                    
                # 클래스별 저신뢰도 detection 분리
                low_cls_mask = low_conf_dets[:, 5] == cls
                low_cls_dets = low_conf_dets[low_cls_mask]
                low_cls_objects = [low_conf_objects[i] for i in range(len(low_conf_objects)) if low_cls_mask[i]]
                
                unmatched_confirmed = unmatched_confirmed_by_class[cls]
                
                # 매칭안된 confirmed tracker들의 predicted position
                low_trks = np.zeros((len(unmatched_confirmed), 5))
                low_confs = np.zeros((len(unmatched_confirmed), 1))
                
                for t, tracker in enumerate(unmatched_confirmed):
                    pos = tracker.get_state()[0]
                    low_confs[t] = tracker.get_confidence()
                    low_trks[t] = [pos[0], pos[1], pos[2], pos[3], low_confs[t, 0]]

                # 두 번째 매칭: unmatched confirmed vs low-confidence (더 관대한 threshold)
                low_matched, low_unmatched_dets, low_unmatched_trks, _ = associate(
                    low_cls_dets, low_trks, self.iou_threshold * 0.8,  # 더 관대한 threshold
                    mahalanobis_distance=self.get_mh_dist_matrix_for_trackers(low_cls_dets, unmatched_confirmed),
                    track_confidence=low_confs, detection_confidence=low_cls_dets[:, 4], emb_cost=None,
                    lambda_iou=self.lambda_iou, lambda_mhd=self.lambda_mhd, lambda_shape=self.lambda_shape
                )

                # 저신뢰도 매칭 처리 - 임시 리스트에 추가
                for m in low_matched:
                    det_idx, trk_idx = m[0], m[1]
                    tracker = unmatched_confirmed[trk_idx]
                    tracker.update(low_cls_dets[det_idx, :], low_cls_dets[det_idx, 4])
                    
                    # activated_stracks에 추가
                    if cls not in activated_trackers_by_class:
                        activated_trackers_by_class[cls] = []
                    activated_trackers_by_class[cls].append(tracker)
                    
                    # track_id 할당
                    low_cls_objects[det_idx].track_id = tracker.id

                # 매칭안된 confirmed tracker들을 lost로 전환 - 임시 리스트에 추가
                for trk_idx in low_unmatched_trks:
                    tracker = unmatched_confirmed[trk_idx]
                    if tracker.state != "Lost":
                        tracker.mark_lost()
                        if cls not in lost_trackers_by_class:
                            lost_trackers_by_class[cls] = []
                        lost_trackers_by_class[cls].append(tracker)

        # ═══════════════════════════════════════════════════════════════════
        # ANCHOR Step 4: Third association - tentative vs unmatched high-confidence
        # ═══════════════════════════════════════════════════════════════════
        to_activate_by_class = {}  # probation_age 지난 tentative들
        
        for cls in unmatched_high_conf_by_class:
            unmatched_det_indices = unmatched_high_conf_by_class[cls]
            if len(unmatched_det_indices) == 0:
                continue
                
            # 해당 클래스의 매칭안된 high-confidence detection들
            cls_mask = high_conf_dets[:, 5] == cls
            cls_dets = high_conf_dets[cls_mask]
            cls_objects = [high_conf_objects[i] for i in range(len(high_conf_objects)) if cls_mask[i]]
            
            remaining_dets = cls_dets[unmatched_det_indices]
            remaining_objects = [cls_objects[i] for i in unmatched_det_indices]
            
            # tentative tracker들과 매칭
            if cls in self.tentative_trackers_by_class and self.tentative_trackers_by_class[cls]:
                tentative_trackers = self.tentative_trackers_by_class[cls]
                
                # tentative tracker들의 predicted position
                tentative_trks = np.zeros((len(tentative_trackers), 5))
                tentative_confs = np.zeros((len(tentative_trackers), 1))
                
                for t, tracker in enumerate(tentative_trackers):
                    pos = tracker.predict()[0]  # predict 호출
                    tentative_confs[t] = tracker.get_confidence()
                    tentative_trks[t] = [pos[0], pos[1], pos[2], pos[3], tentative_confs[t, 0]]

                # 세 번째 매칭: tentative vs unmatched high-confidence
                tentative_matched, tentative_unmatched_dets, tentative_unmatched_trks, _ = associate(
                    remaining_dets, tentative_trks, self.iou_threshold * 0.7,  # tentative용 관대한 threshold
                    mahalanobis_distance=self.get_mh_dist_matrix_for_trackers(remaining_dets, tentative_trackers),
                    track_confidence=tentative_confs, detection_confidence=remaining_dets[:, 4], emb_cost=None,
                    lambda_iou=self.lambda_iou, lambda_mhd=self.lambda_mhd, lambda_shape=self.lambda_shape
                )
                
                # tentative 매칭 처리
                to_activate_tracks = []
                remaining_tentative = []
                
                for m in tentative_matched:
                    det_idx, trk_idx = m[0], m[1]
                    tracker = tentative_trackers[trk_idx]
                    tracker.update(remaining_dets[det_idx, :], remaining_dets[det_idx, 4])
                    
                    # probation_age 확인
                    if self.frame_count - tracker.start_frame > self.probation_age:
                        # 활성화 대상
                        to_activate_tracks.append(tracker)
                        remaining_objects[det_idx].track_id = tracker.id  # 임시 ID 할당 (나중에 변경됨)
                    else:
                        # 아직 수습 중 - tentative_stracks에 추가
                        remaining_tentative.append(tracker)
                        remaining_objects[det_idx].track_id = tracker.id
                        if cls not in tentative_trackers_by_class:
                            tentative_trackers_by_class[cls] = []
                        tentative_trackers_by_class[cls].append(tracker)
                
                # 매칭안된 tentative들 처리
                for trk_idx in tentative_unmatched_trks:
                    tracker = tentative_trackers[trk_idx]
                    # early termination 확인
                    if self.frame_count - tracker.end_frame <= self.early_termination_age:
                        if cls not in tentative_trackers_by_class:
                            tentative_trackers_by_class[cls] = []
                        tentative_trackers_by_class[cls].append(tracker)
                    else:
                        # 조기 제거 - removed_stracks에 추가
                        tracker.mark_removed()
                        if cls not in removed_trackers_by_class:
                            removed_trackers_by_class[cls] = []
                        removed_trackers_by_class[cls].append(tracker)
                
                to_activate_by_class[cls] = to_activate_tracks
                
                # 남은 unmatched detection들 업데이트 (인덱스 변환 필요)
                final_unmatched_indices = []
                for remaining_idx in tentative_unmatched_dets:
                    original_idx = unmatched_det_indices[remaining_idx]
                    final_unmatched_indices.append(original_idx)
                unmatched_high_conf_by_class[cls] = final_unmatched_indices
            else:
                # tentative tracker가 없으면 모든 detection이 여전히 unmatched
                to_activate_by_class[cls] = []

        # ═══════════════════════════════════════════════════════════════════  
        # ANCHOR Step 5: Re-matching - lost vs to_activate
        # ═══════════════════════════════════════════════════════════════════
        for cls in to_activate_by_class:
            to_activate_tracks = to_activate_by_class[cls]
            if not to_activate_tracks:
                continue
                
            if cls in self.lost_trackers_by_class:
                already_lost_stracks = [t for t in self.lost_trackers_by_class[cls] if t.state == "Lost"]
                
                if len(already_lost_stracks) > 0:
                    # DIoU distance 계산
                    diou_distances = self.diou_distance(already_lost_stracks, to_activate_tracks)
                    
                    # Re-matching 수행 (매우 관대한 threshold)
                    rematches, unmatched_lost, unmatched_new = self.linear_assignment_simple(
                        diou_distances, threshold=999999.0
                    )
                    
                    # Re-matching 처리 - 임시 리스트에 추가 (C++과 동일한 re_match 사용)
                    for match in rematches:
                        lost_idx, new_idx = match[0], match[1]
                        lost_tracker = already_lost_stracks[lost_idx]
                        new_tracker = to_activate_tracks[new_idx]
                        
                        # lost tracker와 tentative tracker 병합 (C++ re_match와 동일)
                        lost_tracker.re_match(new_tracker, self.frame_count)
                        
                        # rematched_stracks에 추가
                        if cls not in rematched_trackers_by_class:
                            rematched_trackers_by_class[cls] = []
                        rematched_trackers_by_class[cls].append(lost_tracker)
                        
                        # new tracker 제거 표시 (C++과 동일)
                        new_tracker.state = "Removed"
                    
                    # 매칭안된 새 tracker들 활성화 - activated_stracks에 추가
                    for i in unmatched_new:
                        tracker = to_activate_tracks[i]
                        old_id = tracker.id  # 이전 음수 ID 저장
                        tracker.activate(self.frame_count, self.get_next_id)
                        if cls not in activated_trackers_by_class:
                            activated_trackers_by_class[cls] = []
                        activated_trackers_by_class[cls].append(tracker)
                        
                        # track_id 업데이트
                        for obj in high_conf_objects:
                            if hasattr(obj, 'track_id') and obj.track_id == old_id:
                                obj.track_id = tracker.id  # 새로운 양수 ID로 업데이트
                                break
                else:
                    # lost tracker가 없으면 모든 to_activate를 직접 활성화
                    for tracker in to_activate_tracks:
                        old_id = tracker.id  # 이전 음수 ID 저장
                        tracker.activate(self.frame_count, self.get_next_id)
                        if cls not in activated_trackers_by_class:
                            activated_trackers_by_class[cls] = []
                        activated_trackers_by_class[cls].append(tracker)
                        
                        # track_id 업데이트
                        for obj in high_conf_objects:
                            if hasattr(obj, 'track_id') and obj.track_id == old_id:
                                obj.track_id = tracker.id  # 새로운 양수 ID로 업데이트
                                break

        # ═══════════════════════════════════════════════════════════════════
        # ANCHOR Step 6: Init new stracks - 남은 unmatched high-confidence로 새 tentative 생성
        # ═══════════════════════════════════════════════════════════════════
        for cls in unmatched_high_conf_by_class:
            unmatched_det_indices = unmatched_high_conf_by_class[cls]
            if len(unmatched_det_indices) == 0:
                continue
                
            # 해당 클래스의 매칭안된 detection들
            cls_mask = high_conf_dets[:, 5] == cls
            cls_dets = high_conf_dets[cls_mask]
            cls_objects = [high_conf_objects[i] for i in range(len(high_conf_objects)) if cls_mask[i]]
            
            for i in unmatched_det_indices:
                if cls_dets[i, 4] >= self.det_thresh:
                    # 최대 추적 수 제한 확인
                    if self.check_max_tracks_limit(cls):
                        # 새 tentative tracker 생성
                        dets_embs = np.ones((1, 1))
                        if self.embedder:
                            dets_embs = self.embedder.compute_embedding(img_numpy, cls_dets[i:i+1, :4], tag)
                        
                        tentative_tracker = self.create_tentative_tracker(cls_dets[i, :], dets_embs[0] if self.embedder else None, cls)
                        cls_objects[i].track_id = tentative_tracker.id
                        
                        # 멤버변수 클래스 키 추가
                        if cls not in self.trackers_by_class:
                            self.trackers_by_class[cls] = []
                        if cls not in self.lost_trackers_by_class:
                            self.lost_trackers_by_class[cls] = []
                        if cls not in self.tentative_trackers_by_class:
                            self.tentative_trackers_by_class[cls] = []
                        
                        # tentative_stracks에 추가
                        if cls not in tentative_trackers_by_class:
                            tentative_trackers_by_class[cls] = []
                        tentative_trackers_by_class[cls].append(tentative_tracker)

        # ═══════════════════════════════════════════════════════════════════
        # Step 7: Update state - C++의 Step 5와 동일한 멤버변수 업데이트
        # ═══════════════════════════════════════════════════════════════════
        self._update_member_variables(
            activated_trackers_by_class, 
            lost_trackers_by_class, 
            refind_trackers_by_class, 
            rematched_trackers_by_class,
            tentative_trackers_by_class,
            removed_trackers_by_class
        )

        # ═══════════════════════════════════════════════════════════════════
        # Final output: 멤버변수 업데이트 후 최종 tracked_objects 생성 (C++과 동일)
        # ═══════════════════════════════════════════════════════════════════
        tracked_objects = []
        
        # 모든 input objects에 대해 (class_id, track_id) 복합 키로 매핑 생성
        id_to_object = {}
        for obj in objects:
            if hasattr(obj, 'track_id') and obj.track_id is not None:
                # 클래스별 독립적 ID를 위해 (class_id, track_id) 복합 키 사용
                composite_key = (obj.class_id, obj.track_id)
                id_to_object[composite_key] = obj
        
        # self.trackers_by_class에 있는 activated tracker들만 최종 결과에 포함 (C++과 동일)
        for cls, cls_trackers in self.trackers_by_class.items():
            for tracker in cls_trackers:
                if tracker.is_activated and (tracker.time_since_update < 1) and \
                   (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                    # 클래스별 독립적 ID를 위해 (class_id, track_id) 복합 키 사용
                    composite_key = (int(cls), tracker.id)
                    if composite_key in id_to_object:
                        tracked_objects.append(id_to_object[composite_key])

        return tracked_objects

    def _update_member_variables(self, activated_by_class, lost_by_class, 
                                refind_by_class, rematched_by_class, 
                                tentative_by_class, removed_by_class):
        """C++의 Step 5: Update state와 동일한 로직으로 멤버변수 업데이트"""
        
        # lost tracker들에 대해서만 max_age 체크 (C++과 동일)
        for cls in list(self.lost_trackers_by_class.keys()):
            if cls in self.lost_trackers_by_class:
                surviving_lost = []
                for trk in self.lost_trackers_by_class[cls]:
                    if trk.time_since_update <= self.max_age:  # lost만 max_age 체크
                        surviving_lost.append(trk)
                    else:
                        # max_age 초과시 removed로 전환
                        trk.mark_removed()
                        if cls not in removed_by_class:
                            removed_by_class[cls] = []
                        removed_by_class[cls].append(trk)
                self.lost_trackers_by_class[cls] = surviving_lost
        
        # tracked_stracks 정리 (swap 방식, C++과 동일)
        for cls in list(self.trackers_by_class.keys()):
            tracked_stracks_swap = []
            for trk in self.trackers_by_class[cls]:
                if trk.state == "Tracked":
                    tracked_stracks_swap.append(trk)
            
            self.trackers_by_class[cls] = tracked_stracks_swap
            
            # joint_stracks 연산으로 새로운 tracker들 합치기 (C++과 동일)
            if cls in activated_by_class:
                self.trackers_by_class[cls] = self.joint_stracks(
                    self.trackers_by_class[cls], activated_by_class[cls])
            if cls in refind_by_class:
                self.trackers_by_class[cls] = self.joint_stracks(
                    self.trackers_by_class[cls], refind_by_class[cls])
            if cls in rematched_by_class:
                self.trackers_by_class[cls] = self.joint_stracks(
                    self.trackers_by_class[cls], rematched_by_class[cls])
        
        # tentative_stracks 정리 (swap 방식, C++과 동일)
        for cls in list(self.tentative_trackers_by_class.keys()):
            tentative_stracks_swap = []
            for trk in self.tentative_trackers_by_class[cls]:
                if trk.state == "Tentative":
                    tentative_stracks_swap.append(trk)
            
            self.tentative_trackers_by_class[cls] = tentative_stracks_swap
            
            # 새로운 tentative tracker들 합치기
            if cls in tentative_by_class:
                self.tentative_trackers_by_class[cls] = self.joint_stracks(
                    self.tentative_trackers_by_class[cls], tentative_by_class[cls])
        
        # lost_stracks 업데이트 (C++과 동일)
        for cls in list(self.lost_trackers_by_class.keys()):
            # sub_stracks: lost에서 tracked된 것들 제거
            if cls in self.trackers_by_class:
                self.lost_trackers_by_class[cls] = self.sub_stracks(
                    self.lost_trackers_by_class[cls], self.trackers_by_class[cls])
            
            # 새로 lost된 tracker들 추가
            if cls in lost_by_class:
                if cls not in self.lost_trackers_by_class:
                    self.lost_trackers_by_class[cls] = []
                self.lost_trackers_by_class[cls].extend(lost_by_class[cls])
                
        # 새로운 클래스의 lost tracker들 추가
        for cls in lost_by_class:
            if cls not in self.lost_trackers_by_class:
                self.lost_trackers_by_class[cls] = lost_by_class[cls]

    def create_tentative_tracker(self, bbox, emb, cls):
        """새로운 tentative tracker 생성"""
        tracker = KalmanBoxTracker(bbox, emb, is_tentative=True, class_id=int(cls), id_counter_callback=self.get_next_id)
        tracker.tentative_init(self.frame_count)
        
        # C++과 일치: 단순히 tracker 객체만 반환 (멤버변수 조작 안함)
        return tracker
    
    def diou_distance(self, lost_trackers, new_trackers):
        """
        DIoU (Distance IoU) distance 계산
        lost_trackers: lost 상태의 tracker들 리스트
        new_trackers: 새로 활성화될 tracker들 리스트
        Returns: distance matrix (낮을수록 유사)
        """
        if len(lost_trackers) == 0 or len(new_trackers) == 0:
            return np.empty((0, 0))
        
        # lost tracker들의 bbox 추출
        lost_boxes = np.zeros((len(lost_trackers), 4))
        for i, tracker in enumerate(lost_trackers):
            bbox = tracker.get_state()[0]
            lost_boxes[i] = bbox
        
        # new tracker들의 bbox 추출
        new_boxes = np.zeros((len(new_trackers), 4))
        for i, tracker in enumerate(new_trackers):
            bbox = tracker.get_state()[0]
            new_boxes[i] = bbox
        
        # DIoU distance 계산
        diou_matrix = np.zeros((len(lost_trackers), len(new_trackers)))
        
        for i in range(len(lost_trackers)):
            for j in range(len(new_trackers)):
                lost_box = lost_boxes[i]  # [x1, y1, x2, y2]
                new_box = new_boxes[j]    # [x1, y1, x2, y2]
                
                # IoU 계산
                x1_inter = max(lost_box[0], new_box[0])
                y1_inter = max(lost_box[1], new_box[1])
                x2_inter = min(lost_box[2], new_box[2])
                y2_inter = min(lost_box[3], new_box[3])
                
                if x2_inter > x1_inter and y2_inter > y1_inter:
                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                else:
                    inter_area = 0
                
                lost_area = (lost_box[2] - lost_box[0]) * (lost_box[3] - lost_box[1])
                new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
                union_area = lost_area + new_area - inter_area
                
                iou = inter_area / (union_area + 1e-10)
                
                # 중심점 간 거리 계산
                lost_center_x = (lost_box[0] + lost_box[2]) / 2
                lost_center_y = (lost_box[1] + lost_box[3]) / 2
                new_center_x = (new_box[0] + new_box[2]) / 2
                new_center_y = (new_box[1] + new_box[3]) / 2
                
                center_distance_sq = (lost_center_x - new_center_x) ** 2 + (lost_center_y - new_center_y) ** 2
                
                # 대각선 거리 계산
                c_x1 = min(lost_box[0], new_box[0])
                c_y1 = min(lost_box[1], new_box[1])
                c_x2 = max(lost_box[2], new_box[2])
                c_y2 = max(lost_box[3], new_box[3])
                
                diagonal_distance_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
                
                # DIoU 계산
                diou = iou - (center_distance_sq / (diagonal_distance_sq + 1e-10))
                
                # Distance로 변환 (1 - DIoU, 낮을수록 유사)
                diou_matrix[i, j] = 1 - diou
        
        return diou_matrix
    
    def linear_assignment_simple(self, cost_matrix, threshold=999999.0):
        """
        간단한 linear assignment (Hungarian algorithm 대신)
        cost_matrix: NxM cost matrix
        threshold: 매칭 허용 최대 비용
        Returns: (matches, unmatched_rows, unmatched_cols)
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        matches = []
        used_rows = set()
        used_cols = set()
        
        # 비용이 낮은 순서로 정렬
        flat_indices = np.argsort(cost_matrix.ravel())
        
        for flat_idx in flat_indices:
            row, col = np.unravel_index(flat_idx, cost_matrix.shape)
            
            if row not in used_rows and col not in used_cols and cost_matrix[row, col] <= threshold:
                matches.append([row, col])
                used_rows.add(row)
                used_cols.add(col)
        
        unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in used_rows]
        unmatched_cols = [i for i in range(cost_matrix.shape[1]) if i not in used_cols]
        
        return matches, unmatched_rows, unmatched_cols
    
    def check_max_tracks_limit(self, cls):
        """
        최대 추적 수 제한 확인
        Returns: True if 새로운 tracker 생성 가능, False if 제한 도달
        """
        if self.max_num_tracks is None:
            return True  # 제한 없음
        
        # 현재 활성 tracker 수 계산
        confirmed_count = len(self.trackers_by_class.get(cls, []))
        tentative_count = len(self.tentative_trackers_by_class.get(cls, []))
        lost_count = len(self.lost_trackers_by_class.get(cls, []))
        
        # 실제 활성 tracker 수 = confirmed + tentative (lost는 비활성 상태)
        active_count = confirmed_count + tentative_count
        
        # print(f"[Track count] Class {cls}: confirmed={confirmed_count}, tentative={tentative_count}, lost={lost_count}, active={active_count}, limit={self.max_num_tracks}")
        
        return active_count < self.max_num_tracks
    
    def get_tracker_statistics(self):
        """
        tracker 통계 정보 반환 (디버깅용)
        """
        stats = {}
        for cls in set(list(self.trackers_by_class.keys()) + 
                      list(self.tentative_trackers_by_class.keys()) + 
                      list(self.lost_trackers_by_class.keys())):
            confirmed = len(self.trackers_by_class.get(cls, []))
            tentative = len(self.tentative_trackers_by_class.get(cls, []))
            lost = len(self.lost_trackers_by_class.get(cls, []))
            stats[cls] = {
                'confirmed': confirmed,
                'tentative': tentative, 
                'lost': lost,
                'total_active': confirmed + tentative
            }
        return stats

    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()

    def get_iou_matrix(self, detections: np.ndarray, buffered: bool = False) -> np.ndarray:
        # 모든 클래스의 tracker들을 수집
        all_trackers = []
        for cls_trackers in self.trackers_by_class.values():
            all_trackers.extend(cls_trackers)
        
        if len(all_trackers) == 0:
            return np.zeros((len(detections), 0))
            
        trackers = np.zeros((len(all_trackers), 5))
        for t, trk in enumerate(all_trackers):
            pos = trk.get_state()[0]
            trackers[t] = [pos[0], pos[1], pos[2], pos[3], trk.get_confidence()]

        return iou_batch(detections, trackers) if not buffered else soft_biou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, trackers_list=None, n_dims: int = 4) -> np.ndarray:
        # trackers_list가 주어지면 해당 tracker들 사용, 아니면 모든 클래스의 tracker들 사용
        if trackers_list is None:
            trackers_list = []
            for cls_trackers in self.trackers_by_class.values():
                trackers_list.extend(cls_trackers)
            
        if len(trackers_list) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(trackers_list), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = trackers_list[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1, ))[:n_dims]
        for i in range(len(trackers_list)):
            x[i] = trackers_list[i].kf.x[:n_dims]
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(np.diag(trackers_list[i].kf.covariance[:n_dims, :n_dims]))

        return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)
    
    def get_mh_dist_matrix_for_trackers(self, detections: np.ndarray, trackers_list: list, n_dims: int = 4) -> np.ndarray:
        """특정 tracker 리스트에 대한 Mahalanobis distance matrix 계산"""
        if len(trackers_list) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(trackers_list), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = trackers_list[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1, ))[:n_dims]
        for i in range(len(trackers_list)):
            x[i] = trackers_list[i].kf.x[:n_dims]
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(np.diag(trackers_list[i].kf.covariance[:n_dims, :n_dims]))

        return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        n_dims = 4
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections, trackers_list=None, n_dims=n_dims)

        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)

            mask = (min_mh_dists > limit) & (detections[:, 4] < self.det_thresh)
            boost_detections = detections[mask]
            boost_detections_args = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections, boost_detections) - np.eye(len(boost_detections))
                bdiou_max = bdiou.max(axis=1)

                remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
                args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for i in range(len(args)):
                    boxi = args[i]
                    tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                    args_tmp = np.append(np.intersect1d(boost_detections_args[args], boost_detections_args[tmp]), boost_detections_args[boxi])

                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_detections_args[boxi], 4] == conf_max:
                        remaining_boxes = np.array(remaining_boxes.tolist() + [boost_detections_args[boxi]])

                mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                mask[remaining_boxes] = True

            detections[:, 4] = np.where(mask, self.det_thresh + 1e-4, detections[:, 4])

        return detections

    def dlo_confidence_boost(self, detections: np.ndarray, use_rich_sim: bool, use_soft_boost: bool, use_varying_th: bool) -> np.ndarray:
        sbiou_matrix = self.get_iou_matrix(detections, True)
        if sbiou_matrix.size == 0:
            return detections
            
        # 모든 클래스의 tracker들을 수집
        all_trackers = []
        for cls_trackers in self.trackers_by_class.values():
            all_trackers.extend(cls_trackers)
            
        if len(all_trackers) == 0:
            return detections
            
        trackers = np.zeros((len(all_trackers), 6))
        for t, trk in enumerate(all_trackers):
            pos = trk.get_state()[0]
            trackers[t] = [pos[0], pos[1], pos[2], pos[3], 0, trk.time_since_update - 1]

        if use_rich_sim:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections, trackers_list=None), 1)
            shape_sim = shape_similarity(detections, trackers)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False)

        if not use_soft_boost and not use_varying_th:
            max_s = S.max(1)
            coef = self.dlo_boost_coef
            detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

        else:
            if use_soft_boost:
                max_s = S.max(1)
                alpha = 0.65
                detections[:, 4] = np.maximum(detections[:, 4], alpha*detections[:, 4] + (1-alpha)*max_s**(1.5))
            if use_varying_th:
                threshold_s = 0.95
                threshold_e = 0.8
                n_steps = 20
                alpha = (threshold_s - threshold_e) / n_steps
                tmp = (S > np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)).max(1)
                scores = deepcopy(detections[:, 4])
                scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)

                detections[:, 4] = scores

        return detections
