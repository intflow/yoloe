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

    count = 0  # 정식 tracker용 counter (양수 ID)
    tentative_count = 0  # tentative tracker용 counter (음수 ID)

    def __init__(self, bbox, emb: Optional[np.ndarray] = None, is_tentative: bool = False):
        """
        Initialises a tracker using initial bounding box.
        """

        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        
        # ID 할당 - tentative와 confirmed 완전히 분리
        if is_tentative:
            KalmanBoxTracker.tentative_count += 1
            self.id = -KalmanBoxTracker.tentative_count  # 음수 ID
            self.is_tentative_id = True
        else:
            KalmanBoxTracker.count += 1
            self.id = KalmanBoxTracker.count  # 양수 ID
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
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb
    
    def activate(self, frame_count):
        """tracker를 정식 활성화"""
        # tentative tracker였다면 새로운 confirmed ID 할당
        if self.is_tentative_id:
            KalmanBoxTracker.count += 1
            old_id = self.id
            self.id = KalmanBoxTracker.count  # 새로운 양수 ID
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
        
    def tentative_init(self, frame_count):
        """tentative 모드로 초기화"""
        self.start_frame = frame_count
        self.state = "Tentative"
        self.is_activated = False


class BoostTrack(object):
    def __init__(self, video_name: Optional[str] = None, det_thresh: Optional[float] = None, iou_threshold: Optional[float] = None, max_num_tracks: Optional[int] = None):
        self.frame_count = 0
        self.trackers: List[KalmanBoxTracker] = []
        self.trackers_by_class = {}  # Dictionary to store trackers by class
        
        # Tentative 모드를 위한 컨테이너들
        self.tentative_trackers_by_class = {}  # 수습 중인 tracker들
        self.lost_trackers_by_class = {}  # lost된 tracker들
        
        # Tentative 모드 설정
        self.probation_age = 3  # 수습 기간 (프레임)
        self.early_termination_age = 1  # 조기 제거 기간
        self.max_num_tracks = max_num_tracks  # 최대 추적 수 (None이면 무제한)

        self.max_age = GeneralSettings.max_age(video_name)
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

    def update(self, objects, img_numpy, tag):
        """
        ObjectMeta 리스트를 받아서 tracking하고 track_id를 부여하여 반환
        1. 정상 confidence (>= 0.1) 객체들끼리 클래스별 매칭
        2. 매칭되지 않은 tracker들과 저신뢰도 객체들을 매칭
        """
        if not objects:
            return []

        self.frame_count += 1

        # 정상 confidence와 저신뢰도 객체들 분리
        normal_objects = []
        low_conf_objects = []
        
        for obj in objects:
            if obj.confidence >= 0.1:
                normal_objects.append(obj)
            elif 0.01 <= obj.confidence < 0.1 and obj.class_name == 'object':
                low_conf_objects.append(obj)

        # 정상 객체들을 numpy 배열로 변환
        if normal_objects:
            normal_dets = []
            for obj in normal_objects:
                x1, y1, x2, y2 = obj.box
                det = [x1, y1, x2, y2, obj.confidence, obj.class_id]
                normal_dets.append(det)
            normal_dets = np.array(normal_dets)
        else:
            normal_dets = np.empty((0, 6))

        if self.ecc is not None:
            transform = self.ecc(img_numpy, self.frame_count, tag)
            for cls_trackers in self.trackers_by_class.values():
                for trk in cls_trackers:
                    trk.camera_update(transform)

        tracked_objects = []
        unmatched_trackers_info = []  # 매칭되지 않은 tracker들 정보 저장

        # ═══════════════════════════════════════════════════════════════════
        # 단계 1: 정상 confidence 객체들끼리 클래스별 매칭
        # ═══════════════════════════════════════════════════════════════════
        if len(normal_objects) > 0:
            unique_classes = np.unique(normal_dets[:, 5])
            
            for cls in unique_classes:
                # Get detections for current class
                cls_mask = normal_dets[:, 5] == cls
                cls_dets = normal_dets[cls_mask]
                cls_objects = [obj for i, obj in enumerate(normal_objects) if normal_dets[i, 5] == cls]

                # Get trackers for current class
                if cls not in self.trackers_by_class:
                    self.trackers_by_class[cls] = []
                cls_trackers = self.trackers_by_class[cls]

                # get predicted locations from existing trackers (confirmed only)
                trks = np.zeros((len(cls_trackers), 5))
                confs = np.zeros((len(cls_trackers), 1))

                for t in range(len(trks)):
                    pos = cls_trackers[t].predict()[0]
                    confs[t] = cls_trackers[t].get_confidence()
                    trks[t] = [pos[0], pos[1], pos[2], pos[3], confs[t, 0]]
                
                # tentative tracker들도 예측 수행
                if cls in self.tentative_trackers_by_class:
                    for tentative_tracker in self.tentative_trackers_by_class[cls]:
                        tentative_tracker.predict()

                if self.use_dlo_boost:
                    cls_dets = self.dlo_confidence_boost(cls_dets, self.use_rich_s, self.use_sb, self.use_vt)

                if self.use_duo_boost:
                    cls_dets = self.duo_confidence_boost(cls_dets)

                remain_inds = cls_dets[:, 4] >= self.det_thresh
                cls_dets = cls_dets[remain_inds]
                cls_objects = [cls_objects[i] for i, keep in enumerate(remain_inds) if keep]
                scores = cls_dets[:, 4]

                # Generate embeddings
                dets_embs = np.ones((cls_dets.shape[0], 1))
                emb_cost = None
                if self.embedder and cls_dets.size > 0:
                    dets_embs = self.embedder.compute_embedding(img_numpy, cls_dets[:, :4], tag)
                    trk_embs = []
                    for t in range(len(cls_trackers)):
                        trk_embs.append(cls_trackers[t].get_emb())
                    trk_embs = np.array(trk_embs)
                    if trk_embs.size > 0 and cls_dets.size > 0:
                        emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ trk_embs.reshape((trk_embs.shape[0], -1)).T
                emb_cost = None if self.embedder is None else emb_cost

                matched, unmatched_dets, unmatched_trks, sym_matrix = associate(
                    cls_dets,
                    trks,
                    self.iou_threshold,
                    mahalanobis_distance=self.get_mh_dist_matrix(cls_dets, cls_trackers),
                    track_confidence=confs,
                    detection_confidence=scores,
                    emb_cost=emb_cost,
                    lambda_iou=self.lambda_iou,
                    lambda_mhd=self.lambda_mhd,
                    lambda_shape=self.lambda_shape
                )

                trust = (cls_dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
                af = 0.95
                dets_alpha = af + (1 - af) * (1 - trust)

                # 매칭된 detection들에 track_id 부여
                for m in matched:
                    det_idx, trk_idx = m[0], m[1]
                    cls_trackers[trk_idx].update(cls_dets[det_idx, :], scores[det_idx])
                    cls_trackers[trk_idx].update_emb(dets_embs[det_idx], alpha=dets_alpha[det_idx])
                    
                    # ObjectMeta에 track_id 부여 (confirmed tracker는 양수)
                    cls_objects[det_idx].track_id = cls_trackers[trk_idx].id

                # 매칭되지 않은 detection들은 tentative tracker로 생성
                for i in unmatched_dets:
                    if cls_dets[i, 4] >= self.det_thresh:
                        # 최대 추적 수 제한 확인
                        if self.check_max_tracks_limit(cls):
                            # tentative tracker 생성 (바로 confirmed tracker로 만들지 않음)
                            tentative_tracker = self.create_tentative_tracker(cls_dets[i, :], dets_embs[i], cls)
                            # ObjectMeta에 tentative track_id 부여 (이미 음수 ID)
                            cls_objects[i].track_id = tentative_tracker.id
                        # else:
                        #     print(f"[Max tracks limit] Class {cls}: Cannot create new tracker, limit reached")
                        else:
                            pass

                # 매칭되지 않은 tracker들을 lost 상태로 전환
                for trk_idx in unmatched_trks:
                    if trk_idx < len(cls_trackers):
                        tracker = cls_trackers[trk_idx]
                        if tracker.time_since_update < self.max_age:  # 아직 유효한 tracker만
                            # lost 상태로 전환
                            tracker.mark_lost()
                            
                            # lost tracker 컨테이너로 이동
                            if cls not in self.lost_trackers_by_class:
                                self.lost_trackers_by_class[cls] = []
                            self.lost_trackers_by_class[cls].append(tracker)
                            
                            # 저신뢰도 객체와의 매칭을 위해 정보 저장
                            unmatched_trackers_info.append({
                                'tracker': tracker,
                                'class': cls,
                                'trackers_list': cls_trackers,
                                'tracker_idx': trk_idx
                            })
                
                # lost로 전환된 tracker들을 confirmed tracker 리스트에서 제거
                confirmed_trackers = []
                for i, trk in enumerate(cls_trackers):
                    if i not in unmatched_trks or trk.time_since_update >= self.max_age:
                        confirmed_trackers.append(trk)
                self.trackers_by_class[cls] = confirmed_trackers

                # 활성 tracker들에 해당하는 ObjectMeta만 결과에 추가
                for i, obj in enumerate(cls_objects):
                    if obj.track_id is not None:
                        # 정식 tracker 확인
                        if obj.track_id > 0:  # 양수면 정식 tracker
                            for trk in cls_trackers:
                                if trk.id == obj.track_id:
                                    if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                                        tracked_objects.append(obj)
                                    break
                        else:  # 음수면 tentative tracker
                            if cls in self.tentative_trackers_by_class:
                                for tentative_trk in self.tentative_trackers_by_class[cls]:
                                    if tentative_trk.id == obj.track_id:
                                        if (tentative_trk.time_since_update < 1) and tentative_trk.hit_streak >= 1:
                                            tracked_objects.append(obj)
                                        break

        # ═══════════════════════════════════════════════════════════════════
        # 단계 1.5: Tentative tracker들과 매칭되지 않은 detection들 매칭
        # ═══════════════════════════════════════════════════════════════════
        if len(normal_objects) > 0:
            unique_classes = np.unique(normal_dets[:, 5])
            
            for cls in unique_classes:
                # tentative tracker들 처리
                if cls in self.tentative_trackers_by_class and len(self.tentative_trackers_by_class[cls]) > 0:
                    tentative_trackers = self.tentative_trackers_by_class[cls]
                    
                    # 매칭되지 않은 detection들 중에서 tentative tracker들과 매칭할 대상들 찾기
                    remaining_dets = []
                    remaining_objects = []
                    
                    for i, obj in enumerate(normal_objects):
                        if normal_dets[np.where(np.array([o == obj for o in normal_objects]))[0][0], 5] == cls and obj.track_id is None:
                            remaining_dets.append(normal_dets[np.where(np.array([o == obj for o in normal_objects]))[0][0]])
                            remaining_objects.append(obj)
                    
                    if remaining_dets:
                        remaining_dets = np.array(remaining_dets)
                        
                        # tentative tracker들의 predicted position 추출
                        tentative_trks = np.zeros((len(tentative_trackers), 5))
                        tentative_confs = np.zeros((len(tentative_trackers), 1))
                        
                        for t, tracker in enumerate(tentative_trackers):
                            pos = tracker.get_state()[0]
                            tentative_confs[t] = tracker.get_confidence()
                            tentative_trks[t] = [pos[0], pos[1], pos[2], pos[3], tentative_confs[t, 0]]
                        
                        # tentative tracker들과 남은 detection들 매칭 (더 관대한 threshold)
                        tentative_matched, tentative_unmatched_dets, tentative_unmatched_trks, _ = associate(
                            remaining_dets,
                            tentative_trks,
                            self.iou_threshold * 0.7,  # tentative용 더 관대한 threshold
                            mahalanobis_distance=self.get_mh_dist_matrix_for_trackers(remaining_dets, tentative_trackers),
                            track_confidence=tentative_confs,
                            detection_confidence=remaining_dets[:, 4],
                            emb_cost=None,  # tentative는 embedding 비용 사용 안함
                            lambda_iou=self.lambda_iou,
                            lambda_mhd=self.lambda_mhd,
                            lambda_shape=self.lambda_shape
                        )
                        
                        # 매칭된 tentative tracker들 업데이트
                        for m in tentative_matched:
                            det_idx, trk_idx = m[0], m[1]
                            tentative_tracker = tentative_trackers[trk_idx]
                            tentative_tracker.update(remaining_dets[det_idx, :], remaining_dets[det_idx, 4])
                            
                            # ObjectMeta에 tentative track_id 부여 (이미 음수 ID)
                            remaining_objects[det_idx].track_id = tentative_tracker.id
                            
                            # 수습 기간 통과했으면 정식 활성화
                            if self.frame_count - tentative_tracker.start_frame >= self.probation_age:
                                tentative_tracker.activate(self.frame_count)
                                # confirmed tracker로 이동
                                if cls not in self.trackers_by_class:
                                    self.trackers_by_class[cls] = []
                                self.trackers_by_class[cls].append(tentative_tracker)
                                # 정식 track_id로 변경 (activate에서 새로운 양수 ID 할당됨)
                                remaining_objects[det_idx].track_id = tentative_tracker.id
                
                # tentative tracker들 정리 (수습 기간 통과한 것들 제거)
                to_activate = self.process_tentative_trackers(cls)
                
                # 매칭 실패한 tentative tracker들 조기 제거
                self.remove_failed_tentative_trackers(cls)
                
                # ═══════════════════════════════════════════════════════════════════
                # Re-matching: lost tracker들과 새로 활성화될 tracker들 매칭
                # ═══════════════════════════════════════════════════════════════════
                if to_activate and cls in self.lost_trackers_by_class and len(self.lost_trackers_by_class[cls]) > 0:
                    lost_trackers = self.lost_trackers_by_class[cls]
                    
                    # DIoU distance 계산
                    diou_distances = self.diou_distance(lost_trackers, to_activate)
                    
                    # Re-matching 수행 (매우 관대한 threshold로 거의 모든 매칭 허용)
                    rematches, unmatched_lost, unmatched_new = self.linear_assignment_simple(
                        diou_distances, threshold=999999.0
                    )
                    
                    # Re-matching 성공한 경우들 처리
                    rematched_trackers = []
                    for match in rematches:
                        lost_idx, new_idx = match[0], match[1]
                        lost_tracker = lost_trackers[lost_idx]
                        new_tracker = to_activate[new_idx]
                        
                        # lost tracker를 새 detection으로 재활성화
                        new_bbox = new_tracker.get_state()[0]
                        lost_tracker.re_activate(new_bbox, new_tracker.kf.x[4] if len(new_tracker.kf.x) > 4 else 0.5, self.frame_count)
                        
                        # lost tracker를 confirmed tracker로 복귀
                        if cls not in self.trackers_by_class:
                            self.trackers_by_class[cls] = []
                        self.trackers_by_class[cls].append(lost_tracker)
                        rematched_trackers.append(lost_tracker)
                        
                        # new tracker는 제거 표시 (ID 재사용됨)
                        new_tracker.state = "Removed"
                        
                        # print(f"[Re-match] Class {cls}: Lost tracker {lost_tracker.id} re-matched with new tracker {new_tracker.id}")
                    
                    # Re-matching된 lost tracker들을 lost 리스트에서 제거
                    remaining_lost = [lost_trackers[i] for i in range(len(lost_trackers)) if i not in [m[0] for m in rematches]]
                    self.lost_trackers_by_class[cls] = remaining_lost
                    
                    # Re-matching되지 않은 새 tracker들만 confirmed로 추가
                    for i in unmatched_new:
                        new_tracker = to_activate[i]
                        new_tracker.activate(self.frame_count)
                        if cls not in self.trackers_by_class:
                            self.trackers_by_class[cls] = []
                        self.trackers_by_class[cls].append(new_tracker)
                        # print(f"[New tracker] Class {cls}: New tracker {new_tracker.id} activated")
                        
                elif to_activate:
                    # Lost tracker가 없는 경우 모든 tentative tracker를 confirmed로 활성화
                    for new_tracker in to_activate:
                        new_tracker.activate(self.frame_count)
                        if cls not in self.trackers_by_class:
                            self.trackers_by_class[cls] = []
                        self.trackers_by_class[cls].append(new_tracker)
                        # print(f"[Direct activate] Class {cls}: Tracker {new_tracker.id} activated")

        # ═══════════════════════════════════════════════════════════════════
        # 단계 2: 저신뢰도 객체들과 매칭되지 않은 tracker들을 매칭
        # ═══════════════════════════════════════════════════════════════════
        if low_conf_objects and unmatched_trackers_info:
            
            # 저신뢰도 객체들을 numpy 배열로 변환
            low_dets = []
            for obj in low_conf_objects:
                x1, y1, x2, y2 = obj.box
                det = [x1, y1, x2, y2, obj.confidence, obj.class_id]
                low_dets.append(det)
            low_dets = np.array(low_dets)

            # 매칭되지 않은 tracker들의 predicted position 추출
            unmatched_trks = np.zeros((len(unmatched_trackers_info), 5))
            unmatched_confs = np.zeros((len(unmatched_trackers_info), 1))
            
            for i, info in enumerate(unmatched_trackers_info):
                tracker = info['tracker']
                pos = tracker.get_state()[0]
                unmatched_confs[i] = tracker.get_confidence()
                unmatched_trks[i] = [pos[0], pos[1], pos[2], pos[3], unmatched_confs[i, 0]]

            # 저신뢰도 객체들과 매칭되지 않은 tracker들 간 매칭
            low_scores = low_dets[:, 4]
            
            # 매칭 수행 (더 관대한 threshold 사용)
            low_matched, low_unmatched_dets, low_unmatched_trks, _ = associate(
                low_dets,
                unmatched_trks,
                self.iou_threshold * 0.8,  # 저신뢰도 객체용 더 관대한 threshold
                mahalanobis_distance=self.get_mh_dist_matrix_for_trackers(low_dets, [info['tracker'] for info in unmatched_trackers_info]),
                track_confidence=unmatched_confs,
                detection_confidence=low_scores,
                emb_cost=None,  # 저신뢰도 객체는 embedding 비용 사용 안함
                lambda_iou=self.lambda_iou,
                lambda_mhd=self.lambda_mhd,
                lambda_shape=self.lambda_shape
            )

            # 매칭된 저신뢰도 객체들에 track_id 부여
            for m in low_matched:
                det_idx, trk_idx = m[0], m[1]
                tracker_info = unmatched_trackers_info[trk_idx]
                tracker = tracker_info['tracker']
                
                # tracker 업데이트
                tracker.update(low_dets[det_idx, :], low_scores[det_idx])
                
                # ObjectMeta에 track_id 부여
                low_conf_objects[det_idx].track_id = tracker.id

            # 매칭된 저신뢰도 객체들을 결과에 추가
            for i, obj in enumerate(low_conf_objects):
                if obj.track_id is not None:
                    # tracker 상태 확인
                    for info in unmatched_trackers_info:
                        if info['tracker'].id == obj.track_id:
                            tracker = info['tracker']
                            if (tracker.time_since_update < 1) and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                                tracked_objects.append(obj)
                            break

            # 매칭되지 않은 저신뢰도 객체들은 버림 (새로운 tracker 생성하지 않음)
            # (디버깅 출력 제거)

        # 매칭되지 않은 저신뢰도 객체들은 버림 (새로운 tracker 생성하지 않음)
        elif low_conf_objects:
            pass  # 저신뢰도 객체들을 버림

        # ═══════════════════════════════════════════════════════════════════
        # 단계 3: 죽은 tracker들 제거
        # ═══════════════════════════════════════════════════════════════════
        # 정식 tracker들 제거
        for cls, cls_trackers in self.trackers_by_class.items():
            i = len(cls_trackers)
            for trk in reversed(cls_trackers):
                if trk.time_since_update > self.max_age:
                    cls_trackers.pop(i-1)
                i -= 1
        
        # tentative tracker들 제거 (더 엄격한 기준)
        for cls in list(self.tentative_trackers_by_class.keys()):
            if cls in self.tentative_trackers_by_class:
                tentative_trackers = self.tentative_trackers_by_class[cls]
                surviving_tentative = []
                
                for trk in tentative_trackers:
                    # tentative tracker는 더 빨리 제거 (max_age의 절반)
                    if trk.time_since_update <= self.max_age // 2:
                        surviving_tentative.append(trk)
                
                self.tentative_trackers_by_class[cls] = surviving_tentative
        
        # lost tracker들 제거 (시간 기반)
        for cls in list(self.lost_trackers_by_class.keys()):
            if cls in self.lost_trackers_by_class:
                lost_trackers = self.lost_trackers_by_class[cls]
                surviving_lost = []
                
                for trk in lost_trackers:
                    # lost tracker는 max_age * 2 까지 유지 (re-matching 기회 제공)
                    if trk.time_since_update <= self.max_age * 2:
                        surviving_lost.append(trk)
                    else:
                        # print(f"[Remove lost] Class {cls}: Lost tracker {trk.id} permanently removed")
                        pass
                
                self.lost_trackers_by_class[cls] = surviving_lost

        return tracked_objects
    
    def process_tentative_trackers(self, cls):
        """tentative tracker들을 처리하여 활성화할 tracker들 반환"""
        if cls not in self.tentative_trackers_by_class:
            self.tentative_trackers_by_class[cls] = []
            
        tentative_trackers = self.tentative_trackers_by_class[cls]
        to_activate = []
        remaining_tentative = []
        
        for tracker in tentative_trackers:
            if tracker.state == "Tentative":
                # 수습 기간 통과 여부 확인
                if self.frame_count - tracker.start_frame >= self.probation_age:
                    to_activate.append(tracker)
                else:
                    remaining_tentative.append(tracker)
        
        # tentative tracker 리스트 업데이트
        self.tentative_trackers_by_class[cls] = remaining_tentative
        
        return to_activate
    
    def remove_failed_tentative_trackers(self, cls):
        """매칭 실패한 tentative tracker들을 조기 제거"""
        if cls not in self.tentative_trackers_by_class:
            return
            
        tentative_trackers = self.tentative_trackers_by_class[cls]
        surviving_trackers = []
        
        for tracker in tentative_trackers:
            # 조기 제거 조건 확인
            if tracker.time_since_update <= self.early_termination_age:
                surviving_trackers.append(tracker)
            # else: 조기 제거됨 (surviving_trackers에 추가하지 않음)
                
        self.tentative_trackers_by_class[cls] = surviving_trackers
    
    def create_tentative_tracker(self, bbox, emb, cls):
        """새로운 tentative tracker 생성"""
        tracker = KalmanBoxTracker(bbox, emb, is_tentative=True)  # 🔥 is_tentative=True로 설정
        tracker.tentative_init(self.frame_count)
        
        if cls not in self.tentative_trackers_by_class:
            self.tentative_trackers_by_class[cls] = []
        self.tentative_trackers_by_class[cls].append(tracker)
        
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
