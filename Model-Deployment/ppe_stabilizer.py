"""
PPE Temporal Stabilizer & Anti-Flicker Tracking Engine (with Strict Anti-Duplicate NMS)
---------------------------------------------------------------------------------------
Solves:
1. Detection flickering and rapid confidence toggling across video frames.
2. Bounding box jitter using exponential moving average (EMA) smoothing.
3. Multiple overlapping bounding boxes on the same object (Hardhat, Vest, Person, Mask).
4. Conflicting positive vs negative boxes on the same anatomical region (e.g. Hardhat & NO-Hardhat).
5. False-alarm strobing using temporal consensus hysteresis.
"""

from collections import deque, Counter
import cv2
import numpy as np

CATEGORY_MAP = {
    'hardhat': 'head',
    'helmet': 'head',
    'no-hardhat': 'head',
    'no-helmet': 'head',
    'cap': 'head',
    'safety vest': 'torso',
    'vest': 'torso',
    'no-safety vest': 'torso',
    'no-vest': 'torso',
    'mask': 'face',
    'no-mask': 'face',
    'person': 'person',
    'safety cone': 'equipment',
    'cone': 'equipment',
    'machinery': 'equipment',
    'vehicle': 'equipment'
}

def get_category(cls_name):
    c = str(cls_name).lower().strip()
    for key, cat in CATEGORY_MAP.items():
        if key in c:
            return cat
    return c

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def box_io_min(boxA, boxB):
    """Intersection over minimum area (detects if one box is substantially inside another)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / min(areaA, areaB)

def box_intersection_ratio(inner_box, outer_box):
    """Check how much of inner_box is inside outer_box"""
    xA = max(inner_box[0], outer_box[0])
    yA = max(inner_box[1], outer_box[1])
    xB = min(inner_box[2], outer_box[2])
    yB = min(inner_box[3], outer_box[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    inner_area = max(1.0, (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1]))
    return inter / inner_area

def suppress_duplicates(detections, iou_thresh=0.20, io_min_thresh=0.48):
    """
    Non-Maximum Suppression (NMS) pass to ensure that no two bounding boxes
    overlap on the same physical object or category.
    """
    if not detections:
        return []
    
    # Sort detections by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.get('confidence', 0), reverse=True)
    kept = []
    
    for det in sorted_dets:
        det_cat = get_category(det['class'])
        det_box = det['bbox']
        
        duplicate = False
        for k in kept:
            k_cat = get_category(k['class'])
            if det_cat == k_cat:
                iou = calculate_iou(det_box, k['bbox'])
                io_min = box_io_min(det_box, k['bbox'])
                if iou > iou_thresh or io_min > io_min_thresh:
                    duplicate = True
                    break
        
        if not duplicate:
            kept.append(det)
            
    return kept


class TrackedItem:
    def __init__(self, track_id, class_name, bbox, conf, is_cap=False):
        self.track_id = track_id
        self.class_name = class_name
        self.category = get_category(class_name)
        self.bbox = [float(v) for v in bbox]
        self.conf = float(conf)
        self.is_cap = is_cap
        self.cap_history = deque([1 if is_cap else 0], maxlen=7)
        self.hits = 1
        self.hit_streak = 1
        self.missing_frames = 0
        self.history_conf = deque([conf], maxlen=10)
        self.class_history = deque([(class_name, conf)], maxlen=6)

    def update(self, bbox, conf, new_class=None, alpha=0.72, is_cap=False):
        # Exponential moving average for smooth bounding box movement without jitter
        self.bbox = [
            alpha * cur + (1.0 - alpha) * prev
            for cur, prev in zip(bbox, self.bbox)
        ]
        self.conf = alpha * conf + (1.0 - alpha) * self.conf
        self.hits += 1
        self.hit_streak += 1
        self.missing_frames = 0
        self.history_conf.append(conf)

        # Debounced cap state with hysteresis filtering
        # Prevents momentary 1-2 frame flashes when worker turns head left or right
        self.cap_history.append(1 if is_cap else 0)
        cap_ratio = sum(self.cap_history) / max(1, len(self.cap_history))
        if cap_ratio >= 0.60 and len(self.cap_history) >= 3:
            self.is_cap = True
        elif cap_ratio <= 0.30:
            self.is_cap = False

        if new_class is not None:
            self.class_history.append((new_class, conf))
            # Determine consensus class from recent history
            if self.is_cap:
                self.class_name = 'NO-Hardhat'
                self.category = 'head'
            else:
                # If incoming detection is a confident Hardhat or Vest, transition immediately
                if new_class.lower() in ['hardhat', 'safety vest', 'vest'] and conf >= 0.30:
                    self.class_name = new_class
                    self.category = get_category(new_class)
                elif new_class.lower() in ['no-hardhat', 'no-safety vest'] and conf >= 0.55:
                    self.class_name = new_class
                    self.category = get_category(new_class)
                else:
                    recent_classes = [c for c, _ in self.class_history]
                    c_counts = Counter(recent_classes)
                    most_common_cls, count = c_counts.most_common(1)[0]
                    
                    pos_classes = [c for c in recent_classes if not c.lower().startswith(('no-', 'no ', 'no_'))]
                    if len(pos_classes) >= max(1, len(recent_classes) * 0.35) and not self.class_name.lower().startswith(('no-', 'no ', 'no_')):
                        pass
                    elif count >= (len(recent_classes) + 1) // 2:
                        self.class_name = most_common_cls
                        self.category = get_category(most_common_cls)

    def mark_missed(self):
        self.missing_frames += 1
        self.hit_streak = 0
        self.conf *= 0.94  # gentle confidence decay


class PersonComplianceTracker:
    """Tracks a single person's PPE compliance over time to eliminate flicker and false-missing flashes"""
    def __init__(self, person_id):
        self.person_id = person_id
        self.helmet_history = deque(maxlen=18)
        self.vest_history = deque(maxlen=18)
        self.mask_history = deque(maxlen=18)
        self.no_helmet_direct = deque(maxlen=12)
        self.no_vest_direct = deque(maxlen=12)
        self.consecutive_missing_helmet = 0
        self.consecutive_missing_vest = 0
        self.consecutive_missing_mask = 0
        self.frames_seen = 0
        self.missing_ppe = []
        self.is_compliant = True

    def update(self, has_helmet, has_vest, has_mask, direct_no_helmet=False, direct_no_vest=False):
        self.frames_seen += 1
        self.helmet_history.append(1 if has_helmet else 0)
        self.vest_history.append(1 if has_vest else 0)
        self.mask_history.append(1 if has_mask else 0)
        self.no_helmet_direct.append(1 if direct_no_helmet else 0)
        self.no_vest_direct.append(1 if direct_no_vest else 0)

        if has_helmet:
            self.consecutive_missing_helmet = 0
        else:
            self.consecutive_missing_helmet += 1

        if has_vest:
            self.consecutive_missing_vest = 0
        else:
            self.consecutive_missing_vest += 1

        if has_mask:
            self.consecutive_missing_mask = 0
        else:
            self.consecutive_missing_mask += 1

    def evaluate(self, required_ppe):
        missing = []
        
        # Warm-up grace period: require at least 8 frames of presence before declaring missing PPE
        # (Prevents initial momentary "Missing" flash when a person first walks into view)
        if self.frames_seen < 8:
            self.missing_ppe = []
            self.is_compliant = True
            return True, []

        # Helmet check: requires sustained absence AND consecutive missing frames
        if required_ppe.get('helmet', True):
            h_ratio = sum(self.helmet_history) / max(1, len(self.helmet_history))
            no_h_direct_ratio = sum(self.no_helmet_direct) / max(1, len(self.no_helmet_direct))
            
            # Requires at least 5 consecutive missing frames and sustained low presence (< 25%),
            # OR strong direct detection of NO-Hardhat from the model (ratio >= 65%)
            if (self.consecutive_missing_helmet >= 5 and h_ratio < 0.25) or no_h_direct_ratio >= 0.65:
                missing.append('helmet')

        # Vest check: requires sustained absence AND consecutive missing frames
        if required_ppe.get('vest', True):
            v_ratio = sum(self.vest_history) / max(1, len(self.vest_history))
            no_v_direct_ratio = sum(self.no_vest_direct) / max(1, len(self.no_vest_direct))
            
            if (self.consecutive_missing_vest >= 5 and v_ratio < 0.25) or no_v_direct_ratio >= 0.65:
                missing.append('vest')

        # Mask check
        if required_ppe.get('mask', False):
            m_ratio = sum(self.mask_history) / max(1, len(self.mask_history))
            if self.consecutive_missing_mask >= 5 and m_ratio < 0.25:
                missing.append('mask')

        self.missing_ppe = missing
        self.is_compliant = (len(missing) == 0)
        return self.is_compliant, self.missing_ppe


class PPETemporalStabilizer:
    """
    Complete Temporal Smoothing, Tracking and Anti-Flicker Filter for PPE Detection.
    Maintains persistent object identities, smooths bounding boxes, and prevents
    both rapid oscillating false-alarms AND multiple duplicate bounding boxes on the same object.
    """
    def __init__(self, iou_match_thresh=0.35, max_grace_frames=5, alpha=0.72, min_person_height_ratio=0.18):
        self.iou_match_thresh = iou_match_thresh
        self.max_grace_frames = max_grace_frames
        self.alpha = alpha
        self.min_person_height_ratio = min_person_height_ratio
        self.tracks = {}  # track_id -> TrackedItem
        self.person_compliance = {}  # track_id -> PersonComplianceTracker
        self.next_track_id = 1
        
        # Debounce / hysteresis for overall compliance verdict
        self.compliance_history = deque(maxlen=6)
        self.current_stable_compliant = True

    def reset(self):
        self.tracks.clear()
        self.person_compliance.clear()
        self.next_track_id = 1
        self.compliance_history.clear()
        self.current_stable_compliant = True

    def _deduplicate_tracks(self):
        """
        Merge and prune overlapping tracks representing the same physical object or category.
        """
        to_delete = set()
        track_list = list(self.tracks.items())
        
        for i in range(len(track_list)):
            tid1, t1 = track_list[i]
            if tid1 in to_delete:
                continue
            for j in range(i + 1, len(track_list)):
                tid2, t2 = track_list[j]
                if tid2 in to_delete:
                    continue
                
                # Check if same category (e.g. both head, both torso, both person)
                if t1.category == t2.category:
                    iou = calculate_iou(t1.bbox, t2.bbox)
                    io_min = box_io_min(t1.bbox, t2.bbox)
                    
                    if iou > 0.22 or io_min > 0.52:
                        t1_is_neg = t1.class_name.lower().startswith(('no-', 'no ', 'no_'))
                        t2_is_neg = t2.class_name.lower().startswith(('no-', 'no ', 'no_'))
                        
                        # Decide winner
                        if t1_is_neg != t2_is_neg:
                            # Conflict: Positive vs Negative
                            if not t1_is_neg and (t1.conf >= 0.35 or t1.hits >= t2.hits):
                                winner, loser_id = t1, tid2
                            elif not t2_is_neg and (t2.conf >= 0.35 or t2.hits >= t1.hits):
                                winner, loser_id = t2, tid1
                            elif t1.conf >= t2.conf:
                                winner, loser_id = t1, tid2
                            else:
                                winner, loser_id = t2, tid1
                        else:
                            # Same polarity (e.g. two Hardhats or two Persons)
                            if (t1.hits, t1.conf) >= (t2.hits, t2.conf):
                                winner, loser_id = t1, tid2
                            else:
                                winner, loser_id = t2, tid1
                        
                        # Merge hits and confidence into winner
                        winner.hits += 1
                        winner.conf = max(winner.conf, self.tracks[loser_id].conf)
                        if getattr(t1, 'is_cap', False) or getattr(t2, 'is_cap', False):
                            winner.is_cap = True
                            winner.class_name = 'NO-Hardhat'
                        to_delete.add(loser_id)

        for tid in to_delete:
            if tid in self.tracks:
                del self.tracks[tid]
            if tid in self.person_compliance:
                del self.person_compliance[tid]

    def update(self, raw_detections, frame_shape, required_ppe=None, mode='single'):
        """
        Takes raw YOLO detections from the current frame and returns
        temporally smoothed, stabilized, and strictly deduplicated detections.
        mode: 'single' (focuses on primary foreground worker, zero background interference)
              'multi' (monitors all foreground workers simultaneously)
        """
        if required_ppe is None:
            required_ppe = {'helmet': True, 'vest': True, 'mask': False}

        h_img, w_img = frame_shape[:2]
        min_person_h = h_img * self.min_person_height_ratio

        # First, filter and clean raw detections
        valid_raw = []
        for det in raw_detections:
            conf = det.get('confidence', 0.0)
            if conf < 0.22:
                continue
            bw = det['bbox'][2] - det['bbox'][0]
            bh = det['bbox'][3] - det['bbox'][1]
            if bw < 14 or bh < 14:
                continue
            valid_raw.append(det)

        # Apply intra-frame NMS on raw detections to strip raw duplicates before tracking
        sorted_dets = suppress_duplicates(valid_raw, iou_thresh=0.35, io_min_thresh=0.60)

        # 1. Match raw detections with existing tracks
        matched_tracks = set()
        matched_dets = set()

        # Pass A: Exact Class match
        for d_idx, det in enumerate(sorted_dets):
            det_cls = det['class']
            det_cat = get_category(det_cls)
            det_box = det['bbox']
            # Torso boxes are large, so allow matching at IoU >= 0.20 or io_min >= 0.35
            min_score = 0.20 if det_cat == 'torso' else self.iou_match_thresh
            best_score = min_score
            best_tid = None

            for tid, track in self.tracks.items():
                if tid in matched_tracks:
                    continue
                if track.class_name.lower() == det_cls.lower():
                    iou = calculate_iou(det_box, track.bbox)
                    io_min = box_io_min(det_box, track.bbox)
                    score = max(iou, io_min * 0.75)
                    if score > best_score:
                        best_score = score
                        best_tid = tid

            if best_tid is not None:
                matched_tracks.add(best_tid)
                matched_dets.add(d_idx)
                self.tracks[best_tid].update(det_box, det['confidence'], det['class'], self.alpha, is_cap=det.get('is_cap', False))

        # Pass B: Category match for remaining unmatched detections
        # (Prevents creating two parallel tracks when class shifts between Hardhat and NO-Hardhat)
        for d_idx, det in enumerate(sorted_dets):
            if d_idx in matched_dets:
                continue
            det_cat = get_category(det['class'])
            det_box = det['bbox']
            min_score = 0.20 if det_cat == 'torso' else 0.28
            best_score = min_score
            best_tid = None

            for tid, track in self.tracks.items():
                if tid in matched_tracks:
                    continue
                if track.category == det_cat:
                    iou = calculate_iou(det_box, track.bbox)
                    io_min = box_io_min(det_box, track.bbox)
                    score = max(iou, io_min * 0.75)
                    if score > best_score:
                        best_score = score
                        best_tid = tid

            if best_tid is not None:
                matched_tracks.add(best_tid)
                matched_dets.add(d_idx)
                self.tracks[best_tid].update(det_box, det['confidence'], det['class'], self.alpha, is_cap=det.get('is_cap', False))

        # 2. Create new tracks for unmatched detections
        for d_idx, det in enumerate(sorted_dets):
            if d_idx not in matched_dets:
                conf = det.get('confidence', 0.0)
                det_cat = get_category(det['class'])
                det_box = det['bbox']

                # Reject spawning if this detection overlaps with ANY existing active track of the same category!
                overlaps_existing = False
                for t in self.tracks.values():
                    if t.missing_frames <= 2 and t.category == det_cat:
                        if calculate_iou(det_box, t.bbox) > 0.20 or box_io_min(det_box, t.bbox) > 0.45:
                            overlaps_existing = True
                            break
                if overlaps_existing:
                    continue

                # Spawning threshold: 0.25 to pick up vests and helmets swiftly without hesitation
                if conf >= 0.25:
                    tid = self.next_track_id
                    self.next_track_id += 1
                    self.tracks[tid] = TrackedItem(tid, det['class'], det['bbox'], conf, is_cap=det.get('is_cap', False))
                    matched_tracks.add(tid)

        # 2.5 Deduplicate existing tracks
        self._deduplicate_tracks()

        # 3. Handle unmatched tracks (grace period for temporary dropouts)
        dead_tracks = []
        for tid, track in list(self.tracks.items()):
            if tid not in matched_tracks:
                track.mark_missed()
                if track.missing_frames > self.max_grace_frames or (track.hits < 2 and track.missing_frames > 3):
                    dead_tracks.append(tid)

        for tid in dead_tracks:
            if tid in self.tracks:
                del self.tracks[tid]
            if tid in self.person_compliance:
                del self.person_compliance[tid]

        # 4. Filter active tracked Persons and PPE items
        all_persons = [t for t in self.tracks.values() if t.class_name.lower() == 'person' and t.missing_frames <= 2]
        foreground_persons = [p for p in all_persons if (p.bbox[3] - p.bbox[1]) >= min_person_h]
        ppe_tracks = [t for t in self.tracks.values() if t.class_name.lower() != 'person' and t.missing_frames <= 2]

        stabilized_detections = []
        synthetic_violations = []

        # 5. Stabilize Person PPE Association
        for p in foreground_persons:
            pid = p.track_id
            if pid not in self.person_compliance:
                self.person_compliance[pid] = PersonComplianceTracker(pid)

            px1, py1, px2, py2 = p.bbox
            pw = px2 - px1
            ph = py2 - py1

            # Head and torso zones (generous bounds covering hardhat above head and vest shoulders/stomach)
            head_box = [px1 - 0.12 * pw, py1 - 0.22 * ph, px2 + 0.12 * pw, py1 + 0.44 * ph]
            torso_box = [px1 - 0.15 * pw, py1 + 0.14 * ph, px2 + 0.15 * pw, max(py2, int(py1 + 1.10 * ph))]

            # Check if there is an active direct model detection for torso PPE
            has_direct_torso = any(
                item.category == 'torso' and (box_intersection_ratio(item.bbox, torso_box) > 0.05 or calculate_iou(item.bbox, torso_box) > 0.05)
                for item in ppe_tracks
            )
            # Torso is visible if directly detected by YOLO OR if person box is tall enough (ph >= 160 and ph >= 0.50 * pw)
            has_visible_torso = has_direct_torso or ((ph >= 160) and (ph >= 0.50 * pw))

            has_helmet = False
            direct_no_helmet = False
            has_vest = False
            direct_no_vest = False
            has_mask = False

            for item in ppe_tracks:
                icat = item.category
                icls = item.class_name.lower()
                is_neg = icls.startswith(('no-', 'no ', 'no_'))

                if icat == 'head':
                    if not is_neg and (box_intersection_ratio(item.bbox, head_box) > 0.08 or calculate_iou(item.bbox, head_box) > 0.08):
                        has_helmet = True
                    elif is_neg and (box_intersection_ratio(item.bbox, head_box) > 0.08 or calculate_iou(item.bbox, head_box) > 0.08):
                        direct_no_helmet = True

                if icat == 'torso':
                    if not is_neg and (box_intersection_ratio(item.bbox, torso_box) > 0.05 or calculate_iou(item.bbox, torso_box) > 0.05):
                        has_vest = True
                    elif is_neg and (box_intersection_ratio(item.bbox, torso_box) > 0.05 or calculate_iou(item.bbox, torso_box) > 0.05):
                        direct_no_vest = True

                if icat == 'face':
                    if not is_neg and (box_intersection_ratio(item.bbox, head_box) > 0.08 or calculate_iou(item.bbox, head_box) > 0.08):
                        has_mask = True

            comp_tracker = self.person_compliance[pid]
            # If torso is not in the frame, do not falsely penalize worker for missing vest
            effective_has_vest = has_vest or (not has_visible_torso)
            comp_tracker.update(has_helmet, effective_has_vest, has_mask, direct_no_helmet, direct_no_vest)
            is_comp, missing = comp_tracker.evaluate(required_ppe)
            if not has_visible_torso and 'vest' in missing:
                missing.remove('vest')

            # If sustained missing, synthesize stable violation box ONLY IF torso is visible AND no active item exists
            if 'vest' in missing and not has_vest and has_visible_torso:
                # Ensure no existing vest track overlaps this person's torso
                has_any_vest = any(
                    item.category == 'torso' and not item.class_name.lower().startswith(('no-', 'no ', 'no_'))
                    and (box_intersection_ratio(item.bbox, torso_box) > 0.08 or calculate_iou(item.bbox, torso_box) > 0.08)
                    for item in ppe_tracks
                )
                if not has_any_vest:
                    synthetic_violations.append({
                        'id': 9000 + pid * 2,
                        'class': 'NO-Safety Vest',
                        'confidence': round(p.conf * 0.90, 2),
                        'is_violation': True,
                        'is_inferred': True,
                        'note': f'ไม่พบเสื้อกั๊กบนคนงาน #{pid}',
                        'bbox': [round(px1 + 0.06 * pw, 1), round(py1 + 0.22 * ph, 1), round(px2 - 0.06 * pw, 1), round(py1 + 0.72 * ph, 1)]
                    })

            if 'helmet' in missing and not has_helmet:
                has_any_helmet = any(
                    item.category == 'head' and not item.class_name.lower().startswith(('no-', 'no ', 'no_'))
                    and (box_intersection_ratio(item.bbox, head_box) > 0.08 or calculate_iou(item.bbox, head_box) > 0.08)
                    for item in ppe_tracks
                )
                if not has_any_helmet:
                    synthetic_violations.append({
                        'id': 9000 + pid * 2 + 1,
                        'class': 'NO-Hardhat',
                        'confidence': round(p.conf * 0.90, 2),
                        'is_violation': True,
                        'is_inferred': True,
                        'note': f'ไม่พบหมวกนิรภัยบนคนงาน #{pid}',
                        'bbox': [round(px1 + 0.06 * pw, 1), round(py1, 1), round(px2 - 0.06 * pw, 1), round(py1 + 0.35 * ph, 1)]
                    })

            if 'mask' in missing and not has_mask and required_ppe.get('mask', False):
                has_any_mask = any(
                    item.category == 'face' and not item.class_name.lower().startswith(('no-', 'no ', 'no_'))
                    and (box_intersection_ratio(item.bbox, head_box) > 0.08 or calculate_iou(item.bbox, head_box) > 0.08)
                    for item in ppe_tracks
                )
                if not has_any_mask:
                    synthetic_violations.append({
                        'id': 9000 + pid * 2 + 2,
                        'class': 'NO-Mask',
                        'confidence': round(p.conf * 0.88, 2),
                        'is_violation': True,
                        'is_inferred': True,
                        'note': f'ไม่พบหน้ากากบนคนงาน #{pid}',
                        'bbox': [round(px1 + 0.15 * pw, 1), round(py1 + 0.18 * ph, 1), round(px2 - 0.15 * pw, 1), round(py1 + 0.38 * ph, 1)]
                    })

        # 6. Format initial stabilized detections
        for tid, t in self.tracks.items():
            if t.hits >= 2 or t.conf >= 0.45:
                is_neg = t.class_name.lower().startswith(('no-', 'no ', 'no_')) or getattr(t, 'is_cap', False)
                cls_to_show = 'NO-Hardhat' if getattr(t, 'is_cap', False) else t.class_name
                stabilized_detections.append({
                    'id': tid,
                    'class': cls_to_show,
                    'confidence': round(t.conf, 2),
                    'is_violation': is_neg,
                    'is_cap': getattr(t, 'is_cap', False),
                    'is_inferred': False,
                    'note': 'หมวกแก๊ป (Cap Detected)' if getattr(t, 'is_cap', False) else None,
                    'bbox': [round(v, 1) for v in t.bbox]
                })

        for sv in synthetic_violations:
            stabilized_detections.append(sv)

        # 7. Single Mode vs Multi Mode Strict Spatial Pruning
        if mode == 'single' and foreground_persons:
            # Single mode: focus EXCLUSIVELY on primary worker
            primary_p = max(foreground_persons, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))
            pid = primary_p.track_id
            
            px1, py1, px2, py2 = primary_p.bbox
            pw, ph = px2 - px1, py2 - py1
            
            p_head_box = [px1 - 0.12 * pw, py1 - 0.22 * ph, px2 + 0.12 * pw, py1 + 0.44 * ph]
            p_torso_box = [px1 - 0.15 * pw, py1 + 0.14 * ph, px2 + 0.15 * pw, max(py2, int(py1 + 1.10 * ph))]

            single_filtered = []
            
            # 1. Primary person box
            person_det = next((d for d in stabilized_detections if d['class'].lower() == 'person' and d.get('id') == pid), None)
            if person_det:
                single_filtered.append(person_det)

            # 2. Select at most ONE best headwear item (Hardhat or NO-Hardhat)
            head_candidates = [
                d for d in stabilized_detections 
                if get_category(d['class']) == 'head' and (box_intersection_ratio(d['bbox'], p_head_box) > 0.08 or calculate_iou(d['bbox'], p_head_box) > 0.08)
            ]
            if head_candidates:
                # If there's a cap candidate, it takes precedence as violation
                cap_heads = [d for d in head_candidates if d.get('is_cap')]
                if cap_heads:
                    best_head = max(cap_heads, key=lambda d: d.get('confidence', 0))
                else:
                    pos_heads = [d for d in head_candidates if not d.get('is_violation', False)]
                    if pos_heads:
                        best_head = max(pos_heads, key=lambda d: d.get('confidence', 0))
                    else:
                        best_head = max(head_candidates, key=lambda d: d.get('confidence', 0))
                single_filtered.append(best_head)

            # 3. Select at most ONE best torso item (Safety Vest or NO-Safety Vest)
            torso_candidates = [
                d for d in stabilized_detections 
                if get_category(d['class']) == 'torso' and (box_intersection_ratio(d['bbox'], p_torso_box) > 0.05 or calculate_iou(d['bbox'], p_torso_box) > 0.05)
            ]
            if torso_candidates:
                # If a real Safety Vest is detected, ALWAYS show it immediately without dropping
                pos_torsos = [d for d in torso_candidates if not d.get('is_violation', False)]
                if pos_torsos:
                    best_torso = max(pos_torsos, key=lambda d: d.get('confidence', 0))
                    single_filtered.append(best_torso)
                else:
                    # Direct model detection of NO-Safety Vest: always show immediately
                    direct_neg_torsos = [d for d in torso_candidates if not d.get('is_inferred', False)]
                    if direct_neg_torsos:
                        best_torso = max(direct_neg_torsos, key=lambda d: d.get('confidence', 0))
                        single_filtered.append(best_torso)
                    elif (ph >= 160) and (ph >= 0.50 * pw):
                        # Inferred synthetic missing box: show only if torso area is actually visible
                        best_torso = max(torso_candidates, key=lambda d: d.get('confidence', 0))
                        single_filtered.append(best_torso)

            # 4. Select at most ONE best mask item (if required)
            if required_ppe.get('mask', False):
                face_candidates = [
                    d for d in stabilized_detections 
                    if get_category(d['class']) == 'face' and (box_intersection_ratio(d['bbox'], p_head_box) > 0.08 or calculate_iou(d['bbox'], p_head_box) > 0.08)
                ]
                if face_candidates:
                    best_face = max(face_candidates, key=lambda d: d.get('confidence', 0))
                    single_filtered.append(best_face)

            stabilized_detections = single_filtered
        else:
            # Multi mode: final Anti-Duplicate NMS across all items
            stabilized_detections = suppress_duplicates(stabilized_detections, iou_thresh=0.22, io_min_thresh=0.48)

        # 8. Overall compliance evaluation with hysteresis debouncing
        has_person = len(foreground_persons) > 0
        all_violations = [d for d in stabilized_detections if d.get('is_violation', False)]
        
        if mode == 'single' and foreground_persons:
            primary_p = max(foreground_persons, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))
            pid = primary_p.track_id
            if pid in self.person_compliance:
                p_comp, p_missing = self.person_compliance[pid].evaluate(required_ppe)
                instant_compliant = p_comp
                active_missing = p_missing
            else:
                instant_compliant = True
                active_missing = []
        else:
            instant_compliant = (len(all_violations) == 0) if has_person else True
            active_missing = list(set([
                'helmet' if 'no-hardhat' in d['class'].lower() else ('vest' if 'no-safety vest' in d['class'].lower() else ('mask' if 'no-mask' in d['class'].lower() else d['class']))
                for d in all_violations
            ]))
        
        self.compliance_history.append(1 if instant_compliant else 0)

        # Hysteresis: requires >= 65% consensus over rolling window to flip state
        comp_ratio = sum(self.compliance_history) / max(1, len(self.compliance_history))
        if comp_ratio >= 0.65:
            self.current_stable_compliant = True
        elif comp_ratio <= 0.35:
            self.current_stable_compliant = False

        detected_ppe = list(set([
            d['class'] for d in stabilized_detections 
            if d['class'].lower() != 'person' and not d.get('is_violation', False)
        ]))

        summary = {
            'has_person': has_person,
            'person_count': 1 if mode == 'single' and has_person else len(foreground_persons),
            'is_compliant': self.current_stable_compliant,
            'missing_ppe': active_missing if not self.current_stable_compliant else [],
            'detected_ppe': detected_ppe,
            'detections': stabilized_detections
        }
        return stabilized_detections, summary

    def render(self, frame, detections, summary=None, show_labels=True, show_top_banner=True):
        """
        Render silky smooth, non-flickering, non-overlapping color-coded bounding boxes onto the frame.
        Green = Compliant PPE (Hardhat, Safety Vest, Mask)
        Red = Non-compliant / Missing PPE (NO-Hardhat, NO-Safety Vest)
        Blue = Person
        """
        annotated = frame.copy()
        
        color_map = {
            'hardhat': (0, 220, 0),        # Vibrant Green
            'safety vest': (0, 220, 0),    # Vibrant Green
            'vest': (0, 220, 0),
            'mask': (0, 220, 0),
            'person': (255, 140, 0),       # Blue / Cyan
            'no-hardhat': (0, 0, 240),     # Bright Red
            'no-safety vest': (0, 0, 240), # Bright Red
            'no-mask': (0, 165, 255),      # Orange
            'safety cone': (0, 215, 255),  # Yellow
            'machinery': (128, 128, 128),
            'vehicle': (128, 128, 128)
        }

        for det in detections:
            # Skip drawing synthetic inferred [Missing] boxes on live video (Clean View)
            if det.get('is_inferred'):
                continue

            cls_name = det['class']
            conf = det['confidence']
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            is_cap = det.get('is_cap', False) or ('(cap)' in cls_name.lower()) or (cls_name.lower() == 'cap') or ('cap' in str(det.get('note', '')).lower())
            is_violation = det.get('is_violation', False) or is_cap

            color = (0, 0, 240) if is_violation else color_map.get(cls_name.lower(), (0, 220, 0))
            thickness = 2

            # Draw clean rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            if show_labels:
                display_name = cls_name
                if is_cap:
                    if display_name.lower() in ['cap', 'hardhat', 'no-hardhat']:
                        display_name = "NO-Hardhat (cap)"
                    elif '(cap)' not in display_name.lower():
                        display_name = f"{display_name} (cap)"

                lbl = f"{display_name} {int(conf * 100)}%"
                if det.get('is_inferred'):
                    lbl += " [Missing]"
                
                # Label badge with background
                (lbl_w, lbl_h), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                lbl_y1 = max(0, y1 - lbl_h - 8)
                lbl_y2 = y1
                lbl_x2 = min(frame.shape[1], x1 + lbl_w + 8)
                
                cv2.rectangle(annotated, (x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
                cv2.putText(annotated, lbl, (x1 + 4, lbl_y2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw overall top status badge if enabled
        if show_top_banner and summary is not None and summary.get('has_person'):
            is_comp = summary.get('is_compliant', True)
            status_text = "ALL PPE COMPLIANT" if is_comp else f"NON-COMPLIANT: Missing {', '.join(summary.get('missing_ppe', []))}"
            badge_color = (0, 180, 0) if is_comp else (0, 0, 220)
            
            overlay = annotated.copy()
            cv2.rectangle(overlay, (20, 20), (min(frame.shape[1] - 20, 480), 65), badge_color, -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
            cv2.putText(annotated, status_text, (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated
