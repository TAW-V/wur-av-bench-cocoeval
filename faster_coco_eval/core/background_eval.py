# Background instance evaluation extension for faster-coco-eval
# Original work Copyright (c) 2025 TAW-V

import logging
from typing import Dict, List, Set, Tuple, Union

import numpy as np

from .faster_eval_api import COCOeval_faster

logger = logging.getLogger(__name__)


class BackgroundClassMapper:
    """Utility class for handling foreground/background class ID mapping."""
    
    @staticmethod
    def get_base_class_id(class_id: int) -> int:
        """Extract the base class ID (first 4 digits) from a 5-digit ID.
        
        Args:
            class_id (int): The 5-digit class ID
            
        Returns:
            int: The base class ID (first 4 digits)
        """
        if class_id < 10000:
            # Handle legacy 4-digit IDs by padding with 0
            return class_id * 10
        return (class_id // 10) * 10
    
    @staticmethod
    def is_background_class(class_id: int) -> bool:
        """Check if a class ID represents a background class.
        
        Args:
            class_id (int): The class ID to check
            
        Returns:
            bool: True if background (ends in 1), False if foreground (ends in 0)
        """
        return (class_id & 1) == 1
    
    @staticmethod
    def is_foreground_class(class_id: int) -> bool:
        """Check if a class ID represents a foreground class.
        
        Args:
            class_id (int): The class ID to check
            
        Returns:
            bool: True if foreground (ends in 0), False if background (ends in 1)
        """
        return (class_id & 1) == 0
    
    @staticmethod
    def get_foreground_id(base_class_id: int) -> int:
        """Get the foreground class ID for a base class.
        
        Args:
            base_class_id (int): The base class ID
            
        Returns:
            int: The foreground class ID (base_id + 0)
        """
        return BackgroundClassMapper.get_base_class_id(base_class_id) + 0
    
    @staticmethod
    def get_background_id(base_class_id: int) -> int:
        """Get the background class ID for a base class.
        
        Args:
            base_class_id (int): The base class ID
            
        Returns:
            int: The background class ID (base_id + 1)
        """
        return BackgroundClassMapper.get_base_class_id(base_class_id) + 1
    
    @staticmethod
    def are_same_base_class(class_id_1: int, class_id_2: int) -> bool:
        """Check if two class IDs belong to the same base class.
        
        Args:
            class_id_1 (int): First class ID
            class_id_2 (int): Second class ID
            
        Returns:
            bool: True if both IDs have the same base class
        """
        return BackgroundClassMapper.get_base_class_id(class_id_1) == \
               BackgroundClassMapper.get_base_class_id(class_id_2)


class COCOeval_background(COCOeval_faster):
    """Extended COCO evaluation with background instance support.
    
    This class extends COCOeval_faster to handle background instances according to:
    - Background instances have class IDs ending in 1 (foreground end in 0)
    - Class IDs are 5 digits where last digit is the fg/bg boolean
    - Predictions matched with foreground OR background annotations = TP
    - False negatives are only counted for foreground objects
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize background evaluation with same parameters as COCOeval_faster."""
        super().__init__(*args, **kwargs)
        self.class_mapper = BackgroundClassMapper()
        
    def _prepare_background_categories(self):
        """Prepare category mapping for foreground/background classes.
        
        This method analyzes the ground truth and detection categories to identify
        foreground/background pairs and create appropriate mappings.
        """
        # Get all category IDs from ground truth and detections
        gt_cat_ids = set(self.params.catIds) if self.params.catIds else set()
        if self.cocoGt:
            gt_cat_ids.update(self.cocoGt.getCatIds())
        if self.cocoDt:
            gt_cat_ids.update(self.cocoDt.getCatIds())
            
        # Group categories by base class
        self.base_class_groups = {}
        for cat_id in gt_cat_ids:
            base_id = self.class_mapper.get_base_class_id(cat_id)
            if base_id not in self.base_class_groups:
                self.base_class_groups[base_id] = {'foreground': [], 'background': []}
                
            if self.class_mapper.is_foreground_class(cat_id):
                self.base_class_groups[base_id]['foreground'].append(cat_id)
            else:
                self.base_class_groups[base_id]['background'].append(cat_id)
        
        logger.info(f"Prepared {len(self.base_class_groups)} base class groups for background evaluation")
        
    def computeIoU(self, imgId: int, catId: int) -> Union[List[float], np.ndarray]:
        """Compute IoU between ground truth and detection for background-aware matching.
        
        This method extends the parent computeIoU to allow matching between
        foreground predictions and background annotations (and vice versa) of the same base class.
        
        Args:
            imgId (int): Image ID
            catId (int): Category ID (can be foreground or background, or -1 for useCats=False)
            
        Returns:
            Union[List[float], np.ndarray]: IoUs between gt and dt for the given image and category
        """
        p = self.params
        
        # Handle useCats=False case (catId=-1)
        if catId == -1 or not p.useCats:
            # Get all instances across all categories (both foreground and background)
            gt_instances = self.gt_dataset.get_instances([imgId], p.catIds, False)[0][0]
            dt_instances = self.dt_dataset.get_instances([imgId], p.catIds, False)[0][0]
            
            gt = list(gt_instances)
            dt = list(dt_instances)
        else:
            # useCats=True: Get base class ID for this category
            base_class_id = self.class_mapper.get_base_class_id(catId)
            
            # Get all related categories (both foreground and background) for this base class
            related_cat_ids = []
            if base_class_id in self.base_class_groups:
                related_cat_ids.extend(self.base_class_groups[base_class_id]['foreground'])
                related_cat_ids.extend(self.base_class_groups[base_class_id]['background'])
            else:
                raise ValueError(f"Base class {base_class_id} not found in base_class_groups")
                
            # Get ground truth and detection instances for all related categories
            gt = []
            dt = []
            
            for rel_cat_id in related_cat_ids:
                gt_instances = self.gt_dataset.get_instances([imgId], [rel_cat_id], True)[0][0]
                dt_instances = self.dt_dataset.get_instances([imgId], [rel_cat_id], True)[0][0]
                
                gt.extend(gt_instances)
                dt.extend(dt_instances)

        ####################
        #Below part is copied from core/cocoeval.py
        ####################
        
        if len(gt) == 0 or len(dt) == 0:
            return []
            
        # Sort detections by score
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]
        
        # Compute IoUs using the parent class method
        if p.compute_rle:
            g = [g["rle"] for g in gt]
            d = [d["rle"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise ValueError(f"p.iouType must be bbox or segm or boundary. Get {p.iouType}")
            
        from .mask import iou as maskUtils_iou
        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        ious = maskUtils_iou(d, g, iscrowd)
        
        if p.compute_boundary:
            g_b = [g["boundary"] for g in gt]
            d_b = [d["boundary"] for d in dt]
            
            boundary_ious = maskUtils_iou(d_b, g_b, iscrowd)
            boundary_ious = np.array(boundary_ious)
            iscrowd = np.array(iscrowd)
            if len(gt) and len(dt):
                ious[:, iscrowd == 0] = np.minimum(ious[:, iscrowd == 0], boundary_ious[:, iscrowd == 0])
            else:
                ious = np.minimum(ious, boundary_ious)
                
        return ious
    
    def math_matches(self):
        """Analyze matches with background-aware false negative calculation.
        
        This method extends the parent math_matches to only count false negatives
        for foreground objects, not background objects.
        """
        # Call parent method for basic TP/FP assignment
        for dt_gt, iou in self.eval["matched"].items():
            dt_id, gt_id = dt_gt.split("_")
            
            dt_id = int(dt_id)
            gt_id = int(gt_id)
            
            _gt_ann = self.cocoGt.anns[gt_id]
            _dt_ann = self.cocoDt.anns[dt_id]
            
            _dt_ann["tp"] = True
            _dt_ann["gt_id"] = gt_id
            _dt_ann["iou"] = iou
            _gt_ann["dt_id"] = dt_id
        
        # Mark unmatched detections as FP
        for dt_id in self.cocoDt.anns.keys():
            if self.cocoDt.anns[dt_id].get("gt_id") is None:
                self.cocoDt.anns[dt_id]["fp"] = True
        
        # Mark unmatched ground truths as FN, but ONLY for foreground objects
        for gt_id in self.cocoGt.anns.keys():
            gt_ann = self.cocoGt.anns[gt_id]
            if gt_ann.get("dt_id") is None:
                # Only mark as FN if this is a foreground class
                if self.class_mapper.is_foreground_class(gt_ann["category_id"]):
                    gt_ann["fn"] = True
                else:
                    # Background objects that are unmatched are not counted as FN
                    logger.debug(f"Skipping FN for background annotation {gt_id} of class {gt_ann['category_id']}")
                    
    def evaluate(self):
        """Run evaluation with background instance support."""
        self._prepare_background_categories()
        
        p = self.params
        original_catIds = p.catIds.copy() if p.catIds else []
        original_useCats = p.useCats
        
        # Strategy: We need background GTs to be matchable (for TP credit) but not count toward FN
        # For useCats=True: Filter to foreground catIds, mark background as ignored
        # For useCats=False: Include ALL catIds, mark background as ignored but special handling in computeIoU
        
        # if original_useCats:
        #     # useCats=True: Only evaluate foreground categories
        #     # Background GTs marked as ignored won't participate in matching for their own category
        #     # But they can still match via cross-class matching in computeIoU override
        #     if original_catIds:
        #         foreground_catIds = [cat_id for cat_id in original_catIds 
        #                             if self.class_mapper.is_foreground_class(cat_id)]
        #         p.catIds = foreground_catIds
        #         self.params = p
        #         logger.info(f"Filtered to {len(foreground_catIds)} foreground categories for useCats=True evaluation")
        # else:
        #     # useCats=False: Keep all catIds but we'll handle background specially
        #     # Don't filter catIds - we need both foreground and background for matching
        #     logger.info(f"Using all {len(original_catIds)} categories for useCats=False evaluation")
        
        # Temporarily change category_id of background GTs to their foreground equivalent
        # This allows them to participate in matching without being counted in recall denominator
        # We'll mark them with a special flag so we can identify them later
        background_cat_changes = []
        if self.cocoGt:
            for ann_id, ann in self.cocoGt.anns.items():
                if self.class_mapper.is_background_class(ann["category_id"]):
                    original_cat_id = ann["category_id"]
                    foreground_cat_id = self.class_mapper.get_foreground_id(original_cat_id)
                    # Store for restoration
                    background_cat_changes.append((ann_id, original_cat_id))
                    # Store original category ID for later use
                    ann["_original_cat_id"] = original_cat_id
                    # Temporarily change to foreground category
                    ann["category_id"] = foreground_cat_id
                    # Mark as ignored so it doesn't count in recall denominator
                    ann["ignore"] = ann.get("ignore", 0)  # Preserve original if exists
                    ann["_bg_original_ignore"] = ann["ignore"]
                    ann["ignore"] = 1
                    ann["_is_background_gt"] = True  # Flag for identification
        
        # Run evaluation with modified categories
        super().evaluate()
        
        # Restore original category IDs and ignore flags
        p.catIds = original_catIds
        p.useCats = original_useCats
        self.params = p
        
        if self.cocoGt:
            for ann_id, original_cat_id in background_cat_changes:
                ann = self.cocoGt.anns[ann_id]
                ann["category_id"] = original_cat_id
                # Restore original ignore flag
                if "_bg_original_ignore" in ann:
                    ann["ignore"] = ann["_bg_original_ignore"]
                    del ann["_bg_original_ignore"]
                # Keep _is_background_gt and _original_cat_id flags for later use
                # They will be used in accumulate() and get_background_stats()
        
    def accumulate(self):
        """Accumulate evaluation results and fix recall calculation for background GTs.
        
        The parent accumulate() calculates recall using all GTs (including background).
        We need to recalculate it using only foreground GTs.
        """
        # Call parent accumulate
        super().accumulate()
        
        # Fix recall calculation to use only foreground GTs
        if not self.params.useCats:
            # For useCats=False, we need to recalculate recall
            # The C++ code calculated recall as: matched / (foreground + background GTs)
            # We want: foreground_matched / foreground_GTs
            
            # Count foreground and background GTs
            fg_count = sum(1 for ann in self.cocoGt.anns.values() 
                          if self.class_mapper.is_foreground_class(ann.get("_original_cat_id", ann["category_id"])))
            total_count = len(self.cocoGt.anns)
            
            if fg_count > 0 and total_count > fg_count:
                # Recalculate recall for each IoU threshold
                old_recall = self.eval["recall"].copy()
                
                if 'matched' in self.eval and self.extra_calc:
                    # Method 1: Use matched dict (when extra_calc=True)
                    # We need to recalculate the precision array based on foreground-only recall
                    
                    # Step 1: Get all matched pairs with their scores and GT types
                    matches_info = []
                    for dt_gt_pair, iou in self.eval['matched'].items():
                        dt_id_str, gt_id_str = dt_gt_pair.split('_')
                        dt_id = int(dt_id_str)
                        gt_id = int(gt_id_str)
                        
                        if dt_id in self.cocoDt.anns and gt_id in self.cocoGt.anns:
                            score = self.cocoDt.anns[dt_id]['score']
                            gt_cat = self.cocoGt.anns[gt_id].get("_original_cat_id", self.cocoGt.anns[gt_id]["category_id"])
                            is_fg = self.class_mapper.is_foreground_class(gt_cat)
                            matches_info.append({
                                'score': score,
                                'is_foreground': is_fg,
                                'iou': iou,
                                'dt_id': dt_id,
                                'gt_id': gt_id
                            })
                    
                    # Step 2: Sort by score (descending)
                    matches_info.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Step 3: Count total predictions and FPs
                    total_preds = len(self.cocoDt.anns)
                    matched_dt_ids = set(m['dt_id'] for m in matches_info)
                    fp_count = total_preds - len(matched_dt_ids)
                    
                    # Step 4: For each recall threshold, calculate precision
                    P_new = np.zeros_like(self.eval['precision'])
                    
                    for r_idx, r_thr in enumerate(self.params.recThrs):
                        # Find the minimum number of foreground GTs we need to match to achieve recall >= r_thr
                        fg_needed = int(np.ceil(r_thr * fg_count))
                        
                        # Go through matches in score order and count until we have enough FG matches
                        fg_matched = 0
                        bg_matched = 0
                        for i, match in enumerate(matches_info):
                            if match['is_foreground']:
                                fg_matched += 1
                            else:
                                bg_matched += 1
                            
                            if fg_matched >= fg_needed:
                                # We've achieved the required recall
                                total_tp = fg_matched + bg_matched
                                # FP = total predictions up to this point - total TP
                                # But we need to count how many predictions have score >= this threshold
                                min_score = match['score']
                                preds_above_threshold = sum(1 for ann in self.cocoDt.anns.values() 
                                                           if ann['score'] >= min_score)
                                fp_at_threshold = preds_above_threshold - total_tp
                                
                                # Precision = TP / (TP + FP)
                                precision = total_tp / (total_tp + fp_at_threshold) if (total_tp + fp_at_threshold) > 0 else 0
                                
                                # Store for all IoU thresholds (we only have one in most cases)
                                for t_idx in range(len(self.params.iouThrs)):
                                    P_new[t_idx, r_idx, 0, 0, 0] = precision
                                break
                        else:
                            # Couldn't achieve this recall threshold. Use 0, like the
                            # standard accumulate does, so the area under the curve is
                            # not reduced by unreachable recall levels.
                            for t_idx in range(len(self.params.iouThrs)):
                                P_new[t_idx, r_idx, 0, 0, 0] = 0
                    
                    # Update precision array
                    self.eval['precision'] = P_new
                    
                    # Update recall
                    fg_matches = sum(1 for m in matches_info if m['is_foreground'])
                    corrected_recall = fg_matches / fg_count if fg_count > 0 else 0
                    
                    for t_idx in range(len(self.params.iouThrs)):
                        for a_idx in range(len(self.params.areaRng)):
                            for m_idx in range(len(self.params.maxDets)):
                                self.eval["recall"][t_idx, 0, a_idx, m_idx] = corrected_recall
                    
                    logger.info(f"Recalculated precision-recall curve for foreground-only evaluation")
                    logger.info(f"Corrected recall from {old_recall[0, 0, 0, 0]:.3f} to {corrected_recall:.3f} "
                               f"(fg_matches={fg_matches}, fg_GTs={fg_count})")
                else:
                    # Method 2: Estimate from old recall (when extra_calc=False)
                    # Old recall = total_matches / total_GTs
                    # We can estimate: total_matches = old_recall * total_GTs
                    # Then: fg_matches ≈ total_matches * (fg_count / total_count)
                    # And: corrected_recall = fg_matches / fg_count
                    for t_idx in range(len(self.params.iouThrs)):
                        old_r = old_recall[t_idx, 0, 0, 0]
                        if old_r > 0:
                            total_matches = old_r * total_count
                            # Estimate foreground matches (proportional to GT counts)
                            fg_matches_est = total_matches * (fg_count / total_count)
                            corrected_recall = fg_matches_est / fg_count
                            
                            for a_idx in range(len(self.params.areaRng)):
                                for m_idx in range(len(self.params.maxDets)):
                                    self.eval["recall"][t_idx, 0, a_idx, m_idx] = corrected_recall
                    
                    logger.warning(f"Corrected recall using estimation (extra_calc=False). "
                                  f"Set extra_calc=True for accurate results.")
    
    def get_background_stats(self) -> Dict[str, float]:
        """Get statistics specific to background evaluation.
        
        Returns:
            Dict[str, float]: Dictionary containing background-specific metrics
        """
        if not hasattr(self, 'eval') or not self.eval:
            raise Exception("Please run evaluate() and accumulate() first")
            
        stats = {}
        
        # Count foreground vs background annotations
        fg_count = sum(1 for ann in self.cocoGt.anns.values() 
                      if self.class_mapper.is_foreground_class(ann["category_id"]))
        bg_count = sum(1 for ann in self.cocoGt.anns.values() 
                      if self.class_mapper.is_background_class(ann["category_id"]))
        
        stats["total_foreground_gt"] = fg_count
        stats["total_background_gt"] = bg_count

        # count all predictions
        total_pred = sum(1 for ann in self.cocoDt.anns.values())
        stats["total_pred"] = total_pred
        
        # Count TP/FP/FN for foreground and background separately
        fg_tp = sum(1 for ann in self.cocoGt.anns.values() 
                   if ann.get("dt_id") is not None and 
                   self.class_mapper.is_foreground_class(ann["category_id"]))
        bg_tp = sum(1 for ann in self.cocoGt.anns.values() 
                   if ann.get("dt_id") is not None and 
                   self.class_mapper.is_background_class(ann["category_id"]))
        fg_fn = sum(1 for ann in self.cocoGt.anns.values() 
                   if ann.get("fn", False) and 
                   self.class_mapper.is_foreground_class(ann["category_id"]))
        
        stats["foreground_tp"] = fg_tp
        stats["background_tp"] = bg_tp  
        stats["foreground_fn"] = fg_fn
        stats["background_fn"] = 0  # By design, background FN is always 0
        
        # Calculate precision/recall for foreground only (since FN only counts foreground)
        if fg_count > 0:
            stats["foreground_recall"] = fg_tp / fg_count
        else:
            stats["foreground_recall"] = 0.0
            
        return stats
        
    @property
    def extended_metrics(self):
        """Override extended_metrics to only use foreground categories.
        
        Returns:
            dict: Extended metrics calculated only for foreground categories.
        """
        ####################
        # Below part is mostly copied from core/faster_eval_api.py
        # with modifications to filter background categories
        ####################
        
        # Extract IoU and recall thresholds from parameters
        iou_thrs, rec_thrs = self.params.iouThrs, self.params.recThrs

        # Indices for IoU=0.50, first area, and last max dets
        iou50_idx, area_idx, maxdet_idx = (int(np.argwhere(np.isclose(iou_thrs, 0.50))), 0, -1)
        P = self.eval["precision"]
        S = self.eval["scores"]

        # Get precision for IoU=0.50, area, and max dets
        prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]
        prec = prec_raw.copy().astype(float)
        prec[prec < 0] = np.nan

        # Compute F1 score for each class and recall threshold
        f1_cls = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
        f1_macro = np.nanmean(f1_cls, axis=1)
        best_j = int(f1_macro.argmax())

        # Macro precision and recall at the best F1 score
        macro_precision = float(np.nanmean(prec[best_j]))
        macro_recall = float(rec_thrs[best_j])

        # Score vector for the best recall threshold
        score_vec = S[iou50_idx, best_j, :, area_idx, maxdet_idx].astype(float)
        score_vec[prec_raw[best_j] < 0] = np.nan

        per_class = []
        if self.params.useCats:
            ####################
            # MODIFIED: Filter to only use foreground categories
            # Original used: cat_ids = self.params.catIds
            ####################
            cat_ids = [cat_id for cat_id in self.params.catIds 
                      if self.class_mapper.is_foreground_class(cat_id)]
            
            if cat_ids:  # Only compute per-class metrics if we have foreground categories
                cat_id_to_name = {c["id"]: c["name"] for c in self.cocoGt.loadCats(cat_ids)}
                
                ####################
                # Below part is copied from core/faster_eval_api.py
                ####################
                for k, cid in enumerate(cat_ids):
                    # Precision per category
                    p_slice = P[:, :, k, area_idx, maxdet_idx]
                    valid = p_slice > -1
                    ap_50_95 = float(p_slice[valid].mean()) if valid.any() else float("nan")
                    ap_50 = (
                        float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean())
                        if (p_slice[iou50_idx] > -1).any()
                        else float("nan")
                    )

                    pc = float(prec[best_j, k]) if prec_raw[best_j, k] > -1 else float("nan")
                    rc = macro_recall

                    # Filter out dataset class if any metric is NaN
                    if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
                        continue

                    per_class.append({
                        "class": cat_id_to_name[int(cid)],
                        "map@50:95": ap_50_95,
                        "map@50": ap_50,
                        "precision": pc,
                        "recall": rc,
                    })

        # Add metrics for all classes combined
        per_class.append({
            "class": "all",
            "map@50:95": self.stats_as_dict["AP_all"],
            "map@50": self.stats_as_dict["AP_50"],
            "precision": macro_precision,
            "recall": macro_recall,
        })

        return {
            "class_map": per_class,
            "map": self.stats_as_dict["AP_50"],
            "precision": macro_precision,
            "recall": macro_recall,
        }
    
    def summarize(self):
        """Summarize evaluation results including background statistics."""
        super().summarize()
        
        # Add background-specific statistics
        if hasattr(self, 'eval') and self.eval:
            bg_stats = self.get_background_stats()
            self.print_function("\nBackground Instance Statistics:")
            self.print_function(f" Foreground GT: {bg_stats['total_foreground_gt']}")
            self.print_function(f" Background GT: {bg_stats['total_background_gt']}")
            self.print_function(f" Foreground TP: {bg_stats['foreground_tp']}")
            self.print_function(f" Background TP: {bg_stats['background_tp']}")
            self.print_function(f" Foreground FN: {bg_stats['foreground_fn']}")
            self.print_function(f" Background FN: {bg_stats['background_fn']} (by design)")
            self.print_function(f" Foreground Recall: {bg_stats['foreground_recall']:.3f}")
