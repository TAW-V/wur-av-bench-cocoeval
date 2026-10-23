# Branch keypoints evaluation with background instance support
# Combines keypoint distance-based matching with background-aware evaluation
#
# Copyright (c) Thomas Versmissen 2025

import logging
from typing import Dict

from .branch_keypoints_eval import COCOeval_branch_keypoints
from .background_eval import COCOeval_background, BackgroundClassMapper
from .faster_eval_api import COCOeval_faster

logger = logging.getLogger(__name__)


class COCOeval_branch_keypoints_background(COCOeval_branch_keypoints):
    """Branch keypoints evaluation with background instance support.
    
    This class combines:
    - Distance-based keypoint matching (Edge euclidean + Pt ray distance)
    - Background-aware evaluation (background GTs don't count as FN)
    
    GTs are ignored (don't count toward recall) if:
    - Either keypoint has visibility=0, OR
    - The GT is a background class (class ID ends in 1)
    
    Predictions can still match ignored GTs (absorbing detections without
    counting as FP), but only foreground GTs with visible keypoints
    contribute to the recall denominator.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize branch keypoints background evaluation."""
        super().__init__(*args, **kwargs)
        self.class_mapper = BackgroundClassMapper()
    
    # Reuse methods from COCOeval_background
    _prepare_background_categories = COCOeval_background._prepare_background_categories
    
    def _mark_background_gts_as_ignored(self):
        """Mark background GT annotations as ignored.
        
        Wrapper that calls COCOeval_background's method and stores changes
        for later restoration.
        """
        if not self.cocoGt:
            return
        
        self.background_cat_changes = []
        ignored_count = 0
        
        for ann_id, ann in self.cocoGt.anns.items():
            if self.class_mapper.is_background_class(ann["category_id"]):
                original_cat_id = ann["category_id"]
                foreground_cat_id = self.class_mapper.get_foreground_id(original_cat_id)
                
                self.background_cat_changes.append((ann_id, original_cat_id))
                ann["_original_cat_id"] = original_cat_id
                ann["category_id"] = foreground_cat_id
                ann["_bg_original_ignore"] = ann.get("ignore", 0)
                ann["ignore"] = 1
                ann["_is_background_gt"] = True
                ignored_count += 1
        
        if ignored_count > 0:
            logger.info(f"Marked {ignored_count} background GT annotations as ignored")
    
    def _restore_background_categories(self):
        """Restore original category IDs after evaluation."""
        if not self.cocoGt or not hasattr(self, 'background_cat_changes'):
            return
            
        for ann_id, original_cat_id in self.background_cat_changes:
            ann = self.cocoGt.anns[ann_id]
            ann["category_id"] = original_cat_id
            
            if "_bg_original_ignore" in ann:
                ann["ignore"] = ann["_bg_original_ignore"]
                del ann["_bg_original_ignore"]
    
    def evaluate(self):
        """Run evaluation with invisible keypoint and background handling."""
        # 1. Mark invisible keypoints as ignored
        self._mark_invisible_gts_as_ignored()
        
        # 2. Prepare and mark background GTs as ignored
        self._prepare_background_categories()
        self._mark_background_gts_as_ignored()
        
        # 3. Run grandparent evaluate (COCOeval_faster)
        COCOeval_faster.evaluate(self)
        
        # 4. Restore original category IDs
        self._restore_background_categories()
    
    # Reuse get_background_stats from COCOeval_background with minor adaptation
    def get_background_stats(self) -> Dict[str, float]:
        """Get statistics specific to background evaluation."""
        if not hasattr(self, 'eval') or not self.eval:
            raise Exception("Please run evaluate() and accumulate() first")
            
        stats = {}
        
        fg_count = 0
        bg_count = 0
        fg_visible_count = 0
        
        for ann in self.cocoGt.anns.values():
            cat_id = ann.get("_original_cat_id", ann["category_id"])
            is_bg = self.class_mapper.is_background_class(cat_id)
            
            if is_bg:
                bg_count += 1
            else:
                fg_count += 1
                kpts = ann.get("keypoints", [])
                if len(kpts) >= 6:
                    pt_vis = kpts[self.pt_keypoint_idx * 3 + 2]
                    edge_vis = kpts[self.edge_keypoint_idx * 3 + 2]
                    if pt_vis > 0 and edge_vis > 0:
                        fg_visible_count += 1
        
        stats["total_foreground_gt"] = fg_count
        stats["total_background_gt"] = bg_count
        stats["foreground_visible_gt"] = fg_visible_count
        
        fg_tp = 0
        bg_tp = 0
        
        for ann in self.cocoGt.anns.values():
            if ann.get("dt_id") is not None:
                cat_id = ann.get("_original_cat_id", ann["category_id"])
                if self.class_mapper.is_background_class(cat_id):
                    bg_tp += 1
                else:
                    fg_tp += 1
        
        stats["foreground_tp"] = fg_tp
        stats["background_tp"] = bg_tp
        stats["foreground_fn"] = fg_visible_count - fg_tp
        stats["background_fn"] = 0
        
        if fg_visible_count > 0:
            stats["foreground_recall"] = fg_tp / fg_visible_count
        else:
            stats["foreground_recall"] = 0.0
            
        return stats
    
    def summarize(self):
        """Summarize evaluation results including keypoint and background statistics."""
        COCOeval_faster.summarize(self)
        
        self.print_function("\nKeypoint Branch Evaluation Statistics:")
        self.print_function(f" Distance Threshold: {self.distance_threshold} pixels")
        self.print_function(f" Pt Keypoint Index: {self.pt_keypoint_idx}")
        self.print_function(f" Edge Keypoint Index: {self.edge_keypoint_idx}")
        
        if hasattr(self, 'eval') and self.eval:
            try:
                bg_stats = self.get_background_stats()
                self.print_function("\nBackground Instance Statistics:")
                self.print_function(f" Foreground GT (total): {bg_stats['total_foreground_gt']}")
                self.print_function(f" Foreground GT (visible kpts): {bg_stats['foreground_visible_gt']}")
                self.print_function(f" Background GT: {bg_stats['total_background_gt']}")
                self.print_function(f" Foreground TP: {bg_stats['foreground_tp']}")
                self.print_function(f" Background TP: {bg_stats['background_tp']}")
                self.print_function(f" Foreground FN: {bg_stats['foreground_fn']}")
                self.print_function(f" Background FN: {bg_stats['background_fn']} (by design)")
                self.print_function(f" Foreground Recall: {bg_stats['foreground_recall']:.3f}")
            except Exception as e:
                logger.warning(f"Could not get background statistics: {e}")
