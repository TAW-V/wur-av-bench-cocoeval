# Keypoint branch evaluation extension for faster-coco-eval
# Evaluates keypoints using:
# - Euclidean distance for 'Edge' keypoint (branch origin/base)
# - Distance to ray for 'Pt' keypoint (branch direction point)
#
# Copyright (c) Thomas Versmissen 2025

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .faster_eval_api import COCOeval_faster

logger = logging.getLogger(__name__)


class COCOeval_branch_keypoints(COCOeval_faster):
    """Extended COCO evaluation for keypoint branch-based matching.
    
    This class evaluates keypoint predictions using a hybrid distance metric:
    - 'Edge' keypoint: Euclidean distance from predicted Edge to ground truth Edge
    - 'Pt' keypoint: Distance from predicted Pt to a ray (half-line) that
                     starts at GT Edge, passes through GT Pt, and extends
                     infinitely in that direction
    
    Both distances must be within the threshold for a detection to be considered correct.
    
    Keypoint format in annotations: [x_pt, y_pt, v_pt, x_edge, y_edge, v_edge]
    where v is visibility (0=not labeled, 1=occluded, 2=visible)
    """

    def __init__(
        self,
        *args,
        distance_threshold: float = 10.0,
        pt_keypoint_idx: int = 1,
        edge_keypoint_idx: int = 0,
        **kwargs
    ):
        """Initialize keypoint branch evaluation.
        
        Args:
            *args: Arguments passed to COCOeval_faster.
            distance_threshold (float): Maximum allowed distance (in pixels) for both
                                       Euclidean and line distance. Default: 10.0
            pt_keypoint_idx (int): Index of the 'Pt' keypoint (0-indexed). Default: 0
            edge_keypoint_idx (int): Index of the 'Edge' keypoint (0-indexed). Default: 1
            **kwargs: Keyword arguments passed to COCOeval_faster.
        """
        # Force iouType to keypoints if not already set
        if 'iouType' not in kwargs and len(args) < 3:
            kwargs['iouType'] = 'keypoints'
        
        super().__init__(*args, **kwargs)
        
        self.distance_threshold = distance_threshold
        self.pt_keypoint_idx = pt_keypoint_idx
        self.edge_keypoint_idx = edge_keypoint_idx
        
        # Override sigmas - we only have 2 keypoints
        self.params.kpt_oks_sigmas = np.array([1.0, 1.0])  # Not used but needed for compatibility
        
        logger.info(f"Keypoint branch evaluation initialized with threshold={distance_threshold}px")
    
    def _mark_invisible_gts_as_ignored(self):
        """Mark GT annotations with invisible keypoints as ignored.
        
        If either the Pt or Edge keypoint has visibility=0, the entire GT
        annotation is marked as ignore=1 so it doesn't count toward evaluation.
        """
        if not self.cocoGt:
            return
        
        ignored_count = 0
        for ann_id, ann in self.cocoGt.anns.items():
            if "keypoints" not in ann:
                continue
                
            kpts = ann["keypoints"]
            
            # Get visibility for Pt and Edge keypoints
            pt_vis = kpts[self.pt_keypoint_idx * 3 + 2] if len(kpts) > self.pt_keypoint_idx * 3 + 2 else 0
            edge_vis = kpts[self.edge_keypoint_idx * 3 + 2] if len(kpts) > self.edge_keypoint_idx * 3 + 2 else 0
            
            # If either keypoint is invisible (v=0), mark as ignored
            if pt_vis == 0 or edge_vis == 0:
                ann["ignore"] = 1
                ignored_count += 1
        
        if ignored_count > 0:
            logger.info(f"Marked {ignored_count} GT annotations as ignored (invisible keypoints)")
    
    def evaluate(self):
        """Run evaluation, marking GTs with invisible keypoints as ignored first."""
        self._mark_invisible_gts_as_ignored()
        super().evaluate()
    
    @staticmethod
    def point_to_ray_distance(
        point: Tuple[float, float],
        ray_start: Tuple[float, float],
        ray_through: Tuple[float, float]
    ) -> float:
        """Calculate distance from a point to a ray (half-line).
        
        The ray starts at ray_start, passes through ray_through, and extends
        infinitely ONLY in that direction (not backwards from ray_start).
        
        Args:
            point: (x, y) coordinates of the point to measure distance from
            ray_start: (x, y) coordinates where the ray starts (e.g., GT Edge)
            ray_through: (x, y) coordinates the ray passes through (e.g., GT Pt)
            
        Returns:
            float: Distance from point to ray. This is:
                   - Perpendicular distance if the point projects onto the ray
                   - Euclidean distance to ray_start if point is "behind" the start
        """
        x0, y0 = point
        x1, y1 = ray_start
        x2, y2 = ray_through
        
        # Direction vector of the ray: from ray_start to ray_through
        dx = x2 - x1
        dy = y2 - y1
        
        # Length squared of direction vector
        len_sq = dx * dx + dy * dy
        
        if len_sq < 1e-10:  # Degenerate case: ray_start == ray_through
            # Fall back to Euclidean distance to ray_start
            return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        
        # Calculate parameter t for the projection of point onto the line
        # t = dot(point - ray_start, direction) / |direction|²
        # t < 0 means point projects behind ray_start
        # t >= 0 means point projects onto or beyond the ray
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / len_sq
        
        if t < 0:
            # Point projects behind the ray start - use distance to ray_start
            return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        else:
            # Point projects onto the ray - use perpendicular distance
            # Projection point on line: (x1 + t*dx, y1 + t*dy)
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            return np.sqrt((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2)
    
    @staticmethod
    def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points.
        
        Args:
            pt1: (x, y) coordinates of first point
            pt2: (x, y) coordinates of second point
            
        Returns:
            float: Euclidean distance
        """
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    def compute_keypoint_match(
        self,
        dt_keypoints: np.ndarray,
        gt_keypoints: np.ndarray,
        gt_visibility: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Compute whether detection keypoints match ground truth.
        
        Args:
            dt_keypoints: Detection keypoints array [x_pt, y_pt, v_pt, x_edge, y_edge, v_edge, ...]
            gt_keypoints: Ground truth keypoints array [x_pt, y_pt, v_pt, x_edge, y_edge, v_edge, ...]
            gt_visibility: Ground truth visibility array [v_pt, v_edge, ...]
            
        Returns:
            Tuple of (is_match, edge_distance, pt_distance)
        """
        # Extract Pt keypoint coordinates (index 0)
        pt_idx = self.pt_keypoint_idx
        dt_pt = (dt_keypoints[pt_idx * 3], dt_keypoints[pt_idx * 3 + 1])
        gt_pt = (gt_keypoints[pt_idx * 3], gt_keypoints[pt_idx * 3 + 1])
        gt_pt_visible = gt_visibility[pt_idx] > 0
        
        # Extract Edge keypoint coordinates (index 1)
        edge_idx = self.edge_keypoint_idx
        dt_edge = (dt_keypoints[edge_idx * 3], dt_keypoints[edge_idx * 3 + 1])
        gt_edge = (gt_keypoints[edge_idx * 3], gt_keypoints[edge_idx * 3 + 1])
        gt_edge_visible = gt_visibility[edge_idx] > 0
        
        # Note: GTs with invisible keypoints should be marked as ignored and
        # excluded from evaluation. This method assumes both keypoints are visible.
        # If either is invisible, we fail the match as a safety measure.
        
        if not gt_edge_visible or not gt_pt_visible:
            # GT has invisible keypoints - it's marked as ignore=1.
            # Return a HIGH score so detections CAN match this GT.
            # The ignore flag ensures the match doesn't count as TP or FP.
            return True, 0.0, 0.0
        
        # Calculate Euclidean distance for Edge keypoint (anchor point)
        edge_distance = self.euclidean_distance(dt_edge, gt_edge)
        
        # Calculate ray distance for Pt keypoint
        # Ray starts at GT Edge, passes through GT Pt, and extends infinitely in that direction
        pt_distance = self.point_to_ray_distance(dt_pt, gt_edge, gt_pt)
        
        # Both must be within threshold
        edge_ok = edge_distance <= self.distance_threshold
        pt_ok = pt_distance <= self.distance_threshold
        
        is_match = pt_ok and edge_ok
        
        return is_match, edge_distance, pt_distance
    
    def computeOks(self, imgId: int, catId: int) -> np.ndarray:
        """Compute matching score between ground truth and detection keypoints.
        
        Instead of OKS, this computes a binary match (1.0 or 0.0) based on whether
        both distance criteria are satisfied.
        
        Args:
            imgId (int): Image ID.
            catId (int): Category ID.
            
        Returns:
            np.ndarray: Match scores (1.0 for match, 0.0 for no match) between 
                       gt and dt for the given image and category.
        """
        p = self.params
        
        # Get ground truth and detections
        gts = self.gt_dataset.get(imgId, catId)
        dts = self.dt_dataset.get(imgId, catId)
        
        # Sort detections by score
        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0 : p.maxDets[-1]]
        
        if len(gts) == 0 or len(dts) == 0:
            return []
        
        # Initialize match matrix
        matches = np.zeros((len(dts), len(gts)))
        
        for j, gt in enumerate(gts):
            g = np.array(gt["keypoints"])
            vg = g[2::3]  # Visibility flags
            
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                
                is_match, edge_dist, pt_dist = self.compute_keypoint_match(d, g, vg)
                
                # Store as 1.0 (match) or 0.0 (no match)
                # We use a continuous value based on how well it matches for ranking
                if is_match:
                    # Use inverse of max distance as score for ranking among matches
                    max_dist = max(edge_dist, pt_dist)
                    if max_dist > 0:
                        #The 0.5 factor ensures all matches stay above 0.5, so they pass the standard COCO IoU thresholds (0.50, 0.55, ..., 0.95).
                        matches[i, j] = 1.0 - (max_dist / self.distance_threshold) * 0.5 
                    elif max_dist == 0: 
                        # perfect match
                        matches[i, j] = 1.0
                    else:
                        # should never happen
                        raise ValueError("Distance is negative")
                else:
                    matches[i, j] = 0.0
        
        return matches
    
    def get_detailed_results(self) -> Dict:
        """Get detailed per-detection results with distances.
        
        Returns:
            Dict containing per-detection distance information.
        """
        if not hasattr(self, 'eval') or not self.eval:
            raise Exception("Please run evaluate() and accumulate() first")
        
        results = {
            'threshold': self.distance_threshold,
            'detections': []
        }
        
        # Iterate through matches
        if 'matched' in self.eval:
            for dt_gt, iou in self.eval['matched'].items():
                dt_id, gt_id = dt_gt.split('_')
                dt_id = int(dt_id)
                gt_id = int(gt_id)
                
                if dt_id in self.cocoDt.anns and gt_id in self.cocoGt.anns:
                    dt_ann = self.cocoDt.anns[dt_id]
                    gt_ann = self.cocoGt.anns[gt_id]
                    
                    d = np.array(dt_ann["keypoints"])
                    g = np.array(gt_ann["keypoints"])
                    vg = g[2::3]
                    
                    is_match, edge_dist, pt_dist = self.compute_keypoint_match(d, g, vg)
                    
                    results['detections'].append({
                        'dt_id': dt_id,
                        'gt_id': gt_id,
                        'image_id': dt_ann.get('image_id'),
                        'edge_distance': edge_dist,
                        'pt_distance': pt_dist,
                        'is_match': is_match,
                        'score': dt_ann['score']
                    })
        
        return results
    
    def summarize(self):
        """Summarize evaluation results including distance statistics."""
        super().summarize()
        
        self.print_function("\nKeypoint branch Evaluation Statistics:")
        self.print_function(f" Distance Threshold: {self.distance_threshold} pixels")
        self.print_function(f" Pt Keypoint Index: {self.pt_keypoint_idx}")
        self.print_function(f" Edge Keypoint Index: {self.edge_keypoint_idx}")

