#!/usr/bin/env python3
"""Tests for branch keypoints evaluation."""

import json
import os
import tempfile
import unittest

import numpy as np

from faster_coco_eval import COCO, COCOeval_branch_keypoints


class TestBranchKeypointsEval(unittest.TestCase):
    """Test branch keypoints evaluation."""

    def setUp(self):
        """Create test dataset with Pt and Edge keypoints."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Ground truth dataset
        self.gt_data = {
            "info": {"description": "Test dataset"},
            "licenses": [],
            "images": [
                {"id": 1, "file_name": "test1.jpg", "height": 480, "width": 640},
                {"id": 2, "file_name": "test2.jpg", "height": 480, "width": 640},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    # Pt at (100,100), Edge at (200,100)
                    # Ray: starts at Edge(200,100), goes through Pt(100,100), extends left
                    "keypoints": [ 200, 100, 2, 100, 100, 2],
                    "num_keypoints": 2,
                    "bbox": [90, 90, 120, 20],
                    "area": 2400,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    # Pt at (100,100), Edge at (100,200)
                    # Ray: starts at Edge(100,200), goes through Pt(100,100), extends up
                    "keypoints": [100, 200, 2, 100, 100, 2],
                    "num_keypoints": 2,
                    "bbox": [90, 90, 20, 120],
                    "area": 2400,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "object",
                    "keypoints": ["Pt", "Edge"],
                    "skeleton": [[0, 1]],
                }
            ],
        }
        
        self.gt_file = os.path.join(self.temp_dir, "gt.json")
        with open(self.gt_file, "w") as f:
            json.dump(self.gt_data, f)

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_perfect_match(self):
        """Test perfect keypoint match."""
        # Detection exactly matches ground truth
        dt_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "keypoints": [200, 100, 2, 100, 100, 2],  # Exact match
                "score": 0.9,
            },
        ]
        
        dt_file = os.path.join(self.temp_dir, "dt.json")
        with open(dt_file, "w") as f:
            json.dump(dt_data, f)
        
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(dt_file)
        
        cocoEval = COCOeval_branch_keypoints(
            cocoGt, cocoDt, iouType="keypoints", 
            distance_threshold=10.0,
            print_function=print
        )
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Should have perfect recall for image 1
        self.assertGreater(cocoEval.stats[0], 0)  # AP should be positive

    def test_edge_within_threshold(self):
        """Test Edge keypoint (Euclidean distance) within threshold."""
        # GT Edge at (200,100), predicted Edge at (205, 100) = 5px away
        dt_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "keypoints": [205, 100, 2, 100, 100, 2],  # Edge shifted 5px right
                "score": 0.9,
            },
        ]
        
        dt_file = os.path.join(self.temp_dir, "dt.json")
        with open(dt_file, "w") as f:
            json.dump(dt_data, f)
        
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(dt_file)
        
        cocoEval = COCOeval_branch_keypoints(
            cocoGt, cocoDt, iouType="keypoints",
            distance_threshold=10.0,
            print_function=print
        )
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Should match (5 < 10)
        self.assertGreater(cocoEval.stats[0], 0)

    def test_edge_outside_threshold(self):
        """Test Edge keypoint (Euclidean distance) outside threshold."""
        # GT Edge at (200,100), predicted Edge at (215, 100) = 15px away
        dt_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "keypoints": [215, 100, 2, 100, 100, 2],  # Edge shifted 15px right
                "score": 0.9,
            },
        ]
        
        dt_file = os.path.join(self.temp_dir, "dt.json")
        with open(dt_file, "w") as f:
            json.dump(dt_data, f)
        
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(dt_file)
        
        cocoEval = COCOeval_branch_keypoints(
            cocoGt, cocoDt, iouType="keypoints",
            distance_threshold=10.0,
            print_function=print
        )
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Should NOT match (15 > 10)
        self.assertEqual(cocoEval.stats[0], 0)

    def test_pt_on_ray(self):
        """Test Pt keypoint on the ray (distance = 0)."""
        # GT: Pt at (100,100), Edge at (200,100)
        # Ray starts at GT Edge (200,100), goes through GT Pt (100,100), extends left
        # Predicted Pt at (50,100) is on this ray (extended past GT Pt)
        dt_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "keypoints": [200, 100, 2, 50, 100, 2],  # Pt at (50,100) is on the ray
                "score": 0.9,
            },
        ]
        
        dt_file = os.path.join(self.temp_dir, "dt.json")
        with open(dt_file, "w") as f:
            json.dump(dt_data, f)
        
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(dt_file)
        
        cocoEval = COCOeval_branch_keypoints(
            cocoGt, cocoDt, iouType="keypoints",
            distance_threshold=10.0,
            print_function=print
        )
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Should match (Pt is on ray, distance = 0)
        self.assertGreater(cocoEval.stats[0], 0)

    def test_pt_off_ray_within_threshold(self):
        """Test Pt keypoint off ray but within threshold."""
        # Ray: GT Edge (200,100) -> GT Pt (100,100) -> extends left (horizontal at y=100)
        # Predicted Pt at (100,95) is 5 pixels above the ray
        dt_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "keypoints": [200, 100, 2, 100, 95, 2],  # Pt at (100,95), 5px above ray
                "score": 0.9,
            },
        ]
        
        dt_file = os.path.join(self.temp_dir, "dt.json")
        with open(dt_file, "w") as f:
            json.dump(dt_data, f)
        
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(dt_file)
        
        cocoEval = COCOeval_branch_keypoints(
            cocoGt, cocoDt, iouType="keypoints",
            distance_threshold=10.0,
            print_function=print
        )
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Should match (5 < 10)
        self.assertGreater(cocoEval.stats[0], 0)

    def test_pt_off_ray_outside_threshold(self):
        """Test Pt keypoint off ray and outside threshold."""
        # Ray: GT Edge (200,100) -> GT Pt (100,100) -> extends left (horizontal at y=100)
        # Predicted Pt at (100,85) is 15 pixels above the ray
        dt_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "keypoints": [200, 100, 2, 100, 85, 2],  # Pt at (100,85), 15px above ray
                "score": 0.9,
            },
        ]
        
        dt_file = os.path.join(self.temp_dir, "dt.json")
        with open(dt_file, "w") as f:
            json.dump(dt_data, f)
        
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(dt_file)
        
        cocoEval = COCOeval_branch_keypoints(
            cocoGt, cocoDt, iouType="keypoints",
            distance_threshold=10.0,
            print_function=print
        )
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Should NOT match (15 > 10)
        self.assertEqual(cocoEval.stats[0], 0)

    def test_vertical_ray(self):
        """Test with vertical ray (image 2)."""
        # GT: "keypoints": [100, 200, 2, 100, 100, 2],
        # Predicted Pt at (95,0) should be 5 pixels from the vertical ray
        dt_data = [
            {
                "image_id": 2,
                "category_id": 1,
                "keypoints": [100, 200, 2, 95, 0, 2],
                "score": 0.9,
            },
        ]
        
        dt_file = os.path.join(self.temp_dir, "dt.json")
        with open(dt_file, "w") as f:
            json.dump(dt_data, f)
        
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(dt_file)
        
        cocoEval = COCOeval_branch_keypoints(
            cocoGt, cocoDt, iouType="keypoints",
            distance_threshold=10.0,
            print_function=print
        )
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Should match (5 < 10)
        self.assertGreater(cocoEval.stats[0], 0)

    def test_invisible_keypoint_ignored(self):
        """Test that GT with invisible keypoint is ignored (not counted)."""
        # Create a GT with invisible Pt keypoint (v=0)
        gt_data_invisible = {
            "info": {"description": "Test dataset"},
            "licenses": [],
            "images": [
                {"id": 1, "file_name": "test1.jpg", "height": 480, "width": 640},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    # Pt visibility = 0 (invisible), Edge visibility = 2 (visible)
                    "keypoints": [200, 100, 2, 100, 100, 0],
                    "num_keypoints": 1,
                    "bbox": [90, 90, 120, 20],
                    "area": 2400,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "object",
                    "keypoints": ["Pt", "Edge"],
                    "skeleton": [[0, 1]],
                }
            ],
        }
        
        gt_file = os.path.join(self.temp_dir, "gt_invisible.json")
        with open(gt_file, "w") as f:
            json.dump(gt_data_invisible, f)
        
        # Detection that would match if GT was visible
        dt_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "keypoints": [200, 100, 2, 100, 100, 2],
                "score": 0.9,
            },
        ]
        
        dt_file = os.path.join(self.temp_dir, "dt_invisible.json")
        with open(dt_file, "w") as f:
            json.dump(dt_data, f)
        
        cocoGt = COCO(gt_file)
        cocoDt = cocoGt.loadRes(dt_file)
        
        cocoEval = COCOeval_branch_keypoints(
            cocoGt, cocoDt, iouType="keypoints",
            distance_threshold=10.0,
            print_function=print
        )
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # GT with invisible keypoint should be ignored
        # With no valid GTs, recall is undefined (-1)
        # AP should be -1 or 0 (no valid GTs to evaluate against)
        self.assertEqual(cocoEval.stats[0], -1)

    def test_invisible_keypoint_mixed(self):
        """Test with 2 GTs: one visible (evaluated), one invisible (ignored).
        
        Setup:
        - GT1: Both keypoints visible at (100,100) and (200,100)
        - GT2: Pt keypoint invisible (should be ignored)
        
        Detections:
        - D1 (score=0.80): Matches GT1, thus should be TP
        - D2 (score=0.95): Would match GT2, thus should NOT be TP (GT2 ignored)
        - D3 (score=0.70): Doesn't match anything, thus FP
        
        Expected:
        - 1 TP (D1), 2 FP (D2 doesn't get TP credit, D3 is FP)
        - 0 FN (GT2 is ignored, GT1 is matched)
        - Recall = 1/1 = 1.0, AP > 0
        """
        gt_data_mixed = {
            "info": {"description": "Test dataset"},
            "licenses": [],
            "images": [
                {"id": 1, "file_name": "test1.jpg", "height": 480, "width": 640},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    # GT1: Both keypoints VISIBLE
                    "keypoints": [200, 100, 2, 100, 100, 2],
                    "num_keypoints": 2,
                    "bbox": [90, 90, 120, 20],
                    "area": 2400,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    # GT2: Pt keypoint INVISIBLE (v=0), thus should be ignored
                    "keypoints": [400, 100, 2, 300, 100, 0],
                    "num_keypoints": 1,
                    "bbox": [290, 90, 120, 20],
                    "area": 2400,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "object",
                    "keypoints": ["Pt", "Edge"],
                    "skeleton": [[0, 1]],
                }
            ],
        }
        
        gt_file = os.path.join(self.temp_dir, "gt_mixed.json")
        with open(gt_file, "w") as f:
            json.dump(gt_data_mixed, f)
        
        dt_data = [
            {
                # D1: Matches GT1 (visible), thus TP
                "image_id": 1,
                "category_id": 1,
                "keypoints": [200, 100, 2, 100, 100, 2],  # Exact match to GT1
                "score": 0.80,
            },
            {
                # D2: matches GT2 (invisible/ignored), thus should be ignored
                "image_id": 1,
                "category_id": 1,
                "keypoints": [400, 100, 2, 300, 100, 2],  # Would match GT2
                "score": 0.95,
            },
            {
                # D3: Doesn't match anything, thus FP
                "image_id": 1,
                "category_id": 1,
                "keypoints": [600, 100, 2, 500, 100, 2],  # No matching GT
                "score": 0.85,
            },
        ]
        
        dt_file = os.path.join(self.temp_dir, "dt_mixed.json")
        with open(dt_file, "w") as f:
            json.dump(dt_data, f)
        
        cocoGt = COCO(gt_file)
        cocoDt = cocoGt.loadRes(dt_file)
        
        cocoEval = COCOeval_branch_keypoints(
            cocoGt, cocoDt, iouType="keypoints",
            distance_threshold=10.0,
            print_function=print
        )
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        

        # AP should be 0.5 because we have 1 TP and 1 FP (D2 is ignored)
        self.assertAlmostEqual(cocoEval.stats[0], 0.5, places=2)
        
        # Check recall at first threshold - should be 1.0 (1 matched out of 1 valid GT)
        # recall shape: [num_iou_thrs, num_cats, num_area_ranges, num_max_dets]
        recall_at_first_threshold = cocoEval.eval["recall"][0, 0, 0, 0]
        self.assertAlmostEqual(recall_at_first_threshold, 1.0, places=2)

    def test_point_to_ray_distance_horizontal(self):
        """Test point to ray distance calculation for horizontal ray."""
        # Horizontal ray from (0, 0) through (10, 0), extending right
        # Point at (5, 3) should be 3 units away (perpendicular)
        distance = COCOeval_branch_keypoints.point_to_ray_distance(
            (5, 3), (0, 0), (10, 0)
        )
        self.assertAlmostEqual(distance, 3.0, places=6)

    def test_point_to_ray_distance_vertical(self):
        """Test point to ray distance calculation for vertical ray."""
        # Vertical ray from (0, 0) through (0, 10), extending up
        # Point at (4, 5) should be 4 units away (perpendicular)
        distance = COCOeval_branch_keypoints.point_to_ray_distance(
            (4, 5), (0, 0), (0, 10)
        )
        self.assertAlmostEqual(distance, 4.0, places=6)

    def test_point_to_ray_distance_diagonal(self):
        """Test point to ray distance calculation for diagonal ray."""
        # Diagonal ray from (0, 0) through (1, 1) (45 degrees)
        # Point at (0, 1) should be sqrt(2)/2 ≈ 0.707 units away
        distance = COCOeval_branch_keypoints.point_to_ray_distance(
            (0, 1), (0, 0), (1, 1)
        )
        expected = np.sqrt(2) / 2
        self.assertAlmostEqual(distance, expected, places=6)

    def test_point_to_ray_distance_behind_start(self):
        """Test point to ray distance when point is behind ray start."""
        # Ray from (10, 0) through (20, 0), extending right
        # Point at (5, 0) is BEHIND the ray start, so distance should be
        # Euclidean distance to ray_start = 5
        distance = COCOeval_branch_keypoints.point_to_ray_distance(
            (5, 0), (10, 0), (20, 0)
        )
        self.assertAlmostEqual(distance, 5.0, places=6)

    def test_point_to_ray_distance_behind_start_with_offset(self):
        """Test point to ray distance when point is behind and offset from ray."""
        # Ray from (10, 0) through (20, 0), extending right
        # Point at (5, 3) is behind the ray start
        # Distance should be Euclidean to ray_start: sqrt((10-5)^2 + (0-3)^2) = sqrt(34)
        distance = COCOeval_branch_keypoints.point_to_ray_distance(
            (5, 3), (10, 0), (20, 0)
        )
        expected = np.sqrt(34)
        self.assertAlmostEqual(distance, expected, places=6)


if __name__ == "__main__":
    unittest.main()

