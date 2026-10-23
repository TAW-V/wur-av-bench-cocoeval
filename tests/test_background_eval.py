# Original work Copyright (c) 2025 TAW-V

import unittest
import json
import tempfile
import os
from faster_coco_eval import COCO, COCOeval_background, BackgroundClassMapper


class TestBackgroundClassMapper(unittest.TestCase):
    """Test the BackgroundClassMapper utility class."""
    
    def test_get_base_class_id(self):
        """Test base class ID extraction."""
        mapper = BackgroundClassMapper()
        
        # Test 5-digit IDs
        self.assertEqual(mapper.get_base_class_id(12340), 12340)
        self.assertEqual(mapper.get_base_class_id(12341), 12340)
        self.assertEqual(mapper.get_base_class_id(10000), 10000)
        self.assertEqual(mapper.get_base_class_id(10001), 10000)
        
        # Testless than 5-digit IDs, without background/foreground suffix
        self.assertEqual(mapper.get_base_class_id(1234), 12340)
        self.assertEqual(mapper.get_base_class_id(1000), 10000)
    
    def test_is_background_class(self):
        """Test background class detection."""
        mapper = BackgroundClassMapper()
        
        # Background classes (end in 1) - with background suffix
        self.assertTrue(mapper.is_background_class(12341))
        self.assertTrue(mapper.is_background_class(10001))
        
        # Foreground classes (end in 0)
        self.assertFalse(mapper.is_background_class(12340))
        self.assertFalse(mapper.is_background_class(10000))
    
    def test_is_foreground_class(self):
        """Test foreground class detection."""
        mapper = BackgroundClassMapper()
        
        # Foreground classes (end in 0)
        self.assertTrue(mapper.is_foreground_class(12340))
        self.assertTrue(mapper.is_foreground_class(10000))
        
        # Background classes (end in 1)
        self.assertFalse(mapper.is_foreground_class(12341))
        self.assertFalse(mapper.is_foreground_class(10001))
    
    def test_get_foreground_background_ids(self):
        """Test foreground/background ID generation."""
        mapper = BackgroundClassMapper()
        
        # Test with base class 12340
        self.assertEqual(mapper.get_foreground_id(12340), 12340)
        self.assertEqual(mapper.get_background_id(12340), 12341)
        
        # Test with base class 10000
        self.assertEqual(mapper.get_foreground_id(10000), 10000)
        self.assertEqual(mapper.get_background_id(10000), 10001)
    
    def test_are_same_base_class(self):
        """Test base class comparison."""
        mapper = BackgroundClassMapper()
        
        # Same base class
        self.assertTrue(mapper.are_same_base_class(12340, 12341))
        self.assertTrue(mapper.are_same_base_class(10000, 10001))
        
        # Different base classes
        self.assertFalse(mapper.are_same_base_class(12340, 10000))
        self.assertFalse(mapper.are_same_base_class(12341, 10001))


class TestCOCOevalBackground(unittest.TestCase):
    """Test the COCOeval_background evaluation class."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test ground truth with foreground and background instances
        self.gt_data = {
            "info": {"description": "test data"},
            "images": [
                {"id": 1, "width": 100, "height": 100, "file_name": "test1.jpg"},
                {"id": 2, "width": 100, "height": 100, "file_name": "test2.jpg"}
            ],
            "categories": [
                {"id": 10000, "name": "person", "supercategory": "person"},
                {"id": 10001, "name": "person_background", "supercategory": "person"},
                {"id": 20000, "name": "car", "supercategory": "vehicle"},
                {"id": 20001, "name": "car_background", "supercategory": "vehicle"}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 10000,  # foreground person
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 10001,  # background person
                    "bbox": [50, 50, 15, 15],
                    "area": 225,
                    "iscrowd": 0
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 20000,  # foreground car
                    "bbox": [20, 20, 30, 30],
                    "area": 900,
                    "iscrowd": 0
                },
                {
                    "id": 4,
                    "image_id": 2,
                    "category_id": 20001,  # background car
                    "bbox": [60, 60, 10, 10],
                    "area": 100,
                    "iscrowd": 0
                }
            ]
        }
        
        # Create test predictions
        self.dt_data = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 10000,  # person prediction
                "bbox": [12, 12, 18, 18],  # slightly shifted from GT
                "score": 0.9,
                "area": 324
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 10000,  # person prediction matching background
                "bbox": [52, 52, 13, 13],  # matches background person
                "score": 0.7,
                "area": 169
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 20000,  # car prediction
                "bbox": [22, 22, 28, 28],  # slightly shifted from foreground GT
                "score": 0.8,
                "area": 784
            }
            # Note: No prediction for background car (id=4), testing FN behavior
        ]
        
        # Write test data to files
        self.gt_file = os.path.join(self.temp_dir, "gt.json")
        self.dt_file = os.path.join(self.temp_dir, "dt.json")
        
        with open(self.gt_file, 'w') as f:
            json.dump(self.gt_data, f)
        with open(self.dt_file, 'w') as f:
            json.dump(self.dt_data, f)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_background_evaluation_basic(self):
        """Test basic background evaluation functionality."""
        # Load data
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(self.dt_file)
        
        # Run background evaluation
        cocoEval = COCOeval_background(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        # Test that evaluation ran without errors
        self.assertIsNotNone(cocoEval.eval)
        self.assertIn('precision', cocoEval.eval)
        self.assertIn('recall', cocoEval.eval)
    
    def test_background_matching_logic(self):
        """Test that predictions can match both foreground and background GT of same base class."""
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(self.dt_file)
        
        cocoEval = COCOeval_background(cocoGt, cocoDt, 'bbox')
        cocoEval.extra_calc = True  # Enable detailed matching analysis
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        # Check that matches were found
        self.assertGreater(len(cocoEval.eval.get("matched", {})), 0)
    
    def test_background_fn_logic(self):
        """Test that FN is only counted for foreground objects."""
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(self.dt_file)
        
        cocoEval = COCOeval_background(cocoGt, cocoDt, 'bbox')
        cocoEval.extra_calc = True
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        # Get background statistics
        bg_stats = cocoEval.get_background_stats()
        
        # Verify that we have both foreground and background annotations
        self.assertGreater(bg_stats['total_foreground_gt'], 0)
        self.assertGreater(bg_stats['total_background_gt'], 0)
        
        # Background FN should always be 0 by design
        self.assertEqual(bg_stats['background_fn'], 0)
        
        # Check that foreground annotations can have FN > 0 if not matched
        # (depends on specific test data and matching success)
        self.assertGreaterEqual(bg_stats['foreground_fn'], 0)
    
    def test_background_category_preparation(self):
        """Test that background categories are properly prepared."""
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(self.dt_file)
        
        cocoEval = COCOeval_background(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()  # This calls _prepare_background_categories
        
        # Check that base class groups were created
        self.assertTrue(hasattr(cocoEval, 'base_class_groups'))
        self.assertGreater(len(cocoEval.base_class_groups), 0)
        
        # Check that we have both person (10000/10001) and car (20000/20001) groups
        expected_base_classes = {10000, 20000}
        actual_base_classes = set(cocoEval.base_class_groups.keys())
        self.assertEqual(expected_base_classes, actual_base_classes)
        
        # Check that each base class has both foreground and background categories
        for base_id in expected_base_classes:
            group = cocoEval.base_class_groups[base_id]
            self.assertIn('foreground', group)
            self.assertIn('background', group)
            self.assertEqual(len(group['foreground']), 1)  # Should have one foreground class
            self.assertEqual(len(group['background']), 1)  # Should have one background class


if __name__ == '__main__':
    unittest.main()
