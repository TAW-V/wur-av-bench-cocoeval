# Background Instance Evaluation Extension

This extension adds support for evaluating **background instances** in object detection, allowing for more nuanced evaluation in scenes where true objects in the background could be detected but are irrelevant.

This page describes the method and the data format in more detail. Installation and basic usage are in the main [README](README.md).

## Overview

Traditional COCO evaluation treats all ground truth annotations equally, missing any annotation counts as a false negative. The background instance evaluation introduces a two-tier system:

- **Foreground instances**: Primary objects that should be detected (missing = false negative)
- **Background instances**: Secondary objects that may be detected: a detection matching one is excluded from the evaluation, so it does not count as a false positive, and missing one is never a false negative

## Key Features

- **Cross-class matching**: Predictions can match both foreground and background annotations of the same object type  
- **Selective FN counting**: Only foreground objects count toward false negatives  
- **Backward compatible**: Works with existing COCO datasets  
- **5-digit class IDs**: Clear foreground/background distinction  

## Class ID Format

The system uses 5-digit class IDs where the last digit indicates the instance type:

- **Foreground classes**: End in `0` (e.g., `12340` for person)
- **Background classes**: End in `1` (e.g., `12341` for person_background)
- **Base class**: First 4 digits (e.g., `1234` is the base for both `12340` and `12341`)

```python
# Example class mapping
12340  # person (foreground)
12341  # person_background (background)
20000  # car (foreground)
20001  # car_background (background)
```

## Data Format

### Ground Truth Annotations

```json
{
  "categories": [
    {"id": 12340, "name": "person", "supercategory": "root"},
    {"id": 12341, "name": "person_background", "supercategory": "root"},
    {"id": 20000, "name": "car", "supercategory": "root"},
    {"id": 20001, "name": "car_background", "supercategory": "root"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 12340,
      "bbox": [100, 100, 50, 100],
      "area": 5000,
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 12341,
      "bbox": [300, 150, 30, 60],
      "area": 1800,
      "iscrowd": 0
    }
  ]
}
```

Annotation 1 is a foreground person (`12340`), annotation 2 a background person (`12341`).

### Predictions

Predictions use standard COCO format and always use the foreground class ID, here a person prediction (`12340`):

```json
[
  {
    "image_id": 1,
    "category_id": 12340,
    "bbox": [105, 105, 45, 95],
    "score": 0.9
  }
]
```

## Evaluation Behavior

### Matching Logic

A prediction can match with ground truth annotations of the **same base class**:

- Person prediction (`12340`) can match:
  - Person foreground GT (`12340`) 
  - Person background GT (`12341`) 
- Car prediction (`20000`) can match:
  - Car foreground GT (`20000`) 
  - Car background GT (`20001`) 

### False Negative Counting

**Standard COCO**: All unmatched GT → FN  
**Background-aware**: Only unmatched foreground GT → FN

```python
# Example scenario
ground_truth = [
    {"category_id": 12340},  # person foreground - not matched
    {"category_id": 12341},  # person background - not matched  
    {"category_id": 20000},  # car foreground - matched
]

# Standard COCO: 2 FN (person FG + person BG)
# Background-aware: 1 FN (only person FG)
```

