# Background-aware COCO evaluation

This repository (wur-av-bench-cocoeval) is a fork of [faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval) with two additions for evaluating object detection in cluttered scenes:

- **Background-aware evaluation**: ground truth objects can be labelled as background instances. Detecting them is fine, missing them is not penalized.
- **Branch keypoint evaluation**: pixel distance based matching for two-point branch annotations, as an alternative to OKS.

Both methods were developed for evaluating detection of plant parts (peduncles and petioles) in dense canopies, but they are not specific to that use case. This code implements the Task I and Task II evaluation of the WUR-AV-Bench benchmark and accompanies our paper in Computers and Electronics in Agriculture. The tools for Tasks III and IV live in a separate repository, see [Related repositories](#related-repositories).

## Background-aware evaluation

Standard COCO evaluation treats every ground truth object the same: each missed annotation is a false negative. In cluttered scenes this is often too strict. Plenty of objects are visible enough to annotate, but occluded or far enough away that missing them says little about model quality. Ignoring them completely is also wrong, because a correct detection of such an object would then count as a false positive.

The background-aware evaluation splits ground truth into two tiers:

- **Foreground instances**: objects the model should find. Missing one is a false negative.
- **Background instances**: objects the model may find. A detection matching one is excluded from the evaluation entirely, so it does not count as a false positive, and missing one is never a false negative.

Foreground and background are linked through a class ID convention. IDs are 5 digits, the last digit marks the tier:

```
10000  peduncle             (foreground)
10001  peduncle_background  (background)
20000  petiole              (foreground)
20001  petiole_background   (background)
```

Predictions always use the foreground ID and can match ground truth of either tier with the same base class. Recall is computed against foreground instances only.

### Usage

```python
from faster_coco_eval import COCO, COCOeval_background

cocoGt = COCO("ground_truth.json")   # contains foreground and background annotations
cocoDt = cocoGt.loadRes("predictions.json")

cocoEval = COCOeval_background(cocoGt, cocoDt, "segm")
cocoEval.extra_calc = True           # needed for the detailed statistics below
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

stats = cocoEval.get_background_stats()
print(stats["foreground_recall"])    # foreground TP / foreground GT
print(stats["background_tp"])        # detections matched to background GT
print(stats["background_fn"])        # always 0 by design
```

Evaluation with `useCats = 0` (all classes pooled) is supported as well. For precision-recall and F1 curves that use the same matching rules there is `CurvesBackground`:

```python
from faster_coco_eval.extra import CurvesBackground

curves = CurvesBackground(cocoGt, cocoDt, "segm", useCats=False, iou_tresh=0.50)
fig = curves.plot_pre_rec(return_fig=True)
```

A more detailed description of the method and the data format is in [README_BACKGROUND.md](README_BACKGROUND.md). A small worked example (one image, five annotations, four predictions) is included in `faster_coco_eval/minimal_json_examples/`, together with the script that generates it.

## Branch keypoint evaluation

For thin plant structures we annotate each object with two keypoints instead of a mask: an `Edge` point where the branch attaches, and a `Pt` point that defines its direction. OKS is a poor fit for this annotation style, so `COCOeval_branch_keypoints` matches detections by pixel distance:

- the predicted `Edge` point must lie within `distance_threshold` pixels (Euclidean) of the ground truth `Edge` point, and
- the predicted `Pt` point must lie within `distance_threshold` pixels of the ray that starts at the ground truth `Edge` point and passes through the ground truth `Pt` point.

The ray criterion means a prediction is not punished for placing `Pt` further along or short of the annotated point, as long as the direction is right. Both criteria must hold for a match. The usual AP and AR machinery on top is unchanged.

In the paper the `Edge` keypoint is called the node point N and the `Pt` keypoint the node direction point P. The AP at IoU 0.50 in the summary output corresponds to the binary match criterion described in the paper: internally a match is given a score between 0.5 and 1.0 based on its distances, so the higher IoU thresholds act as progressively tighter pixel tolerances.

Keypoints are stored in the order `[x_edge, y_edge, v_edge, x_pt, y_pt, v_pt]`, matching a category definition of `"keypoints": ["Edge", "Pt"]`. Ground truth objects with an invisible keypoint (v = 0) are excluded from evaluation: detections matching them are absorbed rather than counted as false positives.

```python
from faster_coco_eval import COCO, COCOeval_branch_keypoints

cocoGt = COCO("ground_truth.json")
cocoDt = cocoGt.loadRes("predictions.json")

cocoEval = COCOeval_branch_keypoints(
    cocoGt, cocoDt, iouType="keypoints", distance_threshold=10.0
)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
```

`COCOeval_branch_keypoints_background` combines both methods: distance based keypoint matching with background instances that never count as false negatives. It takes the same arguments as `COCOeval_branch_keypoints` and also provides `get_background_stats()`.

## Installation

Install from source (a C++ compiler is required for the extension module):

```bash
git clone https://github.com/TAW-V/wur-av-bench-cocoeval.git
cd wur-av-bench-cocoeval
pip install -e .
```

Do not install this via `pip install faster-coco-eval`: that pulls the upstream package from PyPI, which does not contain the methods described here. The import name is unchanged, so this fork works as a drop-in replacement for the upstream package (do not install both in the same environment).

## Tests

The new functionality is covered by unit tests:

```bash
python -m unittest tests.test_background_eval tests.test_branch_keypoints
```

## Related repositories

- [wur-av-bench](https://github.com/TAW-V/wur-av-bench): the benchmark tools for Tasks III and IV (multi-view 3D reconstruction and 5D pose evaluation), including the 3D node annotation tool. This repository covers the image-level evaluation of Tasks I and II.
- [faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval): the upstream project this repository is forked from.

## Dataset

The WUR-AV-Bench dataset (RGB images, point clouds, camera poses and the annotation files used with this evaluator) is available at [wur-av-bench-dataset](https://github.com/TAW-V/wur-av-bench-dataset).

## Relation to faster-coco-eval and pycocotools

All credit for the fast C++ evaluation backend and the base API goes to [MiXaiLL76/faster_coco_eval](https://github.com/MiXaiLL76/faster_coco_eval), which is itself a drop-in replacement for [pycocotools](https://github.com/cocodataset/cocoapi). Everything from upstream still works, including `init_as_pycocotools()`. One behavioural note: this fork keeps explicit `ignore` flags on ground truth annotations intact during evaluation (upstream overwrites them with the `iscrowd` flag), since the background-aware evaluation relies on them. Crowd annotations are still ignored as before.

## License

Apache 2.0, same as the upstream project. See [LICENSE](LICENSE).
