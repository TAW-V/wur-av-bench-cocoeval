"""Background-aware Curves class for faster-coco-eval.

This module provides a Curves class that uses COCOeval_background instead of
the standard COCOeval_faster, allowing PR curves to incorporate background
instance evaluation logic.

Original work Copyright (c) 2025 TAW-V
"""

import copy
import logging
import numpy as np
from typing import Optional

from ..core import COCO, COCOeval_background
from .draw import plot_ced_metric, plot_f1_confidence, plot_pre_rec
from .curves import Curves

logger = logging.getLogger(__name__)


class CurvesBackground(Curves):
    """Background-aware version of Curves class.
    
    This class uses COCOeval_background for evaluation, allowing it to
    correctly handle background instances in PR curve generation.
    
    Key behaviors:
    - When useCats=True: Each foreground category is evaluated separately,
      but predictions can still match background GTs of the same base class.
    - When useCats=False: All categories are evaluated together. Predictions
      can match any foreground or background GT of the same base class, but
      only foreground GTs count toward false negatives and recall denominator.
    """
    
    def __init__(
        self,
        cocoGt: COCO = None,
        cocoDt: COCO = None,
        iouType: str = "bbox",
        min_score: float = 0,
        iou_tresh: float = 0.0,
        recall_count: int = 100,
        useCats: bool = False,
        kpt_oks_sigmas: list = None,
    ):
        """Initialize the CurvesBackground object.

        Args:
            cocoGt (COCO, optional): Ground truth COCO object. Defaults to None.
            cocoDt (COCO, optional): Detection results COCO object. Defaults to None.
            iouType (str, optional): Type of IoU evaluation ('bbox', 'segm', 'keypoints'). Defaults to "bbox".
            min_score (float, optional): Minimum score threshold for detections. Defaults to 0.
            iou_tresh (float, optional): IoU threshold for evaluation. Defaults to 0.0.
            recall_count (int, optional): Number of recall thresholds. Defaults to 100.
            useCats (bool, optional): Whether to use categories in evaluation. Defaults to False.
            kpt_oks_sigmas (list, optional): List of OKS sigmas for keypoints evaluation. Defaults to None.

        Raises:
            AssertionError: If cocoGt is None.
        """
        # Store parameters (don't call parent __init__ as it uses COCOeval_faster)
        self.iouType = iouType
        self.min_score = min_score
        self.iou_tresh = iou_tresh
        self.useCats = useCats
        self.recall_count = recall_count
        self.cocoGt = copy.deepcopy(cocoGt)
        self.cocoDt = copy.deepcopy(cocoDt)
        self.eval = None

        if iouType == "keypoints":
            self.useCats = True
            self.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        else:
            self.kpt_oks_sigmas = None

        assert self.cocoGt is not None, "cocoGt is empty"

        if (self.cocoGt is not None) and (self.cocoDt is not None):
            self.drop_cocodt_by_score(min_score=min_score)
            self.evaluate()
    
    def evaluate(self):
        """Run COCO evaluation with background support and accumulate results.
        
        This override uses COCOeval_background instead of COCOeval_faster.

        Raises:
            AssertionError: If cocoDt is None.
        """
        assert self.cocoDt is not None, "cocoDt is empty"

        # Use COCOeval_background instead of COCOeval_faster
        cocoEval = COCOeval_background(
            self.cocoGt,
            self.cocoDt,
            self.iouType,
            kpt_oks_sigmas=self.kpt_oks_sigmas,
        )
        cocoEval.extra_calc = True
        cocoEval.params.maxDets = [len(self.cocoGt.anns)]

        self.recThrs = np.linspace(0, 1, self.recall_count + 1, endpoint=True)
        cocoEval.params.recThrs = self.recThrs

        if self.iouType != "keypoints":
            cocoEval.params.iouThrs = [self.iou_tresh]

        cocoEval.params.areaRng = [[0, 10000000000]]
        cocoEval.params.useCats = int(self.useCats)

        self.cocoEval = cocoEval

        cocoEval.evaluate()
        cocoEval.accumulate()

        self.eval = cocoEval.eval
