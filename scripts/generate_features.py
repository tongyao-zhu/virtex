import argparse
import multiprocessing as mp
import os
from typing import Any, List

from loguru import logger
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from virtex.config import Config
from virtex.factories import PretrainingModelFactory, DownstreamDatasetFactory
from virtex.models.downstream import FeatureExtractor
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser, common_setup

parser = common_parser(
    description="Train SVMs for VOC2007 classification on a pretrained model."
)
group = parser.add_argument_group("Downstream config arguments.")
group.add_argument(
    "--down-config", metavar="FILE", help="Path to a downstream config file."
)
group.add_argument(
    "--down-config-override",
    nargs="*",
    default=[],
    help="A list of key-value pairs to modify downstream config params.",
)
group.add_argument(
    "--csv",
    help="the path to csv",
)
group.add_argument(
    "--mode",
    help='the mode we are in'
)

# fmt: off
parser.add_argument_group("Checkpointing")
group.add_argument(
    "--layer", choices=["layer1", "layer2", "layer3", "layer4", "avgpool"],
    default="layer4", help="Evaluate features extracted from this layer."
)
parser.add_argument(
    "--weight-init", choices=["random", "imagenet", "torchvision", "virtex"],
    default="virtex", help="""How to initialize weights:
        1. 'random' initializes all weights randomly
        2. 'imagenet' initializes backbone weights from torchvision model zoo
        3. {'torchvision', 'virtex'} load state dict from --checkpoint-path
            - with 'torchvision', state dict would be from PyTorch's training
              script.
            - with 'virtex' it should be for our full pretrained model."""
)
parser.add_argument(
    "--checkpoint-path",
    help="Path to load checkpoint and run downstream task evaluation."
)


# fmt: on


def train_test_single_svm(args):
    feats_train, tgts_train, feats_test, tgts_test, cls_name = args
    SVM_COSTS = [0.01, 0.1, 1.0, 10.0]

    cls_labels = np.copy(tgts_train)
    # Meaning of labels in VOC/COCO original loaded target files:
    # label 0 = not present, set it to -1 as svm train target
    # label 1 = present. Make the svm train target labels as -1, 1.
    cls_labels[np.where(cls_labels == 0)] = -1

    # See which cost maximizes the AP for this class.
    best_crossval_ap: float = 0.0
    best_crossval_clf = None
    best_cost: float = 0.0

    # fmt: off
    for cost in SVM_COSTS:
        clf = LinearSVC(
            C=cost, class_weight={1: 2, -1: 1}, penalty="l2",
            loss="squared_hinge", max_iter=2000,
        )
        ap_scores = cross_val_score(
            clf, feats_train, cls_labels, cv=3, scoring="average_precision",
        )
        clf.fit(feats_train, cls_labels)

        # Keep track of best SVM (based on cost) for each class.
        if ap_scores.mean() > best_crossval_ap:
            best_crossval_ap = ap_scores.mean()
            best_crossval_clf = clf
            best_cost = cost

    logger.info(f"Best SVM {cls_name}: cost {best_cost}, mAP {best_crossval_ap}")
    # fmt: on

    # -------------------------------------------------------------------------
    #   TEST THE TRAINED SVM (PER CLASS)
    # -------------------------------------------------------------------------
    predictions = best_crossval_clf.decision_function(feats_test)
    evaluate_data_inds = tgts_test != -1
    eval_preds = predictions[evaluate_data_inds]

    cls_labels = np.copy(tgts_test)
    eval_cls_labels = cls_labels[evaluate_data_inds]
    eval_cls_labels[np.where(eval_cls_labels == 0)] = -1

    # Binarize class labels to make AP targets.
    targets = eval_cls_labels > 0
    return average_precision_score(targets, eval_preds)


def main(_A: argparse.Namespace):
    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device = torch.device("cpu")
    else:
        # Get the current device (this will be zero here by default).
        device = torch.cuda.current_device()

    # Create a downstream config object (this will be immutable) and perform
    # common setup such as logging and setting up serialization directory.
    _DOWNC = Config(_A.down_config, _A.down_config_override)
    common_setup(_DOWNC, _A, job_type="downstream")

    # Create a (pretraining) config object and backup in serialization directory.
    _C = Config(_A.config, _A.config_override)
    _C.dump(os.path.join(_A.serialization_dir, "pretrain_config.yaml"))

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, AND FEATURE EXTRACTOR
    # -------------------------------------------------------------------------
    train_dataset = DownstreamDatasetFactory.from_config(_DOWNC, split="trainval", csv=_A.csv)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_DOWNC.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        shuffle=False,
    )

    print(f"train dataset length {len(train_dataset)}")
    # Initialize from a checkpoint, but only keep the visual module.
    model = PretrainingModelFactory.from_config(_C)

    # Load weights according to the init method, do nothing for `random`, and
    # `imagenet` is already taken care of.
    if _A.weight_init == "virtex":
        ITERATION = CheckpointManager(model=model).load(_A.checkpoint_path)
    elif _A.weight_init == "torchvision":
        # Keep strict=False because this state dict may have weights for
        # last fc layer.
        model.visual.cnn.load_state_dict(
            torch.load(_A.checkpoint_path, map_location="cpu")["state_dict"],
            strict=False,
        )

    model = FeatureExtractor(model, layer_name=_A.layer, flatten_and_normalize=True)
    model = model.to(device).eval()

    # -------------------------------------------------------------------------
    #   EXTRACT FEATURES FOR TRAINING SVMs
    # -------------------------------------------------------------------------

    features_train: List[torch.Tensor] = []
    targets_train: List[torch.Tensor] = []
    print("input csv is {}".format(_A.csv))

    # VOC07 is small, extract all features and keep them in memory.
    count = 0
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Extracting train features:"):
            features = model(batch["image"].to(device))
            count += 1
            print("train features has shape {}, video_id {}".format(features.shape, batch['image_id']))
            if count % 1000 == 0:
                torch.save(features_train, "./features_train_first_{}_{}.pt".format(_A.mode, count))
                features_train = []
            features_train.append(features.cpu())

    features_train = torch.cat(features_train, dim=0).numpy()
    torch.save(features_train, "./features_train_{}.pt".format(_A.mode))
    print('finished saving features')


if __name__ == "__main__":
    _A = parser.parse_args()

    if _A.num_gpus_per_machine > 1:
        raise ValueError("Using multiple GPUs is not supported for this script.")

    # Add an arg in config override if `--weight-init` is imagenet.
    if _A.weight_init == "imagenet":
        _A.config_override.extend(["MODEL.VISUAL.PRETRAINED", True])

    # No distributed training here, just a single process.
    main(_A)
