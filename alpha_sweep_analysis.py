"""
Alpha Weight Sweep Analysis for Ensemble Model
Evaluate how different ensemble weights affect model performance.

Author: Machine Learning Researcher
Date: 2025
"""

import os
import json
import csv
import math
import numpy as np
import random
from pathlib import Path
from textwrap import fill
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import clip
from sklearn.metrics import average_precision_score, f1_score
from episode_replay import load_replay_indices

# ==================== Configuration ====================
# Dataset paths
DATASET_ROOT = os.environ.get("K5_COCO_ROOT", os.environ.get("COCO_ROOT", "coco2014"))
TRAIN2014_DIR = os.path.join(DATASET_ROOT, "train2014")
VAL2014_DIR = os.path.join(DATASET_ROOT, "val2014")

# Dataset split files
DATA_DIR = os.environ.get("K5_DATA_DIR", os.environ.get("DEPF_DATA_DIR", "coco2014_data"))
SPLIT_FILE = os.path.join(DATA_DIR, "coco_mutually_exclusive_split.json")
MAPPING_FILE = os.path.join(DATA_DIR, "category_mapping.json")

# Model checkpoint paths
DEFAULT_RUN_DIR = os.path.join(os.environ.get("MULTISEED_ROOT", "coco2014_multiseed_k5"), "seed_42")
LC_PROTONETS_CKPT = os.environ.get("K5_LC_CHECKPOINT", os.path.join(
    DEFAULT_RUN_DIR, "lc_protonets", "checkpoints",
    "lc_protonets_52way_5shot_dim2048_finetuneTrue_euclidean_bias14.7.pth"
))
COMC_CLASSIFIER_CKPT = os.environ.get("K5_COMC_CHECKPOINT", os.path.join(
    DEFAULT_RUN_DIR, "comc", "comc_classifier_best.pth"
))

# Output paths
OUTPUT_DIR = os.environ.get("K5_OUTPUT_DIR", "coco2014_alpha_sweep_results")
if os.environ.get("K5_EPISODE_MANIFEST") and not os.environ.get("K5_OUTPUT_DIR"):
    raise ValueError("Shared episode replay requires K5_OUTPUT_DIR; use a new directory for supplementary results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Experiment settings
ALPHA_VALUES = np.round(np.arange(0.0, 1.01, 0.1), 1)  # 0.0, 0.1, ..., 1.0
N_WAY = 16
K_SHOT = 5
N_QUERY = 4
N_EPISODES = 200
EPISODE_SEED = 42  # Use a fixed seed for a fair comparison.
EPISODE_MANIFEST = os.environ.get("K5_EPISODE_MANIFEST")
EPISODE_REPLAY_AUDIT = None
ANALYSIS_ALPHA = 0.1
CALIBRATION_BINS = 10
THRESHOLD_VALUES = np.round(np.arange(0.05, 0.951, 0.05), 2)
SAVE_QUERY_LEVEL_CSV = True
FIXED_THRESHOLD = 0.5

# Case-study protocol. Selection is deterministic over every unique query image
# evaluated at ANALYSIS_ALPHA; it never relies on visual inspection of images.
CASE_STUDY_ENABLED = True
CASE_STUDY_DIRNAME = "case_study"
CASE_STUDY_CASES_PER_TYPE = 1
CASE_STUDY_PREFERRED_MIN_TRUE_LABELS = 3
CASE_STUDY_ENFORCE_DISTINCT_EPISODES = True
CASE_STUDY_MAX_ADDITIONAL_PLOT_LABELS = 2
CASE_STUDY_MAX_SUPPORT_PROTOTYPES = 12
CASE_STUDY_SUPPORT_COLUMNS = 4
CASE_STUDY_FIGURE_DPI = 600
# Opt in only after reviewing image content and redistribution permissions.
CASE_STUDY_EXPORT_SOURCE_IMAGES = os.environ.get("DEPF_EXPORT_SOURCE_IMAGES", "0") == "1"
CASE_STUDY_SOURCE_IMAGE_DIRNAME = "source_images"
CASE_STUDY_SOURCE_MANIFEST_FILENAME = "source_images_manifest.json"

CASE_STUDY_MODEL_ORDER = ("tp_cmc", "lc_protonets", "ensemble")
CASE_STUDY_MODEL_NAMES = {
    "tp_cmc": "TP-CMC",
    "lc_protonets": "LC-Protonets",
    "ensemble": "DEPF",
}
CASE_STUDY_MODEL_COLORS = {
    "tp_cmc": "#4C78A8",
    "lc_protonets": "#F58518",
    "ensemble": "#54A24B",
}

CASE_STUDY_TYPE_NAMES = {
    "success": "Successful correction",
    "visual_correction": "Visual-evidence correction",
    "failure": "Failure / boundary",
}

# CoMC parameters
COMC_ALPHA = 0.7
COMC_TEMPERATURE = 0.07
CLIP_MODEL = "ViT-B/32"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("="*70)
print("ALPHA SWEEP ANALYSIS")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Alpha values: {list(ALPHA_VALUES)}")
print(f"Episodes per alpha: {N_EPISODES}")
print(f"Episode seed: {EPISODE_SEED}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*70)


# ==================== Model definitions ====================

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, feature_dim=2048, finetune=True):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        if feature_dim != 2048:
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, feature_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.projection = nn.Flatten()

        self.finetune = finetune
        if not finetune:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        return self.projection(x)


class LCProtonets(nn.Module):
    def __init__(self, feature_extractor, distance_metric='euclidean', init_bias=14.7):
        super(LCProtonets, self).__init__()
        self.feature_extractor = feature_extractor
        self.distance_metric = distance_metric
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def compute_prototypes(self, support_features, support_labels):
        from collections import defaultdict
        label_combinations = [tuple(sorted(labels)) for labels in support_labels]
        combination_to_features = defaultdict(list)
        for features, combo in zip(support_features, label_combinations):
            combination_to_features[combo].append(features)

        prototypes = {}
        for combo, feature_list in combination_to_features.items():
            features_stack = torch.stack(feature_list)
            prototypes[combo] = features_stack.mean(dim=0)
        return prototypes

    def compute_distances(self, query_features, prototypes):
        distances = {}
        for combo, prototype in prototypes.items():
            if self.distance_metric == 'euclidean':
                dist = torch.norm(query_features - prototype.unsqueeze(0), dim=1)
            else:
                query_norm = F.normalize(query_features, p=2, dim=1)
                prototype_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1)
                similarity = F.cosine_similarity(query_norm, prototype_norm, dim=1)
                dist = 1 - similarity
            distances[combo] = dist
        return distances

    def aggregate_to_labels(self, distances, episode_classes, aggregation='max'):
        n_query = list(distances.values())[0].shape[0]
        n_classes = len(episode_classes)
        device = list(distances.values())[0].device

        if aggregation == 'max':
            label_scores = torch.full((n_query, n_classes), float('inf'), device=device)
        else:
            label_scores = torch.zeros((n_query, n_classes), device=device)

        temp_scores = []
        for combo, dist in distances.items():
            for cat_id in combo:
                if cat_id in episode_classes:
                    class_idx = episode_classes.index(cat_id)
                    temp_scores.append((class_idx, dist))

        if aggregation == 'max':
            for class_idx, dist in temp_scores:
                label_scores[:, class_idx] = torch.where(
                    dist < label_scores[:, class_idx],
                    dist,
                    label_scores[:, class_idx]
                )
        label_logits = -label_scores + self.bias
        return label_logits

    def forward(self, episode):
        support_features = self.feature_extractor(episode['support_images'])
        query_features = self.feature_extractor(episode['query_images'])

        prototypes = self.compute_prototypes(support_features, episode['support_labels'])
        distances = self.compute_distances(query_features, prototypes)
        predictions = self.aggregate_to_labels(distances, episode['episode_classes'])

        return predictions


class CoMCClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CoMCClassifier, self).__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        return self.classifier(x)


# ==================== Dataset definitions ====================

class DualPreprocessCOCODataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        train_image_dir: str,
        val_image_dir: str,
        lc_transform=None,
        comc_transform=None,
        cat_id_to_name: Dict[int, str] = None
    ):
        self.samples = samples
        self.train_image_dir = train_image_dir
        self.val_image_dir = val_image_dir
        self.lc_transform = lc_transform
        self.comc_transform = comc_transform
        self.cat_id_to_name = cat_id_to_name or {}

    def __len__(self):
        return len(self.samples)

    def _get_image_path(self, sample: Dict) -> str:
        filename = sample.get('filename', f"{sample['image_id']:012d}.jpg")
        val_path = os.path.join(self.val_image_dir, filename)
        if os.path.exists(val_path):
            return val_path
        train_path = os.path.join(self.train_image_dir, filename)
        if os.path.exists(train_path):
            return train_path
        raise FileNotFoundError(f"Image not found: {filename}")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self._get_image_path(sample)
        image = Image.open(image_path).convert('RGB')

        if self.lc_transform:
            image_lc = self.lc_transform(image)
        else:
            image_lc = image

        if self.comc_transform:
            image_comc = self.comc_transform(image)
        else:
            image_comc = image

        labels = torch.tensor(sample['category_ids'], dtype=torch.long)

        return {
            'image_lc': image_lc,
            'image_comc': image_comc,
            'labels': labels,
            'image_id': sample['image_id'],
            'category_ids': sample['category_ids']
        }


class EpisodeSampler:
    def __init__(self, dataset, n_way, k_shot, n_query=4, seed=42,
                 replay_manifest=None, class_ids=None):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.seed = seed
        self.replay_episodes, self.replay_audit = None, None
        if replay_manifest is not None:
            if class_ids is None:
                raise ValueError("Replay requires ordered novel class_ids")
            self.replay_episodes, self.replay_audit = load_replay_indices(
                replay_manifest, dataset.samples, class_ids, k_shot, n_query)
            self.replay_classes = list(map(int, class_ids))

        self.cat_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            for cat_id in sample['category_ids']:
                self.cat_to_indices[cat_id].append(idx)

    def sample_fixed_classes(self, fixed_cat_ids, episode_idx=0):
        """Sample an episode from fixed classes with reproducible seeding."""
        if self.replay_episodes is not None:
            if list(map(int, fixed_cat_ids)) != self.replay_classes:
                raise ValueError("Replay episode class order changed")
            if not 0 <= episode_idx < len(self.replay_episodes):
                raise ValueError("Replay episode index outside manifest")
            record = self.replay_episodes[episode_idx]
            return self._materialize(record["support"], record["query"], self.replay_classes)
        # Keep each episode identical across runs.
        np.random.seed(self.seed + episode_idx)
        random.seed(self.seed + episode_idx)

        episode_cats = fixed_cat_ids
        support_samples = []
        query_samples = []

        used_support_indices = set()
        used_query_indices = set()

        for cat_id in episode_cats:
            cat_indices = self.cat_to_indices[cat_id]

            available_for_support = [idx for idx in cat_indices
                                    if idx not in used_query_indices]
            available_for_query = [idx for idx in cat_indices
                                 if idx not in used_support_indices]

            needed_support = self.k_shot
            needed_query = self.n_query

            if len(available_for_support) < needed_support or len(available_for_query) < needed_query:
                if len(cat_indices) < self.k_shot + self.n_query:
                    selected_indices = random.choices(cat_indices, k=self.k_shot + self.n_query)
                else:
                    selected_indices = random.sample(cat_indices, k=self.k_shot + self.n_query)
            else:
                support_indices = random.sample(available_for_support, needed_support)
                available_for_query_filtered = [idx for idx in available_for_query
                                              if idx not in support_indices]

                if len(available_for_query_filtered) < needed_query:
                    remaining = needed_query - len(available_for_query_filtered)
                    query_indices = available_for_query_filtered + random.choices(
                        [idx for idx in cat_indices if idx not in available_for_query_filtered],
                        k=remaining
                    )
                else:
                    query_indices = random.sample(available_for_query_filtered, needed_query)

                selected_indices = support_indices + query_indices

            support_indices = selected_indices[:self.k_shot]
            query_indices = selected_indices[self.k_shot:self.k_shot + self.n_query]

            used_support_indices.update(support_indices)
            used_query_indices.update(query_indices)

            support_samples.extend(support_indices)
            query_samples.extend(query_indices)

        return self._materialize(support_samples, query_samples, episode_cats)

    def _materialize(self, support_samples, query_samples, episode_cats):
        # Random sampling and replay use identical image preprocessing and tensor construction.
        support_data = [self.dataset[idx] for idx in support_samples]
        query_data = [self.dataset[idx] for idx in query_samples]

        support_images_lc = torch.stack([item['image_lc'] for item in support_data])
        support_images_comc = torch.stack([item['image_comc'] for item in support_data])
        query_images_lc = torch.stack([item['image_lc'] for item in query_data])
        query_images_comc = torch.stack([item['image_comc'] for item in query_data])

        support_labels = [item['labels'].tolist() for item in support_data]
        query_labels = [item['labels'].tolist() for item in query_data]
        support_image_ids = [item['image_id'] for item in support_data]
        query_image_ids = [item['image_id'] for item in query_data]
        support_filenames = [
            self.dataset.samples[idx].get(
                'filename', f"{int(self.dataset.samples[idx]['image_id']):012d}.jpg"
            )
            for idx in support_samples
        ]
        query_filenames = [
            self.dataset.samples[idx].get(
                'filename', f"{int(self.dataset.samples[idx]['image_id']):012d}.jpg"
            )
            for idx in query_samples
        ]

        return {
            'support_images_lc': support_images_lc,
            'support_images_comc': support_images_comc,
            'query_images_lc': query_images_lc,
            'query_images_comc': query_images_comc,
            'support_labels': support_labels,
            'query_labels': query_labels,
            'support_image_ids': support_image_ids,
            'query_image_ids': query_image_ids,
            'support_sample_indices': support_samples,
            'query_sample_indices': query_samples,
            'support_filenames': support_filenames,
            'query_filenames': query_filenames,
            'episode_classes': list(episode_cats)
        }


# ==================== CoMC functions ====================

def build_class_text_embeddings(class_names, clip_model, template, device):
    text_descriptions = [template.format(name) for name in class_names]
    text_tokens = clip.tokenize(text_descriptions).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.float()


def cross_modal_mapping(image_embeddings, class_text_embeddings, temperature=0.07):
    if image_embeddings.dim() == 2:
        image_embeddings = image_embeddings.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False

    similarities = torch.einsum('bps,cs->bpc', image_embeddings, class_text_embeddings) / temperature
    weights = F.softmax(similarities, dim=-1)
    mapped_embeddings = torch.einsum('bpc,cs->bps', weights, class_text_embeddings)

    if squeeze_output:
        mapped_embeddings = mapped_embeddings.squeeze(1)

    return mapped_embeddings


def comc_inference(clip_model, comc_classifier, query_images_comc,
                   class_text_embeddings, alpha=0.7, temperature=0.07, device=DEVICE):
    query_images_comc = query_images_comc.to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(query_images_comc)
        image_features = image_features.float()
        global_features = image_features / image_features.norm(dim=-1, keepdim=True)
        batch_size = global_features.shape[0]
        local_features = global_features.unsqueeze(1)

    # Cross-modal mapping and fusion.
    mapped_global = cross_modal_mapping(global_features, class_text_embeddings, temperature)
    fused_global = alpha * global_features + (1 - alpha) * mapped_global

    global_logits = comc_classifier(fused_global)
    global_probs = torch.sigmoid(global_logits)

    mapped_local = cross_modal_mapping(local_features, class_text_embeddings, temperature)
    fused_local = alpha * local_features + (1 - alpha) * mapped_local

    batch_size, num_patches, embedding_dim = fused_local.shape
    fused_local_flat = fused_local.view(batch_size * num_patches, embedding_dim)

    local_logits_flat = comc_classifier(fused_local_flat)
    local_probs_flat = torch.sigmoid(local_logits_flat)
    local_probs = local_probs_flat.view(batch_size, num_patches, -1)
    aggregated_local_probs, _ = local_probs.max(dim=1)

    final_probs = 0.5 * (global_probs + aggregated_local_probs)
    return final_probs


# ==================== Ensemble model ====================

class EnsembleModel:
    def __init__(self, lc_protonets, clip_model, comc_classifier,
                 class_text_embeddings, alpha=0.5, comc_alpha=0.7,
                 comc_temperature=0.07, device=DEVICE):
        self.lc_protonets = lc_protonets
        self.clip_model = clip_model
        self.comc_classifier = comc_classifier
        self.class_text_embeddings = class_text_embeddings
        self.alpha = alpha
        self.comc_alpha = comc_alpha
        self.comc_temperature = comc_temperature
        self.device = device

    @torch.no_grad()
    def predict(self, episode, return_individual=False):
        # Path 1: LC-Protonets.
        self.lc_protonets.eval()
        episode_lc = {
            'support_images': episode['support_images_lc'].to(self.device),
            'query_images': episode['query_images_lc'].to(self.device),
            'support_labels': episode['support_labels'],
            'query_labels': episode['query_labels'],
            'episode_classes': episode['episode_classes']
        }
        lc_logits = self.lc_protonets(episode_lc)
        p_proto = torch.sigmoid(lc_logits)

        # Path 2: CoMC.
        self.comc_classifier.eval()
        query_images_comc = episode['query_images_comc']
        p_comc = comc_inference(
            self.clip_model,
            self.comc_classifier,
            query_images_comc,
            self.class_text_embeddings,
            alpha=self.comc_alpha,
            temperature=self.comc_temperature,
            device=self.device
        )

        # Combine predictions.
        p_final = self.alpha * p_proto + (1 - self.alpha) * p_comc

        if return_individual:
            individual_dict = {
                'p_proto': p_proto.cpu(),
                'p_comc': p_comc.cpu()
            }
            return p_final.cpu(), individual_dict
        else:
            return p_final.cpu()


# ==================== Evaluation functions ====================

def compute_multilabel_metrics(predictions, targets):
    predictions_np = predictions.numpy() if torch.is_tensor(predictions) else predictions
    targets_np = targets.numpy() if torch.is_tensor(targets) else targets

    try:
        map_macro = average_precision_score(targets_np, predictions_np, average='macro')
        map_micro = average_precision_score(targets_np, predictions_np, average='micro')
    except Exception as e:
        print(f"Error computing AP: {e}")
        map_macro = 0.0
        map_micro = 0.0

    pred_labels = (predictions_np > FIXED_THRESHOLD).astype(int)

    try:
        f1_macro = f1_score(targets_np, pred_labels, average='macro', zero_division=0)
        f1_micro = f1_score(targets_np, pred_labels, average='micro', zero_division=0)
    except Exception as e:
        print(f"Error computing F1: {e}")
        f1_macro = 0.0
        f1_micro = 0.0

    accuracy = (pred_labels == targets_np).mean()

    return {
        'map_macro': map_macro,
        'map_micro': map_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'accuracy': accuracy
    }


def compute_probability_sharpness(predictions, targets):
    """Quantify how far probabilities are from the 0.5 decision boundary."""
    predictions_np = predictions.numpy() if torch.is_tensor(predictions) else predictions
    targets_np = targets.numpy() if torch.is_tensor(targets) else targets
    predictions_np = np.asarray(predictions_np, dtype=np.float64)
    targets_np = np.asarray(targets_np, dtype=np.int32)

    flat_probs = predictions_np.reshape(-1)
    flat_targets = targets_np.reshape(-1)
    eps = 1e-12
    entropy = -(flat_probs * np.log(flat_probs + eps) +
                (1.0 - flat_probs) * np.log(1.0 - flat_probs + eps))

    pos_probs = flat_probs[flat_targets == 1]
    neg_probs = flat_probs[flat_targets == 0]

    return {
        'mean_abs_distance_to_0_5': float(np.mean(np.abs(flat_probs - 0.5))),
        'median_abs_distance_to_0_5': float(np.median(np.abs(flat_probs - 0.5))),
        'ambiguous_ratio_0_4_0_6': float(np.mean((flat_probs >= 0.4) & (flat_probs <= 0.6))),
        'confident_ratio_0_1_0_9': float(np.mean((flat_probs <= 0.1) | (flat_probs >= 0.9))),
        'mean_entropy': float(np.mean(entropy)),
        'positive_mean_prob': float(np.mean(pos_probs)) if pos_probs.size else 0.0,
        'negative_mean_prob': float(np.mean(neg_probs)) if neg_probs.size else 0.0,
        'positive_ambiguous_ratio_0_4_0_6': float(np.mean((pos_probs >= 0.4) & (pos_probs <= 0.6))) if pos_probs.size else 0.0,
        'negative_ambiguous_ratio_0_4_0_6': float(np.mean((neg_probs >= 0.4) & (neg_probs <= 0.6))) if neg_probs.size else 0.0,
        'probability_separation_pos_minus_neg': float(np.mean(pos_probs) - np.mean(neg_probs)) if pos_probs.size and neg_probs.size else 0.0
    }


def compute_calibration_metrics(predictions, targets, n_bins=10):
    """Multilabel calibration over all query-label pairs."""
    predictions_np = predictions.numpy() if torch.is_tensor(predictions) else predictions
    targets_np = targets.numpy() if torch.is_tensor(targets) else targets
    flat_probs = np.asarray(predictions_np, dtype=np.float64).reshape(-1)
    flat_targets = np.asarray(targets_np, dtype=np.float64).reshape(-1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []

    for bin_idx in range(n_bins):
        left = bins[bin_idx]
        right = bins[bin_idx + 1]
        if bin_idx == n_bins - 1:
            mask = (flat_probs >= left) & (flat_probs <= right)
        else:
            mask = (flat_probs >= left) & (flat_probs < right)

        count = int(mask.sum())
        if count == 0:
            avg_confidence = 0.0
            empirical_accuracy = 0.0
            gap = 0.0
        else:
            avg_confidence = float(flat_probs[mask].mean())
            empirical_accuracy = float(flat_targets[mask].mean())
            gap = abs(avg_confidence - empirical_accuracy)
            ece += (count / flat_probs.size) * gap

        bin_rows.append({
            'bin_left': float(left),
            'bin_right': float(right),
            'count': count,
            'avg_confidence': avg_confidence,
            'empirical_positive_rate': empirical_accuracy,
            'gap': float(gap)
        })

    brier = float(np.mean((flat_probs - flat_targets) ** 2))
    return {
        'ece': float(ece),
        'brier_score': brier,
        'bins': bin_rows
    }


def compute_threshold_sweep(predictions, targets, thresholds=THRESHOLD_VALUES):
    predictions_np = predictions.numpy() if torch.is_tensor(predictions) else predictions
    targets_np = targets.numpy() if torch.is_tensor(targets) else targets
    predictions_np = np.asarray(predictions_np)
    targets_np = np.asarray(targets_np)

    rows = []
    for threshold in thresholds:
        threshold = round(float(threshold), 2)
        pred_labels = (predictions_np > threshold).astype(int)
        rows.append({
            'threshold': threshold,
            'f1_macro': float(f1_score(targets_np, pred_labels, average='macro', zero_division=0)),
            'f1_micro': float(f1_score(targets_np, pred_labels, average='micro', zero_division=0))
        })

    best_macro = max(rows, key=lambda item: item['f1_macro'])
    best_micro = max(rows, key=lambda item: item['f1_micro'])
    return {
        'curve': rows,
        'best_macro': best_macro,
        'best_micro': best_micro
    }


def compute_gmean_relative_gain(metrics, baseline_metrics):
    metric_keys = ['map_macro', 'map_micro', 'f1_macro', 'f1_micro']
    ratios = []
    for key in metric_keys:
        baseline = max(float(baseline_metrics[key]), 1e-12)
        ratios.append(max(float(metrics[key]), 1e-12) / baseline)
    return float(np.prod(ratios) ** (1.0 / len(ratios)))


def to_builtin_float_dict(metrics):
    return {key: float(value) for key, value in metrics.items()}


def build_probability_analysis(predictions_by_model, targets):
    analysis = {}
    for model_name, predictions in predictions_by_model.items():
        analysis[model_name] = {
            'sharpness': compute_probability_sharpness(predictions, targets),
            'calibration': compute_calibration_metrics(predictions, targets, CALIBRATION_BINS),
            'threshold_sweep': compute_threshold_sweep(predictions, targets)
        }
    return analysis


def write_query_level_csv(path, query_records, predictions_by_model, targets,
                          episode_classes_by_record, class_names_by_id):
    targets_np = targets.numpy() if torch.is_tensor(targets) else targets
    prediction_arrays = {
        name: (value.numpy() if torch.is_tensor(value) else value)
        for name, value in predictions_by_model.items()
    }

    fieldnames = [
        'episode_idx',
        'query_idx_global',
        'query_idx_in_episode',
        'image_id',
        'class_idx',
        'class_id',
        'class_name',
        'target',
        'p_lc_protonets',
        'p_tp_cmc',
        'p_ensemble'
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for global_query_idx, record in enumerate(query_records):
            episode_classes = episode_classes_by_record[global_query_idx]
            for class_idx, class_id in enumerate(episode_classes):
                class_key = str(class_id)
                class_name = class_names_by_id.get(class_key, class_names_by_id.get(class_id, str(class_id)))
                writer.writerow({
                    'episode_idx': record['episode_idx'],
                    'query_idx_global': global_query_idx,
                    'query_idx_in_episode': record['query_idx_in_episode'],
                    'image_id': record['image_id'],
                    'class_idx': class_idx,
                    'class_id': class_id,
                    'class_name': class_name,
                    'target': int(targets_np[global_query_idx, class_idx]),
                    'p_lc_protonets': float(prediction_arrays['lc_protonets'][global_query_idx, class_idx]),
                    'p_tp_cmc': float(prediction_arrays['tp_cmc'][global_query_idx, class_idx]),
                    'p_ensemble': float(prediction_arrays['ensemble'][global_query_idx, class_idx])
                })


def get_class_name(class_id, class_names_by_id):
    """Return a stable display name for a COCO category identifier."""
    try:
        normalized_id = int(class_id)
    except (TypeError, ValueError):
        normalized_id = class_id

    return str(class_names_by_id.get(
        str(normalized_id),
        class_names_by_id.get(normalized_id, str(normalized_id))
    ))


def format_label_set(label_ids, class_names_by_id, separator=', '):
    names = [get_class_name(class_id, class_names_by_id) for class_id in label_ids]
    return separator.join(names) if names else 'None'


def threshold_predictions(probabilities, threshold=FIXED_THRESHOLD):
    return np.asarray(probabilities, dtype=np.float64) > float(threshold)


def compute_query_decision_metrics(prediction_mask, target_mask):
    """Compute sample-level multilabel metrics under the fixed decision rule."""
    predicted = np.asarray(prediction_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError(
            f'Prediction and target shapes differ: {predicted.shape} vs {target.shape}'
        )

    tp = int(np.logical_and(predicted, target).sum())
    fp = int(np.logical_and(predicted, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(predicted), target).sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float((2 * tp) / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def has_exact_label_set_match(metrics):
    """Return whether a fixed-threshold prediction has no label-set errors."""
    return bool(metrics['fp'] == 0 and metrics['fn'] == 0)


def _as_numpy_array(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _ids_from_mask(class_ids, mask):
    return [int(class_ids[index]) for index, value in enumerate(mask) if bool(value)]


def build_case_candidates(predictions_by_model, targets, query_records,
                          episode_classes_by_record, threshold=FIXED_THRESHOLD):
    """Build an auditable, image-deduplicated pool for deterministic selection."""
    target_array = _as_numpy_array(targets).astype(np.int32)
    prediction_arrays = {
        model_name: _as_numpy_array(predictions_by_model[model_name]).astype(np.float64)
        for model_name in CASE_STUDY_MODEL_ORDER
    }

    n_records = len(query_records)
    if target_array.shape[0] != n_records:
        raise ValueError('Number of query records does not match target rows.')
    if len(episode_classes_by_record) != n_records:
        raise ValueError('Number of episode-class records does not match query records.')
    for model_name, probabilities in prediction_arrays.items():
        if probabilities.shape != target_array.shape:
            raise ValueError(
                f'{model_name} probabilities have shape {probabilities.shape}, '
                f'expected {target_array.shape}.'
            )

    candidates = []
    seen_episode_images = set()
    duplicate_record_count = 0
    epsilon = 1e-12

    for record_index, record in enumerate(query_records):
        episode_idx = int(record['episode_idx'])
        image_id = int(record['image_id'])
        deduplication_key = (episode_idx, image_id)
        if deduplication_key in seen_episode_images:
            duplicate_record_count += 1
            continue
        seen_episode_images.add(deduplication_key)

        class_ids = [int(class_id) for class_id in episode_classes_by_record[record_index]]
        target_mask = target_array[record_index].astype(bool)
        if len(class_ids) != target_mask.shape[0]:
            raise ValueError(
                f'Episode {episode_idx}, query {record_index} has {len(class_ids)} classes '
                f'but {target_mask.shape[0]} target entries.'
            )

        model_predictions = {}
        model_metrics = {}
        false_positive_ids = {}
        false_negative_ids = {}
        probability_rows = {}
        for model_name in CASE_STUDY_MODEL_ORDER:
            probability_row = prediction_arrays[model_name][record_index]
            prediction_mask = threshold_predictions(probability_row, threshold)
            model_predictions[model_name] = _ids_from_mask(class_ids, prediction_mask)
            model_metrics[model_name] = compute_query_decision_metrics(
                prediction_mask, target_mask
            )
            false_positive_ids[model_name] = _ids_from_mask(
                class_ids, np.logical_and(prediction_mask, np.logical_not(target_mask))
            )
            false_negative_ids[model_name] = _ids_from_mask(
                class_ids, np.logical_and(np.logical_not(prediction_mask), target_mask)
            )
            probability_rows[model_name] = [float(value) for value in probability_row]

        text_prediction_mask = threshold_predictions(
            prediction_arrays['tp_cmc'][record_index], threshold
        )
        visual_prediction_mask = threshold_predictions(
            prediction_arrays['lc_protonets'][record_index], threshold
        )
        fused_prediction_mask = threshold_predictions(
            prediction_arrays['ensemble'][record_index], threshold
        )
        visual_corrections = []
        for class_index, class_id in enumerate(class_ids):
            if (
                not target_mask[class_index]
                and text_prediction_mask[class_index]
                and not visual_prediction_mask[class_index]
                and not fused_prediction_mask[class_index]
            ):
                visual_corrections.append({
                    'class_id': int(class_id),
                    'p_tp_cmc': float(prediction_arrays['tp_cmc'][record_index, class_index]),
                    'p_lc_protonets': float(
                        prediction_arrays['lc_protonets'][record_index, class_index]
                    ),
                    'p_depf': float(prediction_arrays['ensemble'][record_index, class_index])
                })

        text_metrics = model_metrics['tp_cmc']
        visual_metrics = model_metrics['lc_protonets']
        depf_metrics = model_metrics['ensemble']
        exact_label_set_match = {
            model_name: has_exact_label_set_match(metrics)
            for model_name, metrics in model_metrics.items()
        }
        delta_fp = int(text_metrics['fp'] - depf_metrics['fp'])
        delta_f1 = float(depf_metrics['f1'] - text_metrics['f1'])
        eligibility = {
            # DEPF must be exact; both original branches must make a label error.
            'success': bool(
                exact_label_set_match['ensemble']
                and (
                    not exact_label_set_match['tp_cmc']
                    and not exact_label_set_match['lc_protonets']
                )
            ),
            'visual_correction': bool(visual_corrections),
            'failure': bool(
                visual_metrics['f1'] < 1.0 - epsilon
                and text_metrics['f1'] < 1.0 - epsilon
                and depf_metrics['f1'] < 1.0 - epsilon
                and depf_metrics['f1'] <= max(visual_metrics['f1'], text_metrics['f1']) + epsilon
            )
        }

        candidates.append({
            'record_index': int(record_index),
            'episode_idx': episode_idx,
            'query_idx_in_episode': int(record['query_idx_in_episode']),
            'sample_index': int(record.get('sample_index', -1)),
            'image_id': image_id,
            'filename': str(record.get('filename', '')),
            'episode_class_ids': class_ids,
            'target_vector': [int(value) for value in target_mask.astype(np.int32)],
            'ground_truth_ids': _ids_from_mask(class_ids, target_mask),
            'true_label_count': int(target_mask.sum()),
            'probabilities': probability_rows,
            'predicted_label_ids': model_predictions,
            'false_positive_ids': false_positive_ids,
            'false_negative_ids': false_negative_ids,
            'model_metrics': model_metrics,
            'exact_label_set_match': exact_label_set_match,
            'delta_fp_tp_cmc_to_depf': delta_fp,
            'delta_f1_tp_cmc_to_depf': delta_f1,
            'visual_corrections': visual_corrections,
            'eligibility': eligibility
        })

    audit = {
        'raw_query_records': int(n_records),
        'unique_episode_query_images': int(len(candidates)),
        'duplicate_episode_query_records_excluded': int(duplicate_record_count)
    }
    return candidates, audit


def _candidate_rank_key(candidate, case_type):
    text_metrics = candidate['model_metrics']['tp_cmc']
    visual_metrics = candidate['model_metrics']['lc_protonets']
    depf_metrics = candidate['model_metrics']['ensemble']
    stable_tie_break = (
        candidate['episode_idx'],
        candidate['query_idx_in_episode'],
        candidate['image_id']
    )
    label_preference = (
        -int(len(candidate['ground_truth_ids']) >= CASE_STUDY_PREFERRED_MIN_TRUE_LABELS),
        -len(candidate['ground_truth_ids']),
    )

    if case_type == 'success':
        return (
            *label_preference,
            -candidate['delta_fp_tp_cmc_to_depf'],
            -candidate['delta_f1_tp_cmc_to_depf'],
            -text_metrics['fp'],
            *stable_tie_break
        )
    if case_type == 'visual_correction':
        return (
            *label_preference,
            -len(candidate['visual_corrections']),
            -candidate['delta_fp_tp_cmc_to_depf'],
            -candidate['delta_f1_tp_cmc_to_depf'],
            *stable_tie_break
        )
    if case_type == 'failure':
        return (
            *label_preference,
            depf_metrics['f1'],
            -(depf_metrics['fp'] + depf_metrics['fn']),
            -max(text_metrics['f1'], visual_metrics['f1']),
            *stable_tie_break
        )
    raise ValueError(f'Unknown case type: {case_type}')


def select_case_representatives(candidates, cases_per_type=CASE_STUDY_CASES_PER_TYPE,
                                enforce_distinct_episodes=CASE_STUDY_ENFORCE_DISTINCT_EPISODES):
    """Select predefined strata without inspecting images or manually editing ranks."""
    selected_cases = []
    selection_log = []
    used_episode_indices = set()
    used_record_indices = set()
    used_image_ids = set()

    for case_type in ('success', 'visual_correction', 'failure'):
        ranked_candidates = sorted(
            [candidate for candidate in candidates if candidate['eligibility'][case_type]],
            key=lambda candidate: _candidate_rank_key(candidate, case_type)
        )
        selected_for_type = 0
        fallback_used = False

        for case_number in range(cases_per_type):
            available = [
                candidate for candidate in ranked_candidates
                if candidate['record_index'] not in used_record_indices
                and candidate['image_id'] not in used_image_ids
            ]
            # More true labels take precedence over the soft episode preference.
            if available:
                best_label_count = len(available[0]['ground_truth_ids'])
                preferred = [candidate for candidate in available
                             if len(candidate['ground_truth_ids']) == best_label_count]
            else:
                preferred = []
            distinct_episode_candidates = [
                candidate for candidate in preferred
                if candidate['episode_idx'] not in used_episode_indices
            ]
            if enforce_distinct_episodes and distinct_episode_candidates:
                candidate = distinct_episode_candidates[0]
            elif available:
                candidate = available[0]
                fallback_used = fallback_used or enforce_distinct_episodes
            else:
                break

            rank_lookup = {
                ranked_candidate['record_index']: rank
                for rank, ranked_candidate in enumerate(ranked_candidates, start=1)
            }
            selected_case = dict(candidate)
            selected_case.update({
                'case_id': f'{case_type}_{case_number + 1}',
                'case_type': case_type,
                'case_type_display': CASE_STUDY_TYPE_NAMES[case_type],
                'selection_rank_within_stratum': int(rank_lookup[candidate['record_index']]),
                'distinct_episode_fallback_used': bool(
                    enforce_distinct_episodes
                    and candidate['episode_idx'] in used_episode_indices
                ),
                'preferred_min_true_labels': int(CASE_STUDY_PREFERRED_MIN_TRUE_LABELS),
                'low_label_count_fallback_used': bool(
                    len(candidate['ground_truth_ids']) < CASE_STUDY_PREFERRED_MIN_TRUE_LABELS
                ),
            })
            selected_cases.append(selected_case)
            selected_for_type += 1
            used_episode_indices.add(candidate['episode_idx'])
            used_record_indices.add(candidate['record_index'])
            used_image_ids.add(candidate['image_id'])

        selection_log.append({
            'case_type': case_type,
            'eligible_candidate_count': int(len(ranked_candidates)),
            'eligible_with_preferred_label_count': int(sum(
                len(candidate['ground_truth_ids']) >= CASE_STUDY_PREFERRED_MIN_TRUE_LABELS
                for candidate in ranked_candidates
            )),
            'requested_case_count': int(cases_per_type),
            'selected_case_count': int(selected_for_type),
            'distinct_episode_fallback_used': bool(fallback_used)
        })

    return selected_cases, selection_log


def summarize_case_screening(candidates, candidate_audit):
    total = len(candidates)

    def count_and_rate(predicate):
        count = sum(1 for candidate in candidates if predicate(candidate))
        return {
            'count': int(count),
            'fraction': float(count / total) if total else 0.0
        }

    summary = dict(candidate_audit)
    summary.update({
        'screened_unique_queries': int(total),
        'fusion_reduces_fp_vs_tp_cmc': count_and_rate(
            lambda candidate: candidate['delta_fp_tp_cmc_to_depf'] > 0
        ),
        'fusion_improves_sample_f1_vs_tp_cmc': count_and_rate(
            lambda candidate: candidate['delta_f1_tp_cmc_to_depf'] > 1e-12
        ),
        'fusion_lowers_recall_vs_tp_cmc': count_and_rate(
            lambda candidate: (
                candidate['model_metrics']['ensemble']['recall']
                < candidate['model_metrics']['tp_cmc']['recall'] - 1e-12
            )
        ),
        'fusion_worsens_sample_f1_vs_tp_cmc': count_and_rate(
            lambda candidate: candidate['delta_f1_tp_cmc_to_depf'] < -1e-12
        ),
        'depf_exact_label_set_match': count_and_rate(
            lambda candidate: candidate['exact_label_set_match']['ensemble']
        ),
        'both_branches_not_exact_label_set_match': count_and_rate(
            lambda candidate: (
                not candidate['exact_label_set_match']['tp_cmc']
                and not candidate['exact_label_set_match']['lc_protonets']
            )
        ),
        'success_eligible': count_and_rate(
            lambda candidate: candidate['eligibility']['success']
        ),
        'success_eligible_with_at_least_3_true_labels': count_and_rate(
            lambda candidate: candidate['eligibility']['success']
            and len(candidate['ground_truth_ids']) >= 3
        ),
        'visual_correction_eligible': count_and_rate(
            lambda candidate: candidate['eligibility']['visual_correction']
        ),
        'failure_eligible': count_and_rate(
            lambda candidate: candidate['eligibility']['failure']
        )
    })
    return summary


def write_episode_manifest(path, episode_records, class_names_by_id):
    """Persist every support and query image used by the analyzed episodes."""
    serialized_episodes = []
    for episode in episode_records:
        def serialize_image_record(record):
            label_ids = [int(class_id) for class_id in record['label_ids']]
            return {
                **record,
                'label_names': [
                    get_class_name(class_id, class_names_by_id) for class_id in label_ids
                ]
            }

        episode_class_ids = [int(class_id) for class_id in episode['episode_classes']]
        serialized_episodes.append({
            'episode_idx': int(episode['episode_idx']),
            'episode_class_ids': episode_class_ids,
            'episode_class_names': [
                get_class_name(class_id, class_names_by_id) for class_id in episode_class_ids
            ],
            'support_records': [
                serialize_image_record(record) for record in episode['support_records']
            ],
            'query_records': [
                serialize_image_record(record) for record in episode['query_records']
            ]
        })

    with open(path, 'w', encoding='utf-8') as file_handle:
        json.dump({
            'description': (
                'Complete support/query manifest for the fixed evaluation episodes used '
                'by the case study.'
            ),
            'protocol': (EPISODE_REPLAY_AUDIT or {}).get('protocol', 'random_episodic'),
            'protocol_note': (EPISODE_REPLAY_AUDIT or {}).get('protocol_note'),
            'episode_replay': EPISODE_REPLAY_AUDIT,
            'episodes': serialized_episodes
        }, file_handle, indent=2, ensure_ascii=False)


def _case_selection_manifest_entry(case, class_names_by_id):
    entry = dict(case)
    entry.update({
        'episode_class_names': [
            get_class_name(class_id, class_names_by_id)
            for class_id in case['episode_class_ids']
        ],
        'ground_truth_names': [
            get_class_name(class_id, class_names_by_id)
            for class_id in case['ground_truth_ids']
        ],
        'predicted_label_names': {
            model_name: [
                get_class_name(class_id, class_names_by_id)
                for class_id in case['predicted_label_ids'][model_name]
            ]
            for model_name in CASE_STUDY_MODEL_ORDER
        },
        'false_positive_names': {
            model_name: [
                get_class_name(class_id, class_names_by_id)
                for class_id in case['false_positive_ids'][model_name]
            ]
            for model_name in CASE_STUDY_MODEL_ORDER
        },
        'false_negative_names': {
            model_name: [
                get_class_name(class_id, class_names_by_id)
                for class_id in case['false_negative_ids'][model_name]
            ]
            for model_name in CASE_STUDY_MODEL_ORDER
        },
        'visual_correction_labels': [
            {
                **correction,
                'class_name': get_class_name(correction['class_id'], class_names_by_id)
            }
            for correction in case['visual_corrections']
        ]
    })
    return entry


def write_case_screening_csv(path, candidates, selected_cases, class_names_by_id):
    selected_by_record = {
        case['record_index']: case for case in selected_cases
    }
    fieldnames = [
        'episode_idx', 'query_idx_global', 'query_idx_in_episode', 'image_id',
        'filename', 'selected_case_id', 'selected_case_type',
        'selection_rank_within_stratum', 'ground_truth_labels', 'true_label_count',
        'tp_cmc_prediction', 'lc_protonets_prediction', 'depf_prediction',
        'tp_cmc_tp', 'tp_cmc_fp', 'tp_cmc_fn', 'tp_cmc_precision',
        'tp_cmc_recall', 'tp_cmc_f1', 'tp_cmc_exact_match',
        'lc_protonets_tp', 'lc_protonets_fp',
        'lc_protonets_fn', 'lc_protonets_precision', 'lc_protonets_recall',
        'lc_protonets_f1', 'lc_protonets_exact_match',
        'depf_tp', 'depf_fp', 'depf_fn', 'depf_precision', 'depf_recall',
        'depf_f1', 'depf_exact_match',
        'delta_fp_tp_cmc_to_depf', 'delta_f1_tp_cmc_to_depf',
        'eligible_success', 'eligible_visual_correction', 'eligible_failure',
        'visual_correction_labels'
    ]

    with open(path, 'w', newline='', encoding='utf-8') as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            selected_case = selected_by_record.get(candidate['record_index'])
            row = {
                'episode_idx': candidate['episode_idx'],
                'query_idx_global': candidate['record_index'],
                'query_idx_in_episode': candidate['query_idx_in_episode'],
                'image_id': candidate['image_id'],
                'filename': candidate['filename'],
                'selected_case_id': selected_case['case_id'] if selected_case else '',
                'selected_case_type': selected_case['case_type'] if selected_case else '',
                'selection_rank_within_stratum': (
                    selected_case['selection_rank_within_stratum'] if selected_case else ''
                ),
                'ground_truth_labels': format_label_set(
                    candidate['ground_truth_ids'], class_names_by_id
                ),
                'true_label_count': len(candidate['ground_truth_ids']),
                'delta_fp_tp_cmc_to_depf': candidate['delta_fp_tp_cmc_to_depf'],
                'delta_f1_tp_cmc_to_depf': candidate['delta_f1_tp_cmc_to_depf'],
                'eligible_success': int(candidate['eligibility']['success']),
                'eligible_visual_correction': int(
                    candidate['eligibility']['visual_correction']
                ),
                'eligible_failure': int(candidate['eligibility']['failure']),
                'visual_correction_labels': format_label_set(
                    [correction['class_id'] for correction in candidate['visual_corrections']],
                    class_names_by_id
                )
            }
            for model_name in CASE_STUDY_MODEL_ORDER:
                metric = candidate['model_metrics'][model_name]
                csv_prefix = 'depf' if model_name == 'ensemble' else model_name
                row[f'{csv_prefix}_prediction'] = format_label_set(
                    candidate['predicted_label_ids'][model_name], class_names_by_id
                )
                for metric_name in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1'):
                    row[f'{csv_prefix}_{metric_name}'] = metric[metric_name]
                row[f'{csv_prefix}_exact_match'] = int(
                    candidate['exact_label_set_match'][model_name]
                )
            writer.writerow(row)


def build_case_prediction_table_rows(selected_cases, class_names_by_id):
    rows = []
    for case_number, case in enumerate(selected_cases, start=1):
        case_label = (
            f'Case {case_number}: {case["case_type_display"]}\n'
            f'GT: {format_label_set(case["ground_truth_ids"], class_names_by_id)}'
        )
        for model_name in CASE_STUDY_MODEL_ORDER:
            metrics = case['model_metrics'][model_name]
            rows.append({
                'case_id': case['case_id'],
                'case': case_label,
                'case_type': case['case_type'],
                'episode_idx': case['episode_idx'],
                'query_idx_in_episode': case['query_idx_in_episode'],
                'image_id': case['image_id'],
                'model': CASE_STUDY_MODEL_NAMES[model_name],
                'predicted_label_set': format_label_set(
                    case['predicted_label_ids'][model_name], class_names_by_id
                ),
                'false_positive_labels': format_label_set(
                    case['false_positive_ids'][model_name], class_names_by_id
                ),
                'false_negative_labels': format_label_set(
                    case['false_negative_ids'][model_name], class_names_by_id
                ),
                **metrics
            })
    return rows


def write_case_prediction_table_csv(path, rows):
    fieldnames = [
        'case_id', 'case', 'case_type', 'episode_idx', 'query_idx_in_episode',
        'image_id', 'model', 'predicted_label_set', 'false_positive_labels',
        'false_negative_labels', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'
    ]
    with open(path, 'w', newline='', encoding='utf-8') as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_case_study_figure(fig, output_stem, include_svg=True):
    output_paths = []
    png_path = f'{output_stem}.png'
    pdf_path = f'{output_stem}.pdf'
    fig.savefig(png_path, dpi=CASE_STUDY_FIGURE_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    output_paths.extend([png_path, pdf_path])
    if include_svg:
        svg_path = f'{output_stem}.svg'
        fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
        output_paths.append(svg_path)
    plt.close(fig)
    return output_paths


CASE_STUDY_MPL_RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 8,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none'
}


def _plot_class_indices(case, class_names_by_id):
    class_ids = case['episode_class_ids']
    target_mask = np.asarray(case['target_vector'], dtype=bool)
    probability_matrix = np.vstack([
        np.asarray(case['probabilities'][model_name], dtype=np.float64)
        for model_name in CASE_STUDY_MODEL_ORDER
    ])
    prediction_matrix = probability_matrix > FIXED_THRESHOLD
    decision_changes = np.any(
        prediction_matrix != prediction_matrix[0:1, :], axis=0
    )

    ground_truth_indices = set(np.flatnonzero(target_mask).tolist())
    changed_indices = set(np.flatnonzero(decision_changes).tolist())
    essential_indices = ground_truth_indices | changed_indices
    remaining_indices = [
        index for index in range(len(class_ids)) if index not in essential_indices
    ]

    def sort_key(class_index):
        return (
            -float(probability_matrix[:, class_index].max()),
            get_class_name(class_ids[class_index], class_names_by_id).lower(),
            int(class_ids[class_index])
        )

    ordered_essential = sorted(essential_indices, key=sort_key)
    ordered_remaining = sorted(remaining_indices, key=sort_key)
    return ordered_essential + ordered_remaining[:CASE_STUDY_MAX_ADDITIONAL_PLOT_LABELS]


def plot_case_probability_bars(selected_cases, class_names_by_id, output_stem):
    if not selected_cases:
        return []

    with mpl.rc_context(CASE_STUDY_MPL_RCPARAMS):
        panel_heights = [
            max(2.8, 0.45 * len(_plot_class_indices(case, class_names_by_id)) + 1.4)
            for case in selected_cases
        ]
        fig, axes = plt.subplots(
            len(selected_cases), 1,
            figsize=(11.5, sum(panel_heights)),
            squeeze=False
        )
        axes = axes[:, 0]
        legend_handles = []

        for case_number, (ax, case) in enumerate(zip(axes, selected_cases), start=1):
            class_indices = _plot_class_indices(case, class_names_by_id)
            class_ids = case['episode_class_ids']
            target_mask = np.asarray(case['target_vector'], dtype=bool)
            y_positions = np.arange(len(class_indices))
            bar_height = 0.23

            for model_offset, model_name in enumerate(CASE_STUDY_MODEL_ORDER):
                offset = (model_offset - 1) * bar_height
                probabilities = np.asarray(case['probabilities'][model_name])[class_indices]
                bars = ax.barh(
                    y_positions + offset,
                    probabilities,
                    height=bar_height,
                    color=CASE_STUDY_MODEL_COLORS[model_name],
                    label=CASE_STUDY_MODEL_NAMES[model_name]
                )
                if case_number == 1:
                    legend_handles.append(bars)

            labels = [
                f'* {get_class_name(class_ids[class_index], class_names_by_id)}'
                if target_mask[class_index]
                else get_class_name(class_ids[class_index], class_names_by_id)
                for class_index in class_indices
            ]
            ax.axvline(FIXED_THRESHOLD, color='black', linestyle='--', linewidth=1.0)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels)
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel('Predicted probability')
            ax.set_title(
                f'Case {case_number}: {case["case_type_display"]} '
                f'(episode {case["episode_idx"]}, query {case["query_idx_in_episode"]})',
                loc='left', fontweight='bold'
            )
            ax.grid(axis='x', alpha=0.25, linewidth=0.6)
            ax.invert_yaxis()

        fig.legend(
            legend_handles,
            [CASE_STUDY_MODEL_NAMES[model_name] for model_name in CASE_STUDY_MODEL_ORDER],
            loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.01)
        )
        fig.text(
            0.01, 0.005,
            '* Ground-truth class. Each panel includes all ground-truth labels and all '
            'classes whose fixed-threshold decision differs across the three models.',
            fontsize=7
        )
        fig.tight_layout(rect=(0.03, 0.035, 1.0, 0.965))
        return _save_case_study_figure(fig, output_stem)


def render_case_prediction_table(rows, output_stem):
    if not rows:
        return []

    headers = [
        'Case', 'Model', 'Predicted label set', 'TP', 'FP', 'FN',
        'Precision', 'Recall', 'F1'
    ]
    cell_text = []
    cell_colours = []
    for row in rows:
        cell_text.append([
            fill(row['case'], width=25),
            row['model'],
            fill(row['predicted_label_set'], width=36),
            str(row['tp']), str(row['fp']), str(row['fn']),
            f"{row['precision']:.2f}", f"{row['recall']:.2f}", f"{row['f1']:.2f}"
        ])
        row_color = '#EAF4EA' if row['model'] == 'DEPF' else '#FFFFFF'
        cell_colours.append([row_color] * len(headers))

    with mpl.rc_context(CASE_STUDY_MPL_RCPARAMS):
        fig_height = max(4.5, 0.76 * len(rows) + 1.5)
        fig, ax = plt.subplots(figsize=(15.5, fig_height))
        ax.axis('off')
        table = ax.table(
            cellText=cell_text,
            cellColours=cell_colours,
            colLabels=headers,
            colColours=['#E6E6E6'] * len(headers),
            cellLoc='left',
            colLoc='center',
            colWidths=[0.21, 0.10, 0.34, 0.05, 0.05, 0.05, 0.07, 0.06, 0.05],
            bbox=[0.0, 0.0, 1.0, 0.92]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.7)
        for (row_index, column_index), cell in table.get_celld().items():
            cell.set_linewidth(0.35)
            cell.set_edgecolor('#B7B7B7')
            if row_index == 0:
                cell.set_text_props(weight='bold', ha='center')
            elif column_index >= 3:
                cell.set_text_props(ha='center')

        ax.set_title(
            'Fixed-threshold query-level prediction comparison',
            loc='left', fontweight='bold', pad=12
        )
        fig.tight_layout()
        return _save_case_study_figure(fig, output_stem)


def _select_support_prototype_exemplars(episode_record, max_prototypes):
    grouped_supports = defaultdict(list)
    for support_record in episode_record['support_records']:
        label_combination = tuple(sorted(int(label_id) for label_id in support_record['label_ids']))
        grouped_supports[label_combination].append(support_record)

    grouped_items = sorted(
        grouped_supports.items(),
        key=lambda item: (
            -len(item[1]),
            item[0],
            int(item[1][0]['image_id'])
        )
    )
    return [
        {
            'label_ids': list(label_combination),
            'count': int(len(records)),
            'record': records[0]
        }
        for label_combination, records in grouped_items[:max_prototypes]
    ]


def _load_original_image(dataset, sample_index):
    sample = dataset.samples[int(sample_index)]
    image_path = dataset._get_image_path(sample)
    with Image.open(image_path) as image:
        return np.asarray(image.convert('RGB'))


def _copy_case_source_image(dataset, record, destination_directory,
                            role, ordinal=None):
    """Export a metadata-free PNG and a manifest entry without local source paths."""
    sample_index = int(record['sample_index'])
    sample = dataset.samples[sample_index]
    image_id = int(record.get('image_id', sample.get('image_id', sample_index)))
    source_path = None
    path_error = None
    try:
        source_path = Path(dataset._get_image_path(sample))
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as error:
        path_error = error
    if role == 'support':
        filename_prefix = f"support_{int(ordinal):02d}"
    else:
        filename_prefix = 'query'
    destination_path = Path(destination_directory) / (
        f"{filename_prefix}_image_{image_id:012d}.png"
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        'role': role,
        'support_ordinal': int(ordinal) if ordinal is not None else None,
        'image_id': image_id,
        'sample_index': sample_index,
        'destination_path': os.path.relpath(
            str(destination_path), str(Path(destination_directory).parents[1])
        ).replace(os.sep, '/'),
        'status': 'copied'
    }
    try:
        if path_error is not None:
            raise path_error
        with Image.open(source_path) as source_image:
            # A fresh pixel buffer excludes EXIF/GPS, comments and other metadata.
            pixels = source_image.convert('RGB')
            clean_image = Image.new('RGB', pixels.size)
            clean_image.paste(pixels)
            clean_image.save(destination_path, format='PNG')
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as error:
        entry['status'] = 'missing_or_unreadable'
        entry['error_type'] = type(error).__name__
    return entry


def render_case_support_query_panels(selected_cases, episode_records, dataset,
                                     class_names_by_id, output_directory):
    if not selected_cases:
        return []

    episode_by_index = {
        int(episode['episode_idx']): episode for episode in episode_records
    }
    output_paths = []
    source_manifest_cases = []
    with mpl.rc_context(CASE_STUDY_MPL_RCPARAMS):
        for case_number, case in enumerate(selected_cases, start=1):
            episode_record = episode_by_index.get(case['episode_idx'])
            if episode_record is None:
                raise ValueError(f"Missing support manifest for episode {case['episode_idx']}.")

            support_exemplars = _select_support_prototype_exemplars(
                episode_record, CASE_STUDY_MAX_SUPPORT_PROTOTYPES
            )
            case_source_entries = []
            case_source_directory = os.path.join(
                output_directory,
                CASE_STUDY_SOURCE_IMAGE_DIRNAME,
                f"case_{case_number:02d}_{case['case_type']}"
            )
            if CASE_STUDY_EXPORT_SOURCE_IMAGES:
                for exemplar_index, exemplar in enumerate(support_exemplars, start=1):
                    case_source_entries.append(_copy_case_source_image(
                        dataset,
                        exemplar['record'],
                        case_source_directory,
                        role='support',
                        ordinal=exemplar_index
                    ))
            support_count = max(1, len(support_exemplars))
            support_columns = min(CASE_STUDY_SUPPORT_COLUMNS, support_count)
            support_rows = int(math.ceil(support_count / support_columns))
            fig = plt.figure(
                figsize=(support_columns * 2.35 + 5.7, max(5.4, support_rows * 2.45 + 1.4))
            )
            grid = fig.add_gridspec(
                support_rows, support_columns + 2,
                width_ratios=[1.0] * support_columns + [0.08, 2.25],
                wspace=0.16, hspace=0.5
            )

            for exemplar_index, exemplar in enumerate(support_exemplars):
                row_index = exemplar_index // support_columns
                column_index = exemplar_index % support_columns
                axis = fig.add_subplot(grid[row_index, column_index])
                try:
                    axis.imshow(_load_original_image(dataset, exemplar['record']['sample_index']))
                except (FileNotFoundError, OSError):
                    axis.text(0.5, 0.5, 'Image unavailable', ha='center', va='center')
                axis.set_axis_off()
                label_text = fill(
                    format_label_set(exemplar['label_ids'], class_names_by_id), width=20
                )
                axis.set_title(
                    f"prototype n={exemplar['count']}\n{label_text}",
                    fontsize=6.2, loc='left', pad=2
                )

            query_record = next(
                (
                    record for record in episode_record['query_records']
                    if record['query_idx_in_episode'] == case['query_idx_in_episode']
                    and record['image_id'] == case['image_id']
                ),
                None
            )
            if query_record is None:
                raise ValueError(
                    f"Missing query image {case['image_id']} in episode {case['episode_idx']}."
                )
            if CASE_STUDY_EXPORT_SOURCE_IMAGES:
                case_source_entries.append(_copy_case_source_image(
                    dataset,
                    query_record,
                    case_source_directory,
                    role='query'
                ))
            query_axis = fig.add_subplot(grid[:, support_columns + 1])
            try:
                query_axis.imshow(_load_original_image(dataset, query_record['sample_index']))
            except (FileNotFoundError, OSError):
                query_axis.text(0.5, 0.5, 'Image unavailable', ha='center', va='center')
            query_axis.set_axis_off()
            query_axis.set_title(
                'Query\n'
                f"GT: {fill(format_label_set(case['ground_truth_ids'], class_names_by_id), width=27)}\n"
                f"episode {case['episode_idx']}, query {case['query_idx_in_episode']}",
                fontsize=8, loc='left', pad=5
            )

            fig.suptitle(
                f"Case {case_number}: {case['case_type_display']}",
                x=0.01, ha='left', fontweight='bold', fontsize=11
            )
            fig.text(
                0.01, 0.01,
                'Left: one deterministic exemplar per support label-combination prototype; '
                'n denotes the number of support images forming that prototype. '
                'The complete support set is retained in the episode manifest.',
                fontsize=6.5
            )
            output_stem = os.path.join(
                output_directory,
                f"case_{case_number:02d}_{case['case_type']}_support_query"
            )
            output_paths.extend(_save_case_study_figure(fig, output_stem, include_svg=False))
            if CASE_STUDY_EXPORT_SOURCE_IMAGES:
                source_manifest_cases.append({
                    'case_number': int(case_number),
                    'case_id': case['case_id'],
                    'case_type': case['case_type'],
                    'case_type_display': case['case_type_display'],
                    'episode_idx': int(case['episode_idx']),
                    'query_idx_in_episode': int(case['query_idx_in_episode']),
                    'images': case_source_entries
                })

    if CASE_STUDY_EXPORT_SOURCE_IMAGES:
        source_manifest_path = os.path.join(
            output_directory, CASE_STUDY_SOURCE_MANIFEST_FILENAME
        )
        source_manifest = {
            'source_directory': CASE_STUDY_SOURCE_IMAGE_DIRNAME,
            'selection_scope': (
                'Only the support prototype exemplars and query images displayed in '
                'the case-study support/query figures are exported.'
            ),
            'cases': source_manifest_cases
        }
        with open(source_manifest_path, 'w', encoding='utf-8') as file_handle:
            json.dump(source_manifest, file_handle, indent=2, ensure_ascii=False)
        output_paths.append(source_manifest_path)

    return output_paths


def generate_case_study_outputs(predictions_by_model, targets, query_records,
                                episode_classes_by_record, episode_records, dataset,
                                class_names_by_id, output_directory):
    """Create reproducible screening records and manuscript-ready case-study assets."""
    os.makedirs(output_directory, exist_ok=True)
    candidates, candidate_audit = build_case_candidates(
        predictions_by_model=predictions_by_model,
        targets=targets,
        query_records=query_records,
        episode_classes_by_record=episode_classes_by_record,
        threshold=FIXED_THRESHOLD
    )
    selected_cases, selection_log = select_case_representatives(candidates)
    screening_summary = summarize_case_screening(candidates, candidate_audit)

    episode_manifest_path = os.path.join(output_directory, 'episode_manifest.json')
    screening_csv_path = os.path.join(output_directory, 'case_screening_audit.csv')
    write_episode_manifest(episode_manifest_path, episode_records, class_names_by_id)
    write_case_screening_csv(
        screening_csv_path, candidates, selected_cases, class_names_by_id
    )

    output_paths = [episode_manifest_path, screening_csv_path]
    prediction_rows = build_case_prediction_table_rows(selected_cases, class_names_by_id)
    if selected_cases:
        prediction_csv_path = os.path.join(output_directory, 'case_prediction_table.csv')
        write_case_prediction_table_csv(prediction_csv_path, prediction_rows)
        output_paths.append(prediction_csv_path)
        output_paths.extend(plot_case_probability_bars(
            selected_cases,
            class_names_by_id,
            os.path.join(output_directory, 'case_probability_bars')
        ))
        output_paths.extend(render_case_prediction_table(
            prediction_rows,
            os.path.join(output_directory, 'case_prediction_table')
        ))
        output_paths.extend(render_case_support_query_panels(
            selected_cases,
            episode_records,
            dataset,
            class_names_by_id,
            output_directory
        ))

    manifest = {
        'protocol': {
            'comparison_models': [
                'TP-CMC', 'LC-Protonets', 'DEPF (full model)'
            ],
            'external_baselines': (
                'Not included because same-episode prediction probabilities are unavailable.'
            ),
            'analysis_alpha': float(ANALYSIS_ALPHA),
            'fusion_formula': (
                f'p_DEPF = {ANALYSIS_ALPHA:.1f} * p_LC-Protonets + '
                f'{1.0 - ANALYSIS_ALPHA:.1f} * p_TP-CMC'
            ),
            'fixed_threshold_rule': f'p > {FIXED_THRESHOLD:.1f}',
            'evaluation_configuration': {
                'n_episodes': int(N_EPISODES),
                'episode_replay': EPISODE_REPLAY_AUDIT,
                'n_way': int(N_WAY),
                'k_shot': int(K_SHOT),
                'n_query_per_class': int(N_QUERY),
                'episode_seed': int(EPISODE_SEED),
                'comc_alpha': float(COMC_ALPHA),
                'comc_temperature': float(COMC_TEMPERATURE),
                'clip_model': CLIP_MODEL,
                'lc_protonets_checkpoint': os.path.basename(LC_PROTONETS_CKPT),
                'tp_cmc_checkpoint': os.path.basename(COMC_CLASSIFIER_CKPT)
            },
            'candidate_pool': (
                'All unique query images from the fixed evaluation episodes; duplicate '
                'image occurrences within an episode are removed before case selection.'
            ),
            'selection_rules': {
                'success': (
                    'DEPF has an exact fixed-threshold label-set match (FP=0 and FN=0), '
                    'while both TP-CMC and LC-Protonets each have an FP or FN. '
                    'The final aligned four-model comparison additionally requires MMCI to be incorrect.'
                ),
                'visual_correction': (
                    'At least one negative class has TP-CMC above the threshold while '
                    'LC-Protonets and DEPF are at or below it.'
                ),
                'failure': (
                    'All three models have sample-level F1 < 1 and DEPF does not exceed '
                    'the better branch on sample-level F1.'
                )
            },
            'ranking_rules': {
                'success': 'Prefer >=3 true labels, then more true labels, then FP reduction, F1 gain and stable IDs.',
                'visual_correction': (
                    'Prefer >=3 true labels and more labels, then corrected classes, FP reduction, F1 gain and stable IDs.'
                ),
                'failure': 'Prefer >=3 true labels and more labels, then lowest DEPF F1, most errors and stable IDs.'
            },
            'preferred_min_true_labels': int(CASE_STUDY_PREFERRED_MIN_TRUE_LABELS),
            'low_label_count_fallback': 'Allowed only within eligible cases; recorded per selected case.',
            'unique_query_image_preference': 'A query image is selected at most once across case types.',
            'distinct_episode_preference': bool(CASE_STUDY_ENFORCE_DISTINCT_EPISODES),
            'support_display': (
                'One deterministic exemplar per support label-combination prototype; '
                'the full support set remains in episode_manifest.json.'
            ),
            'probability_bar_rule': (
                'Include every ground-truth class and every class with a cross-model '
                'fixed-threshold decision change, then add the highest-scoring remaining classes.'
            ),
            'source_image_export': {
                'enabled': bool(CASE_STUDY_EXPORT_SOURCE_IMAGES),
                'directory': CASE_STUDY_SOURCE_IMAGE_DIRNAME,
                'manifest': CASE_STUDY_SOURCE_MANIFEST_FILENAME,
                'scope': 'Only images displayed in support/query case figures.'
            }
        },
        'screening_summary': screening_summary,
        'selection_log': selection_log,
        'selected_cases': [
            _case_selection_manifest_entry(case, class_names_by_id)
            for case in selected_cases
        ],
        'generated_files': [os.path.basename(path) for path in output_paths]
    }
    manifest_path = os.path.join(output_directory, 'case_selection_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as file_handle:
        json.dump(manifest, file_handle, indent=2, ensure_ascii=False)
    output_paths.append(manifest_path)

    print(
        f"Case study screened {screening_summary['screened_unique_queries']} unique "
        f"episode-query images and selected {len(selected_cases)} cases."
    )
    for log_entry in selection_log:
        print(
            f"  {log_entry['case_type']}: {log_entry['selected_case_count']}/"
            f"{log_entry['requested_case_count']} selected from "
            f"{log_entry['eligible_candidate_count']} eligible candidates"
        )

    return {
        'output_directory': os.path.basename(output_directory),
        'selected_case_count': int(len(selected_cases)),
        'screening_summary': screening_summary,
        'selection_log': selection_log,
        'manifest_file': os.path.basename(manifest_path),
        'selected_case_ids': [case['case_id'] for case in selected_cases]
    }


def plot_probability_histograms(predictions_by_model, targets, output_path):
    targets_np = targets.numpy() if torch.is_tensor(targets) else targets
    targets_flat = np.asarray(targets_np).reshape(-1)
    model_titles = {
        'tp_cmc': 'TP-CMC',
        'lc_protonets': 'LC-Protonets',
        'ensemble': 'Ensemble'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, model_name in zip(axes, ['tp_cmc', 'lc_protonets', 'ensemble']):
        probs = predictions_by_model[model_name]
        probs_np = probs.numpy() if torch.is_tensor(probs) else probs
        probs_flat = np.asarray(probs_np).reshape(-1)
        pos_probs = probs_flat[targets_flat == 1]
        neg_probs = probs_flat[targets_flat == 0]

        ax.hist(neg_probs, bins=40, range=(0, 1), alpha=0.55, density=True,
                label='Negative labels', color='#4C78A8')
        ax.hist(pos_probs, bins=40, range=(0, 1), alpha=0.55, density=True,
                label='Positive labels', color='#F58518')
        ax.axvline(FIXED_THRESHOLD, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(model_titles[model_name], fontweight='bold')
        ax.set_xlabel('Predicted probability')
        ax.grid(alpha=0.25)

    axes[0].set_ylabel('Density')
    axes[-1].legend(loc='upper center', bbox_to_anchor=(-0.75, -0.18), ncol=2)
    fig.suptitle('Query-level Probability Distributions by Branch', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_threshold_curves(analysis_by_model, output_path):
    model_styles = {
        'tp_cmc': ('TP-CMC', '#4C78A8'),
        'lc_protonets': ('LC-Protonets', '#F58518'),
        'ensemble': ('Ensemble', '#54A24B')
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for model_name, (label, color) in model_styles.items():
        curve = analysis_by_model[model_name]['threshold_sweep']['curve']
        thresholds = [row['threshold'] for row in curve]
        f1_macro = [row['f1_macro'] for row in curve]
        f1_micro = [row['f1_micro'] for row in curve]
        axes[0].plot(thresholds, f1_macro, marker='o', linewidth=2, label=label, color=color)
        axes[1].plot(thresholds, f1_micro, marker='o', linewidth=2, label=label, color=color)

    for ax, title in zip(axes, ['Macro-F1 vs Threshold', 'Micro-F1 vs Threshold']):
        ax.axvline(
            FIXED_THRESHOLD,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'Fixed threshold {FIXED_THRESHOLD:.1f}'
        )
        ax.set_xlabel('Decision threshold')
        ax.set_ylabel('F1')
        ax.set_title(title, fontweight='bold')
        ax.grid(alpha=0.25)
        ax.set_xlim(0.05, 0.95)

    axes[1].legend(loc='lower center', bbox_to_anchor=(-0.1, -0.32), ncol=4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_reliability_diagrams(analysis_by_model, output_path):
    model_titles = {
        'tp_cmc': 'TP-CMC',
        'lc_protonets': 'LC-Protonets',
        'ensemble': 'Ensemble'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, model_name in zip(axes, ['tp_cmc', 'lc_protonets', 'ensemble']):
        calibration = analysis_by_model[model_name]['calibration']
        bins = calibration['bins']
        confidences = [row['avg_confidence'] for row in bins if row['count'] > 0]
        positives = [row['empirical_positive_rate'] for row in bins if row['count'] > 0]

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
        ax.plot(confidences, positives, marker='o', linewidth=2, color='#4C78A8')
        ax.set_title(f"{model_titles[model_name]}\nECE={calibration['ece']:.3f}, Brier={calibration['brier_score']:.3f}",
                     fontweight='bold')
        ax.set_xlabel('Mean predicted probability')
        ax.grid(alpha=0.25)

    axes[0].set_ylabel('Empirical positive rate')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def evaluate_with_alpha(ensemble_model, sampler, novel_cat_ids, n_episodes, device=DEVICE,
                        collect_analysis=False):
    """Evaluate the model with the specified ALPHA."""
    all_p_final = []
    all_p_proto = []
    all_p_comc = []
    all_targets = []
    query_records = []
    episode_classes_by_record = []
    episode_records = []

    episode_iter = tqdm(
        range(n_episodes),
        desc=f"alpha={ensemble_model.alpha:.1f} episodes",
        leave=False,
        dynamic_ncols=True
    )

    for episode_idx in episode_iter:
        episode = sampler.sample_fixed_classes(novel_cat_ids, episode_idx)
        if collect_analysis:
            p_final, individual = ensemble_model.predict(episode, return_individual=True)
        else:
            p_final = ensemble_model.predict(episode, return_individual=False)
            individual = None

        # Build targets.
        n_query = p_final.shape[0]
        n_classes = len(episode['episode_classes'])
        target_tensor = torch.zeros(n_query, n_classes)

        for i, labels in enumerate(episode['query_labels']):
            for label in labels:
                if label in episode['episode_classes']:
                    class_idx = episode['episode_classes'].index(label)
                    target_tensor[i, class_idx] = 1

        all_p_final.append(p_final)
        all_targets.append(target_tensor)

        if collect_analysis:
            all_p_proto.append(individual['p_proto'])
            all_p_comc.append(individual['p_comc'])
            episode_records.append({
                'episode_idx': int(episode_idx),
                'episode_classes': [int(class_id) for class_id in episode['episode_classes']],
                'support_records': [
                    {
                        'support_idx_in_episode': int(support_idx),
                        'sample_index': int(episode['support_sample_indices'][support_idx]),
                        'image_id': int(image_id),
                        'filename': str(episode['support_filenames'][support_idx]),
                        'label_ids': [int(label_id) for label_id in episode['support_labels'][support_idx]]
                    }
                    for support_idx, image_id in enumerate(episode['support_image_ids'])
                ],
                'query_records': [
                    {
                        'query_idx_in_episode': int(query_idx),
                        'sample_index': int(episode['query_sample_indices'][query_idx]),
                        'image_id': int(image_id),
                        'filename': str(episode['query_filenames'][query_idx]),
                        'label_ids': [int(label_id) for label_id in episode['query_labels'][query_idx]]
                    }
                    for query_idx, image_id in enumerate(episode['query_image_ids'])
                ]
            })
            for query_idx, image_id in enumerate(episode['query_image_ids']):
                query_records.append({
                    'episode_idx': int(episode_idx),
                    'query_idx_in_episode': int(query_idx),
                    'sample_index': int(episode['query_sample_indices'][query_idx]),
                    'image_id': int(image_id),
                    'filename': str(episode['query_filenames'][query_idx])
                })
                episode_classes_by_record.append(list(episode['episode_classes']))

    # Concatenate predictions and targets.
    all_p_final = torch.cat(all_p_final, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics.
    metrics = compute_multilabel_metrics(all_p_final, all_targets)
    if not collect_analysis:
        return {'metrics': metrics}

    predictions_by_model = {
        'lc_protonets': torch.cat(all_p_proto, dim=0),
        'tp_cmc': torch.cat(all_p_comc, dim=0),
        'ensemble': all_p_final
    }
    return {
        'metrics': metrics,
        'targets': all_targets,
        'predictions_by_model': predictions_by_model,
        'query_records': query_records,
        'episode_classes_by_record': episode_classes_by_record,
        'episode_records': episode_records
    }


# ==================== Main function ====================

def main():
    global N_EPISODES, EPISODE_REPLAY_AUDIT
    print("\n" + "="*70)
    print("STEP 1: Loading Data and Models")
    print("="*70)

    # Load the dataset split.
    print("Loading dataset split...")
    with open(SPLIT_FILE, 'r') as f:
        data_split = json.load(f)

    with open(MAPPING_FILE, 'r') as f:
        category_mapping = json.load(f)

    cat_name_to_id = category_mapping['cat_name_to_id']
    cat_id_to_name = category_mapping['cat_id_to_name']

    novel_samples = data_split['novel_samples']
    novel_class_names = data_split['metadata']['novel_classes']
    novel_cat_ids = [cat_name_to_id[name] for name in novel_class_names]

    print(f" Dataset loaded!")
    print(f"  Novel classes: {len(novel_class_names)}")

    # Define preprocessing.
    lc_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    _, clip_preprocess = clip.load(CLIP_MODEL, device=DEVICE)

    # Create datasets.
    print("Creating dataset...")
    dataset = DualPreprocessCOCODataset(
        samples=novel_samples,
        train_image_dir=TRAIN2014_DIR,
        val_image_dir=VAL2014_DIR,
        lc_transform=lc_preprocess,
        comc_transform=clip_preprocess,
        cat_id_to_name=cat_id_to_name
    )
    print(f" Dataset created: {len(dataset)} samples")

    sampler = EpisodeSampler(dataset, N_WAY, K_SHOT, N_QUERY, seed=EPISODE_SEED,
                             replay_manifest=EPISODE_MANIFEST, class_ids=novel_cat_ids)
    if sampler.replay_episodes is not None:
        N_EPISODES = len(sampler.replay_episodes)
        EPISODE_REPLAY_AUDIT = sampler.replay_audit
        print("Replaying shared episodes:", json.dumps(EPISODE_REPLAY_AUDIT))
        if EPISODE_REPLAY_AUDIT.get("protocol_note"):
            print(EPISODE_REPLAY_AUDIT["protocol_note"])

    # Load LC-Protonets.
    print("\nLoading LC-Protonets...")
    lc_feature_extractor = FeatureExtractor(
        pretrained=True, feature_dim=2048, finetune=True
    ).to(DEVICE)
    lc_protonets = LCProtonets(
        feature_extractor=lc_feature_extractor,
        distance_metric='euclidean',
        init_bias=14.7
    ).to(DEVICE)
    lc_protonets.load_state_dict(torch.load(LC_PROTONETS_CKPT, map_location=DEVICE))
    lc_protonets.eval()
    print(" LC-Protonets loaded!")

    # Load CLIP and CoMC.
    print("\nLoading CoMC classifier...")
    clip_model, _ = clip.load(CLIP_MODEL, device=DEVICE)
    clip_model.eval()

    comc_checkpoint = torch.load(COMC_CLASSIFIER_CKPT, map_location=DEVICE)
    comc_class_names = comc_checkpoint['class_names']

    if comc_class_names == novel_class_names:
        print(" Class names aligned!")
    else:
        print("WARNING: Class names mismatch!")

    comc_classifier = CoMCClassifier(
        embedding_dim=512, num_classes=len(novel_class_names)
    ).to(DEVICE)
    comc_classifier.load_state_dict(comc_checkpoint['classifier_state_dict'])
    comc_classifier.eval()
    print(" CoMC classifier loaded!")

    # Build class text embeddings.
    print("\nBuilding class text embeddings...")
    TEXT_TEMPLATE = "a photo of a {}"
    class_text_embeddings = build_class_text_embeddings(
        novel_class_names, clip_model, TEXT_TEMPLATE, DEVICE
    )
    print(f" Class text embeddings ready: {class_text_embeddings.shape}")

    # Create the sampler.
    print(f" Episode sampler ready: {N_WAY}-way, {K_SHOT}-shot")
    print(f"  Seed: {EPISODE_SEED} (ensures same episodes for all alphas)")

    print("\n" + "="*70)
    print("STEP 2: Running Alpha Sweep")
    print("="*70)

    # Store results.
    results_by_alpha = {}
    probability_analysis_payload = None
    case_study_summary = None

    # Evaluate each ALPHA value.
    for alpha in tqdm(ALPHA_VALUES, desc="Testing alphas"):
        print(f"\n{'='*70}")
        print(f"Testing ALPHA = {alpha:.1f}")
        print(f"{'='*70}")

        # Create the ensemble model.
        ensemble_model = EnsembleModel(
            lc_protonets=lc_protonets,
            clip_model=clip_model,
            comc_classifier=comc_classifier,
            class_text_embeddings=class_text_embeddings,
            alpha=float(alpha),
            comc_alpha=COMC_ALPHA,
            comc_temperature=COMC_TEMPERATURE,
            device=DEVICE
        )

        # Evaluate.
        collect_analysis = np.isclose(float(alpha), ANALYSIS_ALPHA)
        evaluation = evaluate_with_alpha(
            ensemble_model=ensemble_model,
            sampler=sampler,
            novel_cat_ids=novel_cat_ids,
            n_episodes=N_EPISODES,
            device=DEVICE,
            collect_analysis=collect_analysis
        )
        metrics = evaluation['metrics']

        # Store results.
        results_by_alpha[float(alpha)] = {
            'metrics': to_builtin_float_dict(metrics)
        }

        if collect_analysis:
            print("\nRunning probability sharpness and calibration analysis...")
            predictions_by_model = evaluation['predictions_by_model']
            targets = evaluation['targets']
            probability_analysis = build_probability_analysis(predictions_by_model, targets)

            probability_analysis_payload = {
                'alpha': float(alpha),
                'analysis': probability_analysis,
                'targets': targets,
                'predictions_by_model': predictions_by_model,
                'query_records': evaluation['query_records'],
                'episode_classes_by_record': evaluation['episode_classes_by_record'],
                'episode_records': evaluation['episode_records']
            }

        print(f"\nResults for ALPHA = {alpha:.1f}:")
        print(f"  mAP (macro): {metrics['map_macro']:.4f}")
        print(f"  mAP (micro): {metrics['map_micro']:.4f}")
        print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
        print(f"  F1 (micro):  {metrics['f1_micro']:.4f}")

    print("\n" + "="*70)
    print("STEP 3: Analyzing Results")
    print("="*70)

    # Find the best ALPHA for each individual metric.
    best_map_macro_alpha = max(results_by_alpha.keys(),
                               key=lambda k: results_by_alpha[k]['metrics']['map_macro'])
    best_map_micro_alpha = max(results_by_alpha.keys(),
                               key=lambda k: results_by_alpha[k]['metrics']['map_micro'])
    best_f1_macro_alpha = max(results_by_alpha.keys(),
                              key=lambda k: results_by_alpha[k]['metrics']['f1_macro'])
    best_f1_micro_alpha = max(results_by_alpha.keys(),
                              key=lambda k: results_by_alpha[k]['metrics']['f1_micro'])

    baseline_metrics = results_by_alpha[0.0]['metrics']
    for alpha, result in results_by_alpha.items():
        result['gmean_relative_gain'] = compute_gmean_relative_gain(
            result['metrics'],
            baseline_metrics
        )

    best_gmean_alpha = max(
        results_by_alpha.keys(),
        key=lambda k: results_by_alpha[k]['gmean_relative_gain']
    )
    best_gmean = results_by_alpha[best_gmean_alpha]['gmean_relative_gain']
    best_gmean_metrics = results_by_alpha[best_gmean_alpha]['metrics']

    print(f"\n Best GMean ALPHA: {best_gmean_alpha:.1f}")
    print(f"   GMean Relative Gain: {best_gmean:.4f}")
    print(f"   Metrics: mAP_mac={best_gmean_metrics['map_macro']:.4f}, "
          f"mAP_mic={best_gmean_metrics['map_micro']:.4f}, "
          f"F1_mac={best_gmean_metrics['f1_macro']:.4f}, "
          f"F1_mic={best_gmean_metrics['f1_micro']:.4f}")

    print(f"\n Best ALPHAs for Individual Metrics:")
    print(f"   mAP (macro): {best_map_macro_alpha:.1f} "
          f"(value={results_by_alpha[best_map_macro_alpha]['metrics']['map_macro']:.4f})")
    print(f"   mAP (micro): {best_map_micro_alpha:.1f} "
          f"(value={results_by_alpha[best_map_micro_alpha]['metrics']['map_micro']:.4f})")
    print(f"   F1 (macro):  {best_f1_macro_alpha:.1f} "
          f"(value={results_by_alpha[best_f1_macro_alpha]['metrics']['f1_macro']:.4f})")
    print(f"   F1 (micro):  {best_f1_micro_alpha:.1f} "
          f"(value={results_by_alpha[best_f1_micro_alpha]['metrics']['f1_micro']:.4f})")

    print("\n" + "="*70)
    print("STEP 4: Creating Visualization")
    print("="*70)

    # Create visualizations.
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract data.
    alphas = sorted(results_by_alpha.keys())
    map_macro_values = [results_by_alpha[a]['metrics']['map_macro'] for a in alphas]
    map_micro_values = [results_by_alpha[a]['metrics']['map_micro'] for a in alphas]
    f1_macro_values = [results_by_alpha[a]['metrics']['f1_macro'] for a in alphas]
    f1_micro_values = [results_by_alpha[a]['metrics']['f1_micro'] for a in alphas]

    # Plot the four metric curves.
    ax.plot(alphas, map_macro_values, 'o-', linewidth=2.5, markersize=8,
            label='mAP (macro)', color='#FF6B6B', alpha=0.9)
    ax.plot(alphas, map_micro_values, 's-', linewidth=2.5, markersize=8,
            label='mAP (micro)', color='#4ECDC4', alpha=0.9)
    ax.plot(alphas, f1_macro_values, '^-', linewidth=2.5, markersize=8,
            label='F1 (macro)', color='#45B7D1', alpha=0.9)
    ax.plot(alphas, f1_micro_values, 'v-', linewidth=2.5, markersize=8,
            label='F1 (micro)', color='#FFA07A', alpha=0.9)

    # Mark the highest GMean with a gold star.
    ax.scatter([best_gmean_alpha], [results_by_alpha[best_gmean_alpha]['metrics']['map_macro']],
               s=400, marker='*', color='gold', edgecolors='green',
               linewidths=2, zorder=10, label=f'Best GMean (alpha={best_gmean_alpha:.1f})')

    # Mark the best point for each metric with a red circle.
    ax.scatter([best_map_macro_alpha],
               [results_by_alpha[best_map_macro_alpha]['metrics']['map_macro']],
               s=200, marker='o', color='none', edgecolors='red',
               linewidths=2.5, zorder=9)
    ax.scatter([best_map_micro_alpha],
               [results_by_alpha[best_map_micro_alpha]['metrics']['map_micro']],
               s=200, marker='o', color='none', edgecolors='red',
               linewidths=2.5, zorder=9)
    ax.scatter([best_f1_macro_alpha],
               [results_by_alpha[best_f1_macro_alpha]['metrics']['f1_macro']],
               s=200, marker='o', color='none', edgecolors='red',
               linewidths=2.5, zorder=9)
    ax.scatter([best_f1_micro_alpha],
               [results_by_alpha[best_f1_micro_alpha]['metrics']['f1_micro']],
               s=200, marker='o', color='none', edgecolors='red',
               linewidths=2.5, zorder=9, label='Best Individual Metric')

    # Set the plot style.
    ax.set_xlabel('Ensemble Weight (alpha)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Alpha Sweep Analysis: Ensemble Performance vs Weight alpha\n'
                 f'(N={N_EPISODES} episodes, {N_WAY}-way {K_SHOT}-shot)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.0)

    # Add text annotations.
    textstr = f'Best GMean alpha = {best_gmean_alpha:.1f}\nGMean = {best_gmean:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'alpha_sweep_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f" Visualization saved!")

    if probability_analysis_payload is not None:
        print("\n" + "="*70)
        print("STEP 4B: Saving Probability Sharpness Evidence")
        print("="*70)

        analysis_alpha = probability_analysis_payload['alpha']
        analysis_by_model = probability_analysis_payload['analysis']
        predictions_by_model = probability_analysis_payload['predictions_by_model']
        targets = probability_analysis_payload['targets']

        hist_path = os.path.join(OUTPUT_DIR, f'probability_histograms_alpha_{analysis_alpha:.1f}.png')
        threshold_path = os.path.join(OUTPUT_DIR, f'f1_threshold_curves_alpha_{analysis_alpha:.1f}.png')
        reliability_path = os.path.join(OUTPUT_DIR, f'reliability_diagrams_alpha_{analysis_alpha:.1f}.png')
        plot_probability_histograms(predictions_by_model, targets, hist_path)
        plot_threshold_curves(analysis_by_model, threshold_path)
        plot_reliability_diagrams(analysis_by_model, reliability_path)

        print(f" Probability histograms saved to: {hist_path}")
        print(f" F1-threshold curves saved to: {threshold_path}")
        print(f" Reliability diagrams saved to: {reliability_path}")

        if SAVE_QUERY_LEVEL_CSV:
            csv_path = os.path.join(OUTPUT_DIR, f'query_level_probabilities_alpha_{analysis_alpha:.1f}.csv')
            write_query_level_csv(
                csv_path,
                probability_analysis_payload['query_records'],
                predictions_by_model,
                targets,
                probability_analysis_payload['episode_classes_by_record'],
                cat_id_to_name
            )
            print(f" Query-level probabilities saved to: {csv_path}")

        if CASE_STUDY_ENABLED:
            print("\n" + "="*70)
            print("STEP 4C: Building Reproducible Case Study")
            print("="*70)
            case_study_directory = os.path.join(
                OUTPUT_DIR,
                f'{CASE_STUDY_DIRNAME}_alpha_{analysis_alpha:.1f}'
            )
            case_study_summary = generate_case_study_outputs(
                predictions_by_model=predictions_by_model,
                targets=targets,
                query_records=probability_analysis_payload['query_records'],
                episode_classes_by_record=(
                    probability_analysis_payload['episode_classes_by_record']
                ),
                episode_records=probability_analysis_payload['episode_records'],
                dataset=dataset,
                class_names_by_id=cat_id_to_name,
                output_directory=case_study_directory
            )
            print(f" Case-study assets saved to: {case_study_directory}")

    print("\n" + "="*70)
    print("STEP 5: Saving Results")
    print("="*70)

    # Save the complete results.
    results_to_save = {
        'configuration': {
            'n_episodes': N_EPISODES,
            'episode_replay': EPISODE_REPLAY_AUDIT,
            'n_way': N_WAY,
            'k_shot': K_SHOT,
            'n_query': N_QUERY,
            'episode_seed': EPISODE_SEED,
            'comc_alpha': COMC_ALPHA,
            'comc_temperature': COMC_TEMPERATURE,
            'analysis_alpha': ANALYSIS_ALPHA,
            'fixed_threshold': FIXED_THRESHOLD,
            'calibration_bins': CALIBRATION_BINS,
            'threshold_values': [round(float(value), 2) for value in THRESHOLD_VALUES]
        },
        'results_by_alpha': results_by_alpha,
        'best_gmean': {
            'alpha': best_gmean_alpha,
            'gmean_relative_gain': best_gmean,
            'metrics': best_gmean_metrics
        },
        'best_individual_metrics': {
            'map_macro': {
                'alpha': best_map_macro_alpha,
                'value': results_by_alpha[best_map_macro_alpha]['metrics']['map_macro']
            },
            'map_micro': {
                'alpha': best_map_micro_alpha,
                'value': results_by_alpha[best_map_micro_alpha]['metrics']['map_micro']
            },
            'f1_macro': {
                'alpha': best_f1_macro_alpha,
                'value': results_by_alpha[best_f1_macro_alpha]['metrics']['f1_macro']
            },
            'f1_micro': {
                'alpha': best_f1_micro_alpha,
                'value': results_by_alpha[best_f1_micro_alpha]['metrics']['f1_micro']
            }
        },
        'class_names': novel_class_names
    }

    if probability_analysis_payload is not None:
        results_to_save['probability_sharpness_analysis'] = {
            'alpha': probability_analysis_payload['alpha'],
            'models': probability_analysis_payload['analysis'],
            'interpretation_note': (
                'Sharpness is measured over query-label probabilities. '
                'mean_abs_distance_to_0_5 and confident_ratio_0_1_0_9 increase '
                'when probabilities move away from the fixed 0.5 threshold, '
                'whereas ambiguous_ratio_0_4_0_6 decreases when fewer predictions '
                'remain near the decision boundary.'
            )
        }

    if case_study_summary is not None:
        results_to_save['case_study'] = case_study_summary

    results_file = os.path.join(OUTPUT_DIR, 'alpha_sweep_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f" Results saved to: {results_file}")

    # Create the comparison table.
    print("\n" + "="*70)
    print("ALPHA SWEEP SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Alpha':<6} {'mAP(mac)':<11} {'mAP(mic)':<11} {'F1(mac)':<11} {'F1(mic)':<11} {'GMean':<10}")
    print("-" * 70)
    for alpha in sorted(results_by_alpha.keys()):
        m = results_by_alpha[alpha]['metrics']
        gmean = results_by_alpha[alpha]['gmean_relative_gain']
        marker = ' <-- BEST GMean' if alpha == best_gmean_alpha else ''
        print(f"{alpha:<6.1f} {m['map_macro']:<11.4f} {m['map_micro']:<11.4f} "
              f"{m['f1_macro']:<11.4f} {m['f1_micro']:<11.4f} {gmean:<10.4f}{marker}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nRecommendation by GMean: Use ALPHA = {best_gmean_alpha:.1f}")
    print(f"This provides the best GMean relative gain of {best_gmean:.4f}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
