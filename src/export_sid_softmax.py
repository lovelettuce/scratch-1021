"""Export per-sample teacher-forced SID token probabilities."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import hydra
import torch
import tensorflow as tf
from omegaconf import OmegaConf

from src.utils import custom_hydra_resolvers  # noqa: F401

from src.data.loading.components.collate_functions import collate_fn_train
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.data.loading.components.iterators import TFRecordIterator
from src.data.loading.utils import combine_list_of_tensor_dicts
from src.utils.file_utils import list_files

LOGGER = logging.getLogger("export_sid_softmax")
DEFAULT_RUN_DIR = Path("outputs/1022-reproduction-rqkmeans/beauty_tiger_a100_1gpu")
DEFAULT_SID_PATH = Path(
    "outputs/1022-reproduction-rqkmeans/beauty_rkmeans_inference/pickle/merged_predictions_tensor.pt"
)


def _to_absolute(path: str | Path, base_dir: Path) -> Path:
    path = Path(path).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _get_checkpoint(run_dir: Path, checkpoint: Optional[str]) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint under {run_dir / 'checkpoints'}, "
            f"found {len(checkpoints)}. Specify --checkpoint explicitly."
        )
    return checkpoints[0].resolve()


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    cfg = OmegaConf.load(config_path)
    root_dir = config_path.parent.parent.parent.parent.parent.resolve()
    cfg.data_dir = str(_to_absolute(cfg.data_dir, root_dir))
    sid_path = _to_absolute(DEFAULT_SID_PATH, root_dir)
    if not sid_path.is_file():
        raise FileNotFoundError(f"Beauty SID map not found: {sid_path}")
    cfg.semantic_id_path = str(sid_path)

    # Instantiate only the model subtree; no trainer, callbacks or loggers are needed.
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    model._make_deterministic(is_training=False)

    data_cfg = cfg.data_loading.test_dataloader_config.dataloader
    dataset_node = data_cfg.dataset_config
    dataset_cfg = SimpleNamespace(
        user_id_field=dataset_node.user_id_field,
        data_iterator=hydra.utils.instantiate(dataset_node.data_iterator),
        preprocessing_functions=[
            hydra.utils.instantiate(function)
            for function in dataset_node.preprocessing_functions
        ],
        iterate_per_row=dataset_node.get("iterate_per_row", True),
        keep_user_id=dataset_node.get("keep_user_id", False),
        num_placeholder_tokens_map=OmegaConf.to_container(
            dataset_node.get("num_placeholder_tokens_map", {}), resolve=True
        ),
        field_type_map={
            feature.name: hydra.utils.instantiate(feature.type)
            for feature in cfg.data_loading.features_config.features
            if feature.get("type") is not None
        },
        min_sequence_length=dataset_node.get("min_sequence_length", 1),
        feature_map=dataset_node.get("feature_map", None),
        features_to_consider=list(dataset_node.features_to_consider),
        file_format=dataset_node.get("file_format", None),
        semantic_id_map=hydra.utils.instantiate(dataset_node.semantic_id_map),
    )
    data_cfg = SimpleNamespace(**OmegaConf.to_container(data_cfg, resolve=True))
    data_cfg.dataset_config = dataset_cfg
    return cfg, model, data_cfg, sid_path


def _iter_processed_records(data_cfg, file_path: str) -> Iterator[Dict[str, Any]]:
    dataset_cfg = data_cfg.dataset_config
    iterator: TFRecordIterator = dataset_cfg.data_iterator
    iterator.update_list_of_file_paths([file_path])
    iterator.should_shuffle_rows = False

    for record_index, row in enumerate(iterator.iterrows()):
        target_item_id = None
        for preprocess in dataset_cfg.preprocessing_functions:
            row = preprocess(row, dataset_config=dataset_cfg)
            if row is None:
                break
            target_name = getattr(preprocess, "func", preprocess).__name__
            if target_name == "convert_fields_to_tensors":
                target_item_id = row["sequence_data"].reshape(-1)[-1].clone()
        if row:
            if target_item_id is None:
                raise RuntimeError("Could not capture the target item ID before SID mapping")
            row["__target_item_id"] = target_item_id
            row["__source_file"] = file_path
            row["__source_record_index"] = record_index
            yield row


def _collate_records(records: Sequence[Dict[str, Any]], data_cfg):
    metadata = [
        (row.pop("__source_file"), row.pop("__source_record_index"))
        for row in records
    ]
    item_ids = torch.stack([row.pop("__target_item_id") for row in records])
    batch = combine_list_of_tensor_dicts(list(records))
    labels = {
        name: hydra.utils.instantiate(label)
        if OmegaConf.is_config(label) or isinstance(label, dict)
        else label
        for name, label in data_cfg.labels.items()
    }
    model_input, label_data = collate_fn_train(
        batch=batch,
        labels=labels,
        sequence_length=data_cfg.sequence_length,
        masking_token=data_cfg.masking_token,
        padding_token=data_cfg.padding_token,
        oov_token=data_cfg.oov_token,
    )
    user_ids = torch.stack(
        [row[data_cfg.dataset_config.user_id_field].reshape(-1)[0] for row in records]
    )
    return model_input, label_data, user_ids, item_ids, metadata


def compute_teacher_forced_probabilities(
    model,
    model_input: SequentialModelInputData,
    label_data: SequentialModuleLabelData,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return probabilities [B,H,V] and the corresponding target SIDs [B,H]."""
    if len(label_data.labels) != 1:
        raise ValueError(f"Expected one SID label field, got {list(label_data.labels)}")
    target_sids = label_data.labels["sequence_data"].reshape(
        model_input.mask.size(0), -1
    )
    if target_sids.shape[1] != model.num_hierarchies:
        raise ValueError(
            f"Expected {model.num_hierarchies} target SID levels, got {target_sids.shape[1]}"
        )

    model_output = model(
        attention_mask_encoder=model_input.mask,
        future_ids=target_sids,
        **{
            model.feature_to_model_input_map.get(name, name): value
            for name, value in model_input.transformed_sequences.items()
        },
    )[:, :-1]
    if model_output.shape[1] != model.num_hierarchies:
        raise ValueError(
            f"Expected {model.num_hierarchies} decoder states, got {model_output.shape[1]}"
        )

    level_logits = [
        model.decoder.decoder_mlp[level](model_output[:, level])
        for level in range(model.num_hierarchies)
    ]
    logits = torch.stack(level_logits, dim=1).float()
    probabilities = torch.softmax(logits, dim=-1)
    target_nll = -probabilities.gather(-1, target_sids.unsqueeze(-1)).squeeze(-1).log()
    for level in range(model.num_hierarchies):
        head_loss = model.loss_function(
            logits[:, level], target_sids[:, level].long()
        )
        if not torch.allclose(
            head_loss,
            target_nll[:, level].mean(),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(f"Head {level} softmax/NLL disagrees with cross entropy")
    return probabilities, target_sids


def _ensure_output_available(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    return path


def export(
    run_dir: Path,
    checkpoint: Optional[str],
    output_path: Path,
    device_name: str,
    batch_size: int,
    max_samples: Optional[int],
) -> Path:
    run_dir = run_dir.resolve()
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Archived Hydra config not found: {config_path}")
    checkpoint_path = _get_checkpoint(run_dir, checkpoint)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when specified")
    output_path = _ensure_output_available(output_path)

    cfg, model, data_cfg, sid_path = _load_model(config_path, checkpoint_path, device)
    expected_levels = int(cfg.num_hierarchies)
    if model.num_hierarchies != expected_levels:
        raise ValueError(
            f"Model has {model.num_hierarchies} levels, config expects {expected_levels}"
        )
    vocab_sizes = {head.out_features for head in model.decoder.decoder_mlp}
    if len(model.decoder.decoder_mlp) != expected_levels or len(vocab_sizes) != 1:
        raise ValueError("Output heads do not have a consistent configured shape")
    vocab_size = vocab_sizes.pop()

    file_paths = sorted(list_files(data_cfg.data_folder, suffix="*.tfrecord.gz"))
    if not file_paths:
        raise FileNotFoundError(f"No test TFRecords found in {data_cfg.data_folder}")

    probability_batches: List[torch.Tensor] = []
    target_batches: List[torch.Tensor] = []
    user_id_batches: List[torch.Tensor] = []
    item_id_batches: List[torch.Tensor] = []
    source_files: List[str] = []
    source_record_indices: List[int] = []
    pending_records: List[Dict[str, Any]] = []
    sample_count = 0

    def process_records(records: List[Dict[str, Any]]) -> None:
        nonlocal sample_count
        model_input, label_data, user_ids, item_ids, locations = _collate_records(records, data_cfg)
        model_input.mask = model_input.mask.to(device)
        model_input.transformed_sequences = {
            key: value.to(device)
            for key, value in model_input.transformed_sequences.items()
        }
        label_data.labels = {key: value.to(device) for key, value in label_data.labels.items()}
        with torch.inference_mode():
            probabilities, targets = compute_teacher_forced_probabilities(
                model, model_input, label_data
            )
        if probabilities.shape[1:] != (expected_levels, vocab_size):
            raise ValueError(
                f"Unexpected probability shape {tuple(probabilities.shape)}; "
                f"expected [B,{expected_levels},{vocab_size}]"
            )
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Softmax output contains non-finite or negative values")
        if not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[..., 0]),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("A per-level softmax distribution does not sum to one")
        probability_batches.append(probabilities.cpu())
        target_batches.append(targets.cpu())
        user_id_batches.append(user_ids.cpu())
        item_id_batches.append(item_ids.cpu())
        source_files.extend(path for path, _ in locations)
        source_record_indices.extend(index for _, index in locations)
        sample_count += len(records)

    for file_path in file_paths:
        for record in _iter_processed_records(data_cfg, file_path):
            pending_records.append(record)
            if len(pending_records) == batch_size:
                process_records(pending_records)
                pending_records = []
            if max_samples is not None and sample_count >= max_samples:
                pending_records = []
                break
            if (
                pending_records
                and max_samples is not None
                and sample_count + len(pending_records) >= max_samples
            ):
                remaining = max_samples - sample_count
                process_records(pending_records[:remaining])
                pending_records = []
                break
        if max_samples is not None and sample_count >= max_samples:
            break
    if pending_records and (max_samples is None or sample_count < max_samples):
        process_records(pending_records)

    if not probability_batches:
        raise RuntimeError("No test samples were exported")
    probabilities = torch.cat(probability_batches)
    target_sids = torch.cat(target_batches)
    if probabilities.shape[0] != sample_count or target_sids.shape != (sample_count, expected_levels):
        raise RuntimeError("Saved probability and target counts do not match processed samples")

    sid_map = torch.load(sid_path, map_location="cpu", weights_only=True)
    if sid_map.ndim != 2 or sid_map.shape[0] < expected_levels:
        raise ValueError(f"Invalid SID map shape: {tuple(sid_map.shape)}")
    item_ids = torch.cat(item_id_batches).long()
    if item_ids.min() < 0 or item_ids.max() >= sid_map.shape[1]:
        raise ValueError("A test target item ID is outside the SID map")
    expected_sids = sid_map[:expected_levels, item_ids].T.long()
    if not torch.equal(target_sids.long(), expected_sids):
        raise ValueError("Test target SIDs do not match the configured Beauty SID map")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probabilities": probabilities.float(),
        "target_sids": target_sids.long(),
        "target_item_ids": item_ids,
        "user_ids": torch.cat(user_id_batches).long(),
        "source_files": source_files,
        "source_record_indices": torch.tensor(source_record_indices, dtype=torch.long),
        "metadata": {
            "mode": "teacher_forcing",
            "probability_condition": "true_sid_prefix",
            "probability_shape": list(probabilities.shape),
            "num_samples": sample_count,
            "num_hierarchies": expected_levels,
            "vocab_size": vocab_size,
            "probability_dtype": "float32",
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path.resolve()),
            "semantic_id_path": str(sid_path.resolve()),
            "data_dir": str(Path(cfg.data_dir).resolve()),
            "test_split": str(Path(data_cfg.data_folder).resolve()),
            "target_definition": "last item in each original test sequence",
            "generation_constraints_applied": False,
        },
    }
    torch.save(payload, output_path)
    LOGGER.info("Saved %d samples with probabilities shaped %s to %s", sample_count, tuple(probabilities.shape), output_path)
    return output_path


def _parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/beauty_test_teacher_forcing_softmax.pt")
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    export(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
