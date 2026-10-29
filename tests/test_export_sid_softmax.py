from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.utils import custom_hydra_resolvers  # noqa: F401
from src.data.loading.components.label_function import NextKTokenMasking
from src.export_sid_softmax import (
    _collate_records,
    _ensure_output_available,
    compute_teacher_forced_probabilities,
)


class ToyModel(nn.Module):
    num_hierarchies = 4
    feature_to_model_input_map = {"sequence_data": "input_ids"}

    def __init__(self):
        super().__init__()
        self.loss_function = nn.CrossEntropyLoss()
        self.decoder = SimpleNamespace(
            decoder_mlp=nn.ModuleList([nn.Linear(2, 3, bias=False) for _ in range(4)])
        )
        for level, head in enumerate(self.decoder.decoder_mlp):
            head.weight.data.fill_(0)
            head.weight.data[level % 3].fill_(level + 1)
        self.seen_future_ids = None

    def forward(self, attention_mask_encoder, future_ids, input_ids):
        self.seen_future_ids = future_ids.clone()
        batch = input_ids.shape[0]
        states = torch.arange(5, dtype=torch.float32).view(1, 5, 1).expand(batch, -1, 2)
        return states


def test_teacher_forced_probabilities_align_each_head_to_its_level():
    model = ToyModel()
    model_input = SimpleNamespace(
        mask=torch.ones(2, 3, dtype=torch.long),
        transformed_sequences={"sequence_data": torch.ones(2, 3, dtype=torch.long)},
    )
    labels = torch.tensor([[0, 1, 2, 1], [2, 1, 0, 2]])
    label_data = SimpleNamespace(labels={"sequence_data": labels.flatten()})

    probabilities, targets = compute_teacher_forced_probabilities(
        model, model_input, label_data
    )

    assert probabilities.shape == (2, 4, 3)
    assert torch.equal(targets, labels)
    assert torch.equal(model.seen_future_ids, labels)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(2, 4), atol=1e-6, rtol=1e-6
    )
    expected = torch.stack(
        [
            torch.softmax(model.decoder.decoder_mlp[level](torch.ones(2, 2) * level), -1)
            for level in range(4)
        ],
        dim=1,
    )
    torch.testing.assert_close(probabilities, expected)


def test_collation_preserves_last_item_sid_target_and_source_location():
    records = []
    for idx, target_item_id in enumerate((91, 92)):
        records.append(
            {
                "sequence_data": torch.tensor([10, 11, 12, 13, 20 + idx, 30 + idx, 40 + idx, 50 + idx]),
                "user_id": torch.tensor([700 + idx]),
                "__target_item_id": torch.tensor(target_item_id),
                "__source_file": f"partition_{idx}.tfrecord.gz",
                "__source_record_index": idx + 5,
            }
        )
    data_cfg = SimpleNamespace(
        dataset_config=SimpleNamespace(user_id_field="user_id"),
        labels={"sequence_data": SimpleNamespace(transform=NextKTokenMasking(next_k=4))},
        sequence_length=8,
        masking_token=-1,
        padding_token=-1,
        oov_token=None,
    )

    model_input, label_data, user_ids, item_ids, locations = _collate_records(records, data_cfg)

    assert label_data.labels["sequence_data"].reshape(2, 4).tolist() == [
        [20, 30, 40, 50],
        [21, 31, 41, 51],
    ]
    assert item_ids.tolist() == [91, 92]
    assert user_ids.tolist() == [700, 701]
    assert locations == [("partition_0.tfrecord.gz", 5), ("partition_1.tfrecord.gz", 6)]
    assert model_input.transformed_sequences["sequence_data"][:, -4:].tolist() == [
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path / "softmax.pt"
    output.write_bytes(b"preserve")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _ensure_output_available(output)

    assert output.read_bytes() == b"preserve"
