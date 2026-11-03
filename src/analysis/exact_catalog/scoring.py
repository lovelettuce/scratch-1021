"""Teacher-forced raw-vocabulary scores with one encoder pass per history batch."""
from __future__ import annotations

import torch


def configure_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def encode(model, model_input):
    device = next(model.parameters()).device
    kwargs = {model.feature_to_model_input_map.get(k, k): v.to(device)
              for k, v in model_input.transformed_sequences.items()}
    return model.encoder_forward_pass(attention_mask=model_input.mask.to(device), **kwargs)


def candidate_scores(model, encoded, mask, candidates, chunk_size=128):
    """candidates: [history, candidate, SID position]; return CPU per-token logp.

    No legal-child normalization, KV cache or probability multiplication. Each
    output position sees only its own candidate's preceding SID tokens.
    """
    if chunk_size < 1:
        raise ValueError('candidate chunk size must be positive')
    b, count, levels = candidates.shape
    if b != encoded.shape[0] or levels != model.num_hierarchies or count == 0:
        raise ValueError('Candidate/encoder dimensions disagree')
    result = []
    with torch.inference_mode(), torch.autocast(device_type=encoded.device.type, enabled=False):
        for start in range(0, count, chunk_size):
            target = candidates[:, start:start+chunk_size].to(encoded.device).long()
            width = target.shape[1]
            flat = target.reshape(-1, levels)
            hidden = model.decoder_forward_pass(
                # BOS plus L-1 tokens supplies exactly L prediction positions.
                future_ids=flat[:, :-1], encoder_output=encoded.repeat_interleave(width, dim=0),
                attention_mask_for_encoder=mask.repeat_interleave(width, dim=0), use_cache=False)
            values = []
            for level, head in enumerate(model.decoder.decoder_mlp):
                logits = head(hidden[:, level]).float()
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite teacher-forced logits')
                values.append(logits.log_softmax(-1).gather(1, flat[:, level:level+1]).squeeze(1))
            result.append(torch.stack(values, -1).reshape(b, width, levels).cpu())
    return torch.cat(result, dim=1)


def catalog_scores(model, encoded, mask, catalog, chunk_size=128):
    """Store only total scores, keeping per-token allocations bounded by chunk."""
    scores = torch.empty(encoded.shape[0], len(catalog), dtype=torch.float32)
    for start in range(0, len(catalog), chunk_size):
        ids = catalog[start:start+chunk_size].unsqueeze(0).expand(encoded.shape[0], -1, -1)
        scores[:, start:start+len(ids[0])] = candidate_scores(
            model, encoded, mask, ids, chunk_size).sum(-1)
    return scores
