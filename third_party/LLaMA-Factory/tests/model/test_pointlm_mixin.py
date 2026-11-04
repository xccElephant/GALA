from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from llamafactory.model.pointlm.pointlm_mixin import PointLMMixin


class _DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def embed_tokens(self, input_ids: torch.LongTensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        return torch.zeros(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        return BaseModelOutputWithPast(last_hidden_state=inputs_embeds)


class _DummyPointLM(PointLMMixin, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
            tokenizer_vocab_size=8,
            vocab_size=8,
        )
        self.model = _DummyBackbone(hidden_size=4)
        self.lm_head = nn.Linear(4, 8, bias=False)

    def _process_separate_point_clouds(
        self,
        input_ids,
        inputs_embeds,
        attention_mask,
        separate_point_clouds,
        separate_point_clouds_counts,
    ):
        pcd_pos_info = [[(1, 2, 0)], [(1, 2, 0)]]
        return inputs_embeds, attention_mask, pcd_pos_info, []

    def _prepare_labels_with_separate_point_clouds(self, labels, logits_seq_len, pcd_pos_info):
        return labels

    def _collect_pcd_diagnostics(self, *args, **kwargs) -> None:
        return None

    def loss_function(self, logits, labels, vocab_size, **loss_kwargs):
        return torch.tensor(0.0)


def test_forward_skips_generation_repadding_when_eval_has_labels():
    model = _DummyPointLM()
    model.eval()

    def _unexpected_repad(*args, **kwargs):
        raise AssertionError("generation-only repadding should not run for eval loss")

    model._right_pad_to_left_pad = _unexpected_repad

    batch_size, seq_len = 2, 4
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    separate_point_clouds = torch.zeros(batch_size, 1, 1, 6)
    separate_point_clouds_counts = torch.ones(batch_size, dtype=torch.long)

    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        separate_point_clouds=separate_point_clouds,
        separate_point_clouds_counts=separate_point_clouds_counts,
    )


def test_forward_keeps_generation_repadding_when_eval_has_no_labels():
    model = _DummyPointLM()
    model.eval()
    repad_calls = 0

    def _track_repad(inputs_embeds, attention_mask):
        nonlocal repad_calls
        repad_calls += 1
        return inputs_embeds, attention_mask

    model._right_pad_to_left_pad = _track_repad

    batch_size, seq_len = 2, 4
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    separate_point_clouds = torch.zeros(batch_size, 1, 1, 6)
    separate_point_clouds_counts = torch.ones(batch_size, dtype=torch.long)

    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        separate_point_clouds=separate_point_clouds,
        separate_point_clouds_counts=separate_point_clouds_counts,
    )

    assert repad_calls == 1
