"""
PointLM Mixin for point cloud processing capabilities.

This mixin provides shared functionality for all PointLM model variants.
It encodes point clouds with the Utonia backbone and
inserts the resulting features into the token embedding sequence at
<|point_start|>/<|point_end|> placeholder positions (separate_point_clouds mode).
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from .utils import is_rank0_process

IGNORE_INDEX = -100
_DIAG_LOG_INTERVAL = 10  # log diagnostics every N training steps
logger = logging.get_logger(__name__)


class PointLMMixin:
    """
    Mixin class providing point cloud processing capabilities for PointLM models.

    Should be mixed with a base LLM class (e.g. Qwen3ForCausalLM). After the
    base class initializes ``model``, ``vocab_size``, and ``lm_head``, call
    ``_init_point_cloud_modules(config)`` to set up the point cloud encoder and MLP projector.

    During the forward pass, ``separate_point_clouds`` is encoded in a single
    batched backbone call, then the
    resulting per-cloud feature tensors are inserted between every pair of
    ``<|point_start|>`` / ``<|point_end|>`` tokens found in ``input_ids``.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_point_cloud_modules(self, config) -> None:
        """
        Initialize the point cloud encoder and projector.

        Sets the following attributes on ``self``:
          - ``point_backbone``  the Utonia encoder
          - ``point_proj``  two-layer MLP projector
          - ``point_start_token_id``, ``point_end_token_id``, ``point_token_id``
        """
        point_config = getattr(config, "point_config", None)
        if point_config is None:
            raise ValueError("Config must have a 'point_config' attribute.")

        backbone_type = point_config["backbone_type"]
        if backbone_type != "utonia":
            raise ValueError(
                f"Unsupported point backbone_type {backbone_type!r}; "
                "only 'utonia' is supported."
            )

        from .utonia_encoder import Utonia

        enc_channels = point_config["enc_channels"]
        self.point_backbone = Utonia(
            in_channels=point_config["in_channels"],
            order=point_config["order"],
            stride=tuple(point_config["stride"]),
            enc_depths=point_config["enc_depths"],
            enc_channels=enc_channels,
            enc_num_head=point_config["enc_num_head"],
            enc_patch_size=point_config["enc_patch_size"],
            mlp_ratio=point_config["mlp_ratio"],
            mask_token=point_config["mask_token"],
            enc_mode=point_config["enc_mode"],
            enable_flash=point_config["enable_flash"],
            rope_base=point_config["rope_base"],
            drop_path=point_config["drop_path"],
        )
        embed_channels = enc_channels[-1]
        if is_rank0_process():
            logger.warning(
                f"[PointLM] Using Utonia backbone: "
                f"in_channels={point_config['in_channels']}, "
                f"enc_channels={enc_channels}, "
                f"enc_mode={point_config['enc_mode']}, "
                f"rope_base={point_config['rope_base']}."
            )

        self.point_proj = nn.Sequential(
            nn.Linear(embed_channels, embed_channels),
            nn.GELU(),
            nn.Linear(embed_channels, config.hidden_size),
        )

        self.point_start_token_id = config.point_start_token_id
        self.point_end_token_id = config.point_end_token_id
        self.point_token_id = config.point_token_id

        self._proj_grad_norm_cache: float = 0.0

    def set_point_backbone_dtype(self, dtype: torch.dtype) -> None:
        """Cast all point cloud encoder parameters to *dtype*."""
        for param in self.point_backbone.parameters():
            param.data = param.data.to(dtype)

    def register_ntl(self, tokenizer) -> None:
        """Register Number Token Loss alongside the cross-entropy.

        Must be called after tokenizer is fully initialised (including any
        ``resize_token_embeddings``).  Controlled via ``use_ntl_loss`` in
        FinetuningArguments; called automatically by ``load_model`` when set.
        """
        try:
            from ntloss import NTLoss
        except ImportError as e:
            raise ImportError(
                "ntloss is not installed. Run: pip install ntloss"
            ) from e

        self._ntl_fn = NTLoss(tokenizer=tokenizer)
        self._ntl_vocab_size = len(tokenizer)
        if is_rank0_process():
            logger.warning(
                f"[PointLM] NTLoss enabled (weight=0.3, vocab_size={len(tokenizer)})."
            )

    def get_model(self):
        return self.model

    # ------------------------------------------------------------------
    # Point-cloud encoding
    # ------------------------------------------------------------------

    def _encode_point_clouds_batched(
        self,
        point_clouds: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        """
        Encode *N* point clouds in a single backbone forward pass.

        Args:
            point_clouds: NaN-padded tensor shaped ``(N, max_pts, 12)`` —
                ``[gx, gy, gz, x, y, z, r, g, b, nx, ny, nz]``.
            device: target device for the output tensors.
            dtype:  target dtype for the output tensors.

        Returns:
            List of *N* tensors, each shaped ``(1, num_tokens_i, hidden_size)``.
            Empty point clouds yield a zero tensor with ``num_tokens_i == 0``.
        """
        self.point_backbone.to(torch.float32)
        n_clouds = point_clouds.shape[0]

        all_grid_coords: List[torch.Tensor] = []
        all_feats: List[torch.Tensor] = []
        all_batch_ids: List[torch.Tensor] = []
        valid_indices: List[int] = []

        for i in range(n_clouds):
            pcd = point_clouds[i]
            valid = ~torch.isnan(pcd).any(dim=1)
            pcd = pcd[valid]
            if pcd.shape[0] == 0:
                continue
            all_grid_coords.append(pcd[:, :3].int())
            # feat = cat(coord, color, normal) = columns 3:12
            all_feats.append(pcd[:, 3:12].float())
            all_batch_ids.append(torch.full((pcd.shape[0],), i, dtype=torch.long))
            valid_indices.append(i)

        empty = torch.zeros(1, 0, self.config.hidden_size, device=device, dtype=dtype)
        if not all_feats:
            return [empty.clone() for _ in range(n_clouds)]

        cat_grid_coords = torch.cat(all_grid_coords).to(device)
        cat_feats = torch.cat(all_feats).to(device)
        cat_batch_ids = torch.cat(all_batch_ids).to(device)

        input_dict = {
            "coord": cat_feats[:, :3],
            "grid_coord": cat_grid_coords,
            "feat": cat_feats,
            "batch": cat_batch_ids,
        }

        # spconv sparse-conv kernels require fp32; autocast must be disabled.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            context, output_batch_indices = self.point_backbone(
                input_dict, return_batch_indices=True
            )

        features_list: List[torch.Tensor] = [empty.clone() for _ in range(n_clouds)]
        for i in valid_indices:
            mask = output_batch_indices == i
            feat_i = context[mask].unsqueeze(0)  # (1, T_i, C)
            features_list[i] = self.point_proj(feat_i.to(dtype))

        return features_list

    # ------------------------------------------------------------------
    # Point-cloud insertion
    # ------------------------------------------------------------------

    def _process_separate_point_clouds(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor,
        separate_point_clouds: torch.Tensor,
        separate_point_clouds_counts: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, torch.Tensor, List[List[Tuple[int, int, int]]], List[torch.Tensor]]:
        """
        Encode all point clouds and splice their features into the embedding sequence.

        For each sample *i* with ``separate_point_clouds_counts[i] == N_i``, this
        method locates *N_i* consecutive ``<|point_start|>`` / ``<|point_end|>``
        token pairs in ``input_ids[i]`` (left-to-right) and replaces the span
        between each pair with the corresponding encoded point-cloud features.

        Args:
            input_ids: ``(B, S)``
            inputs_embeds: ``(B, S, H)``
            attention_mask: ``(B, S)``
            separate_point_clouds: ``(total_N, max_pts, 12)`` NaN-padded.
            separate_point_clouds_counts: ``(B,)``

        Returns:
            ``(inputs_embeds, attention_mask, all_pos_info, all_features)`` where
            ``all_pos_info[i]`` is a list of ``(orig_start, orig_end, n_patches)``
            tuples recording the original (pre-expansion) token positions, and
            ``all_features`` is the list of encoded per-cloud tensors.
        """
        all_features = self._encode_point_clouds_batched(
            separate_point_clouds, inputs_embeds.device, inputs_embeds.dtype
        )

        if is_rank0_process() and not hasattr(self, "_pcd_tokens_logged"):
            self._pcd_tokens_logged = True
            if all_features:
                logger.warning(
                    f"[PointLM] First point cloud encoded to {all_features[0].shape[1]} tokens "
                    f"(padded input shape: {separate_point_clouds[0].shape})."
                )

        counts = separate_point_clouds_counts.tolist()
        total_expected = sum(int(c) for c in counts)
        if total_expected != len(all_features):
            raise ValueError(
                f"[PointLM] separate_point_clouds_counts mismatch: sum(counts)={total_expected}, "
                f"but encoded point clouds={len(all_features)}."
            )

        features_per_sample: List[List[torch.Tensor]] = []
        offset = 0
        for count in counts:
            features_per_sample.append(all_features[offset : offset + int(count)])
            offset += int(count)

        all_pos_info: List[List[Tuple[int, int, int]]] = []
        new_input_embeds: List[torch.Tensor] = []
        new_attention_mask: List[torch.Tensor] = []
        max_len = 0

        for bi in range(inputs_embeds.shape[0]):
            cur_ids = input_ids[bi]
            cur_embeds = inputs_embeds[bi]
            cur_mask = attention_mask[bi]
            n_pcds = int(counts[bi]) if bi < len(counts) else 0

            if n_pcds == 0:
                new_input_embeds.append(cur_embeds)
                new_attention_mask.append(cur_mask)
                all_pos_info.append([])
                max_len = max(max_len, cur_embeds.shape[0])
                continue

            starts = torch.where(cur_ids == self.config.point_start_token_id)[0].tolist()
            ends = torch.where(cur_ids == self.config.point_end_token_id)[0].tolist()

            if len(starts) < n_pcds or len(ends) < n_pcds:
                raise ValueError(
                    f"[PointLM] Sample {bi}: expected {n_pcds} point-token pairs, "
                    f"found {len(starts)} start tokens and {len(ends)} end tokens. "
                    f"Check that the data has the correct number of placeholders."
                )

            sample_features = features_per_sample[bi]
            if len(sample_features) != n_pcds:
                raise ValueError(
                    f"[PointLM] Sample {bi}: expected {n_pcds} encoded point clouds, "
                    f"but got {len(sample_features)}."
                )
            sample_pos_info: List[Tuple[int, int, int]] = []
            embed_segs: List[torch.Tensor] = []
            mask_segs: List[torch.Tensor] = []
            prev = 0

            for i in range(n_pcds):
                feat = sample_features[i].to(device=cur_embeds.device).squeeze(0)
                n_patches = feat.shape[0]
                orig_start, orig_end = starts[i], ends[i]

                if orig_end <= orig_start:
                    raise ValueError(
                        f"[PointLM] Sample {bi}: invalid token pair "
                        f"start={orig_start}, end={orig_end}."
                    )

                embed_segs.append(cur_embeds[prev : orig_start + 1])
                embed_segs.append(feat)
                mask_segs.append(cur_mask[prev : orig_start + 1])
                mask_segs.append(
                    torch.ones(n_patches, device=cur_mask.device, dtype=cur_mask.dtype)
                )
                prev = orig_end
                sample_pos_info.append((orig_start, orig_end, n_patches))

            embed_segs.append(cur_embeds[prev:])
            mask_segs.append(cur_mask[prev:])
            cur_embeds = torch.cat(embed_segs, dim=0)
            cur_mask = torch.cat(mask_segs, dim=0)

            all_pos_info.append(sample_pos_info)
            new_input_embeds.append(cur_embeds)
            new_attention_mask.append(cur_mask)
            max_len = max(max_len, cur_embeds.shape[0])

        # Pad every sample to the same length.
        for i in range(len(new_input_embeds)):
            e = new_input_embeds[i]
            if e.shape[0] < max_len:
                pad = e[-1:].expand(max_len - e.shape[0], -1)
                new_input_embeds[i] = torch.cat([e, pad], dim=0)

            m = new_attention_mask[i]
            if m.shape[0] < max_len:
                new_attention_mask[i] = F.pad(m, (0, max_len - m.shape[0]))

        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)

        assert attention_mask.shape[1] == inputs_embeds.shape[1], (
            "attention_mask and inputs_embeds length mismatch after point-cloud insertion."
        )
        if (
            self.training
            and is_rank0_process()
            and not hasattr(self, "_point_insert_first_summary_logged")
        ):
            self._point_insert_first_summary_logged = True
            counts_preview = ",".join(map(str, counts[:8]))
            if len(counts) > 8:
                counts_preview += ",..."
            logger.warning(
                f"[PointLM][InsertDebug] First insertion summary: batch={inputs_embeds.shape[0]}, "
                f"orig_seq_len={input_ids.shape[1]}, expanded_max_len={inputs_embeds.shape[1]}, "
                f"total_clouds={len(all_features)}, counts=[{counts_preview}]"
            )
        return inputs_embeds, attention_mask, all_pos_info, all_features

    def _prepare_labels_with_separate_point_clouds(
        self,
        labels: torch.LongTensor,
        max_num_tokens: int,
        all_pos_info: List[List[Tuple[int, int, int]]],
    ) -> torch.LongTensor:
        """
        Align the label tensor with the expanded embedding sequence.

        Inserts ``IGNORE_INDEX`` at every position that now holds injected
        point-cloud features, then pads each sample to ``max_num_tokens``.

        Args:
            labels: ``(B, S)`` original labels.
            max_num_tokens: target sequence length after expansion.
            all_pos_info: output of :meth:`_process_separate_point_clouds`.

        Returns:
            ``(B, max_num_tokens)`` updated labels.
        """
        new_labels: List[torch.Tensor] = []

        for i in range(labels.shape[0]):
            cur_labels = labels[i]
            pos_info = all_pos_info[i] if i < len(all_pos_info) else []

            segs: List[torch.Tensor] = []
            prev = 0
            for orig_start, orig_end, n_patches in pos_info:
                segs.append(cur_labels[prev : orig_start + 1])
                segs.append(
                    torch.full(
                        (n_patches,),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )
                prev = orig_end
            segs.append(cur_labels[prev:])
            cur_labels = torch.cat(segs, dim=0)
            cur_labels = F.pad(
                cur_labels,
                (0, max_num_tokens - cur_labels.shape[0]),
                value=IGNORE_INDEX,
            )
            new_labels.append(cur_labels)

        return torch.stack(new_labels, dim=0)

    @staticmethod
    def _right_pad_to_left_pad(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        """Convert arbitrarily-padded batch to left-padded.

        After point-cloud insertion ``_process_separate_point_clouds`` may
        produce samples with **both** left-padding (from tokenizer
        ``padding_side="left"``) and right-padding (from batch alignment).
        For decoder-only generation the padding must sit on the **left** so
        that every sequence ends at the rightmost column.

        This helper extracts the contiguous valid region (where
        ``attention_mask == 1``) for each sample and right-aligns it.
        """
        B, S, H = inputs_embeds.shape
        valid_lens = attention_mask.sum(dim=1)  # (B,)

        new_embeds = torch.zeros_like(inputs_embeds)
        new_mask = torch.zeros_like(attention_mask)

        for i in range(B):
            vl = int(valid_lens[i].item())
            if vl == S:
                new_embeds[i] = inputs_embeds[i]
                new_mask[i] = attention_mask[i]
            elif vl == 0:
                continue
            else:
                valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
                start = int(valid_positions[0].item())
                pad = S - vl
                new_embeds[i, pad:] = inputs_embeds[i, start : start + vl]
                new_mask[i, pad:] = 1

        return new_embeds, new_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        num_logits_to_keep: int = 0,
        separate_point_clouds: Optional[torch.Tensor] = None,
        separate_point_clouds_counts: Optional[torch.Tensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for PointLM models.

        Accepts both ``logits_to_keep`` (transformers >=5.x) and
        ``num_logits_to_keep`` (transformers 4.x) for backward compatibility.
        """
        # Reconcile the two naming conventions across transformers versions
        _logits_to_keep = logits_to_keep if logits_to_keep != 0 else num_logits_to_keep

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        pcd_processed = False
        pcd_pos_info: List[List[Tuple[int, int, int]]] = []
        pcd_all_features: List[torch.Tensor] = []

        if (
            separate_point_clouds is not None
            and separate_point_clouds_counts is not None
        ):
            inputs_embeds, attention_mask, pcd_pos_info, pcd_all_features = (
                self._process_separate_point_clouds(
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    separate_point_clouds,
                    separate_point_clouds_counts,
                )
            )
            pcd_processed = True

            # After point-cloud insertion the sequence is longer than the
            # original input_ids.  Rebuild cache_position and position_ids so
            # that RoPE cos/sin match the expanded sequence length.
            new_seq_len = inputs_embeds.shape[1]
            past_seen = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_seen = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen, past_seen + new_seq_len,
                device=inputs_embeds.device,
            )

            if (not self.training) and labels is None:
                # During generation the tokenizer uses left-padding, so after
                # point-cloud insertion samples may have both left- and
                # right-padding.  Consolidate to pure left-padding so every
                # sequence ends at the rightmost column.
                if inputs_embeds.shape[0] > 1:
                    inputs_embeds, attention_mask = self._right_pad_to_left_pad(
                        inputs_embeds, attention_mask,
                    )

                # Derive per-sample position_ids from the (post-insertion)
                # attention_mask so that left-padding zeros get position 0
                # instead of a monotonically increasing index.
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                if past_seen:
                    position_ids = position_ids + past_seen

                # Cache the expanded attention_mask so that
                # _update_model_kwargs_for_generation can splice it in
                # correctly (left-pad zeros must be preserved instead of
                # being filled with ones).
                self._expanded_attn_mask = attention_mask
            else:
                position_ids = cache_position.unsqueeze(0)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        slice_indices = (
            slice(-_logits_to_keep, None) if isinstance(_logits_to_keep, int) else _logits_to_keep
        )
        sliced_hidden = hidden_states[:, slice_indices, :]

        use_fused_lce = (
            self.training
            and labels is not None
            and not hasattr(self, "_ntl_fn")
            and getattr(self.config, "enable_liger_kernel", False)
        )

        loss = None
        logits = None

        if use_fused_lce:
            from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

            if pcd_processed:
                logits_len = sliced_hidden.shape[1]
                labels = self._prepare_labels_with_separate_point_clouds(
                    labels, logits_len, pcd_pos_info
                )
                assert labels.shape[1] == logits_len, (
                    "Label length does not match hidden-state length after point-cloud expansion."
                )

            labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)
            shift_labels = labels[..., 1:].contiguous()

            flat_hidden = sliced_hidden.view(-1, sliced_hidden.shape[-1])
            flat_labels = shift_labels.view(-1)

            num_items = loss_kwargs.get("num_items_in_batch", None)
            if num_items is not None:
                reduction = "sum"
            else:
                reduction = "mean"

            loss = liger_fused_linear_cross_entropy(
                flat_hidden,
                self.lm_head.weight,
                flat_labels,
                bias=getattr(self.lm_head, "bias", None),
                ignore_index=IGNORE_INDEX,
                reduction=reduction,
            )

            if num_items is not None:
                if torch.is_tensor(num_items):
                    num_items = num_items.to(loss.device)
                loss = loss / num_items
        else:
            logits = self.lm_head(sliced_hidden)

            if labels is not None:
                if pcd_processed:
                    labels = self._prepare_labels_with_separate_point_clouds(
                        labels, logits.shape[1], pcd_pos_info
                    )
                    assert labels.shape[1] == logits.shape[1], (
                        "Label length does not match logit length after point-cloud expansion."
                    )

                actual_vocab_size = getattr(self.config, "tokenizer_vocab_size", self.config.vocab_size)
                if logits.shape[-1] > actual_vocab_size:
                    logits = logits[..., :actual_vocab_size]

                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=actual_vocab_size,
                    **loss_kwargs,
                )

                if hasattr(self, "_ntl_fn"):
                    ce_loss = loss
                    ntl_logits = logits[..., : self._ntl_vocab_size]
                    num_items = loss_kwargs.get("num_items_in_batch", None)
                    if num_items is not None:
                        ntl_loss = self._ntl_fn(ntl_logits, labels, reduction="sum")
                        if torch.is_tensor(num_items):
                            num_items = num_items.to(ntl_loss.device)
                        ntl_loss = ntl_loss / num_items
                    else:
                        ntl_loss = self._ntl_fn(ntl_logits, labels)
                    loss = ce_loss + 0.3 * ntl_loss
                    self._last_ce_loss = ce_loss.detach()
                    self._last_ntl_loss = ntl_loss.detach()

        self._collect_pcd_diagnostics(pcd_all_features, pcd_processed, input_ids)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        is_first_iteration=None,
        separate_point_clouds=None,
        separate_point_clouds_counts=None,
        **kwargs,
    ):
        """
        Prepare model inputs for autoregressive generation.

        In transformers >=5.x the generation loop is split into a prefill
        phase (``is_first_iteration=True``) and subsequent decode steps.

        During **prefill** the base class will honour ``inputs_embeds`` only
        when ``is_first_iteration=True``.  We therefore pass our custom kwargs
        through so that ``forward`` can perform point-cloud injection.

        During **decode** (subsequent tokens) point-cloud features already
        reside in the KV-cache, so we set them to ``None`` to skip redundant
        re-encoding.
        """
        is_prefill = is_first_iteration is True or (
            is_first_iteration is None and past_key_values is None
        )

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_prefill:
            model_inputs["separate_point_clouds"] = separate_point_clouds
            model_inputs["separate_point_clouds_counts"] = separate_point_clouds_counts
        else:
            model_inputs["separate_point_clouds"] = None
            model_inputs["separate_point_clouds_counts"] = None
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        num_new_tokens=1,
    ):
        """
        Update model kwargs after each generation step.

        After point-cloud injection the KV-cache is longer than the original
        ``attention_mask`` / ``cache_position``.  We let the base class do its
        standard update first, then reconcile all positional bookkeeping with
        the actual cache length.
        """
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        pkv = model_kwargs.get("past_key_values")
        if pkv is not None and hasattr(pkv, "get_seq_length"):
            past_length = pkv.get_seq_length()

            attention_mask = model_kwargs.get("attention_mask")
            if attention_mask is not None and attention_mask.shape[1] < past_length + 1:
                # On the first decode step after prefill with point-cloud
                # insertion, the base-class attention_mask is still at the
                # original (pre-expansion) length.  We must splice in the
                # cached expanded mask (which preserves left/right-pad zeros)
                # instead of blindly padding with ones.
                expanded = getattr(self, "_expanded_attn_mask", None)
                if expanded is not None:
                    # expanded: (B, expanded_seq_len) from prefill
                    # Append a trailing 1 for the newly generated token.
                    model_kwargs["attention_mask"] = torch.cat(
                        [expanded, expanded.new_ones((expanded.shape[0], 1))],
                        dim=-1,
                    )
                    del self._expanded_attn_mask
                else:
                    extra = past_length + 1 - attention_mask.shape[1]
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], extra))],
                        dim=-1,
                    )

            cp = model_kwargs.get("cache_position")
            if cp is not None and cp[-1].item() != past_length:
                model_kwargs["cache_position"] = torch.tensor(
                    [past_length], dtype=cp.dtype, device=cp.device,
                )

            # Derive per-sample position_ids from the current attention_mask.
            # In a left-padded batch each sample may have a different number
            # of valid (attended) tokens, so the next position differs across
            # samples: it equals the count of ones in that sample's mask minus
            # one (because the last attended token has position count-1, and
            # the *next* token to be generated sits at position count).
            attn = model_kwargs.get("attention_mask")
            pos_ids = model_kwargs.get("position_ids")
            if attn is not None and pos_ids is not None:
                next_pos = attn.long().sum(dim=-1, keepdim=True) - 1  # (B, 1)
                model_kwargs["position_ids"] = next_pos

        return model_kwargs

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def update_proj_grad_norm_cache(self) -> None:
        """Update projector grad-norm cache from current parameter gradients.

        This is called by the trainer right after each backward pass and before
        optimizer step/zero_grad, avoiding hook timing issues under different
        distributed training wrappers.
        """
        gnorm_sq = 0.0
        has_grad = False
        for p in self.point_proj.parameters():
            if p.grad is not None:
                has_grad = True
                gnorm_sq += p.grad.detach().float().norm().item() ** 2
        self._proj_grad_norm_cache = gnorm_sq ** 0.5 if has_grad else 0.0

    def _collect_pcd_diagnostics(
        self,
        all_features: List[torch.Tensor],
        pcd_processed: bool,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> None:
        """Collect point-cloud related diagnostics for logging via the trainer.

        Stored on ``self._pcd_diag`` as a plain dict; the trainer reads and
        clears it every logging step.
        """
        if not (self.training and is_rank0_process()):
            return

        if not hasattr(self, "_pcd_diag_step"):
            self._pcd_diag_step = 0
        self._pcd_diag_step += 1

        if self._pcd_diag_step % _DIAG_LOG_INTERVAL != 1:
            return

        diag: dict = {}

        # -- point_proj weight / gradient norms --
        proj_w_norm = 0.0
        for p in self.point_proj.parameters():
            proj_w_norm += p.data.float().norm().item() ** 2
        diag["pcd/proj_weight_norm"] = proj_w_norm ** 0.5
        diag["pcd/proj_grad_norm"] = self._proj_grad_norm_cache

        # -- point_backbone weight norm (frozen, should be constant) --
        bb_w_norm = 0.0
        for p in self.point_backbone.parameters():
            bb_w_norm += p.data.float().norm().item() ** 2
        diag["pcd/backbone_weight_norm"] = bb_w_norm ** 0.5

        # -- encoded feature statistics --
        pcd_token_norms = None
        if pcd_processed and all_features:
            nonempty = [f for f in all_features if f.numel() > 0]
            if nonempty:
                cat_feat = torch.cat([f.flatten() for f in nonempty])
                diag["pcd/feat_mean"] = cat_feat.mean().item()
                diag["pcd/feat_std"] = cat_feat.std().item()
                diag["pcd/feat_abs_max"] = cat_feat.abs().max().item()
                total_tokens = sum(f.shape[1] for f in nonempty)
                diag["pcd/avg_tokens_per_cloud"] = total_tokens / len(nonempty)

                # Per-token L2 norm: this is what directly affects QK dot
                # products in attention. Each feature is (1, T, H).
                all_tokens_2d = torch.cat(
                    [f.squeeze(0) for f in nonempty], dim=0
                )  # (total_T, H)
                pcd_token_norms = all_tokens_2d.float().norm(dim=-1)  # (total_T,)
                diag["pcd/token_norm_mean"] = pcd_token_norms.mean().item()
                diag["pcd/token_norm_std"] = pcd_token_norms.std().item()

        # -- text embedding comparison (per-token L2 norm) --
        # If pcd token norms << text token norms, Q·K_pcd will be
        # systematically smaller and the LLM will attend away from pcd tokens.
        if input_ids is not None:
            try:
                with torch.no_grad():
                    mask = (
                        (input_ids != self.point_start_token_id)
                        & (input_ids != self.point_end_token_id)
                        & (input_ids != self.point_token_id)
                    )
                    text_ids = input_ids[mask][:512]
                    if text_ids.numel() > 0:
                        text_embeds = self.model.embed_tokens(text_ids).float()
                        text_token_norms = text_embeds.norm(dim=-1)  # (T,)
                        diag["pcd/text_token_norm_mean"] = text_token_norms.mean().item()

                        if pcd_token_norms is not None:
                            diag["pcd/norm_ratio"] = (
                                diag["pcd/token_norm_mean"]
                                / diag["pcd/text_token_norm_mean"]
                            )
            except Exception:
                pass

        self._pcd_diag = diag
