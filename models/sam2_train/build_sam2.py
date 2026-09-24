"""Build the only retained architecture: final Hiera-S E3SAM2."""

from collections.abc import Mapping

import torch

from models.sam2_train.modeling.backbones.hieradet import Hiera
from models.sam2_train.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from models.sam2_train.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from models.sam2_train.modeling.memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
from models.sam2_train.modeling.position_encoding import PositionEmbeddingSine
from models.sam2_train.modeling.sam.mask_decoder import MaskDecoder
from models.sam2_train.modeling.sam.prompt_encoder import PromptEncoder
from models.sam2_train.modeling.sam.transformer import RoPEAttention, TwoWayTransformer
from models.sam2_train.modeling.sam2_base_1_train_encoder import SAM2_train_encoder


def load_pretrained_train20(model, checkpoint_path):
    """Load a flat state dict or the 'model' state dict in a SAM2 checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a state-dict mapping, optionally wrapped under 'model'.")
    current = model.state_dict()
    compatible = {key: value for key, value in checkpoint.items() if key in current}
    if not compatible:
        raise RuntimeError(
            f"No pretrained tensors matched the model in {checkpoint_path}; "
            "check the checkpoint format and model architecture."
        )
    # Custom modules may be missing; same-name shape mismatches still raise.
    model.load_state_dict(compatible, strict=False)
    print(f"Pretrained loading: {len(compatible)}/{len(current)} tensors")
    return len(compatible)


def build_model(config, pretrained_checkpoint: str | None = None):
    image_size = config.image_size
    model = SAM2_train_encoder(
        image_encoder=ImageEncoder(
            trunk=Hiera(
                embed_dim=96,
                num_heads=1,
                stages=(1, 2, 11, 2),
                global_att_blocks=(7, 10, 13),
                window_pos_embed_bkg_spatial_size=(7, 7),
                window_spec=(8, 4, 14, 7),
            ),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(num_pos_feats=256, temperature=10000, normalize=True, scale=None),
                d_model=256,
                backbone_channel_list=[768, 384, 192, 96],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest",
            ),
            scalp=1,
        ),
        memory_attention=MemoryAttention(
            d_model=256,
            pos_enc_at_input=True,
            layer=MemoryAttentionLayer(
                activation="relu",
                dim_feedforward=2048,
                dropout=0.1,
                pos_enc_at_attn=False,
                self_attention=RoPEAttention(
                    rope_theta=10000.0, feat_sizes=[32, 32], embedding_dim=256,
                    num_heads=1, downsample_rate=1, dropout=0.1,
                ),
                d_model=256,
                pos_enc_at_cross_attn_keys=True,
                pos_enc_at_cross_attn_queries=False,
                cross_attention=RoPEAttention(
                    rope_theta=10000.0, feat_sizes=[32, 32], rope_k_repeat=True,
                    embedding_dim=256, num_heads=1, downsample_rate=1,
                    dropout=0.1, kv_in_dim=64,
                ),
            ),
            num_layers=4,
        ),
        memory_encoder=MemoryEncoder(
            out_dim=64,
            position_encoding=PositionEmbeddingSine(num_pos_feats=64, normalize=True, scale=None, temperature=10000),
            mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
            fuser=Fuser(layer=CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True), num_layers=2),
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_size // 16, image_size // 16),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
        ),
        memory_bank_size=4,
        use_high_res_features=True,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
    )
    if pretrained_checkpoint:
        load_pretrained_train20(model, pretrained_checkpoint)
    return model
