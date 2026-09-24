

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from operator import mul
from functools import reduce

from typing import Any, Dict, List, Tuple

from models.sam2_train.modeling.backbones.image_encoder import ImageEncoder
from models.sam2_train.modeling.sam.mask_decoder import MaskDecoder
from models.sam2_train.modeling.sam.prompt_encoder import PromptEncoder
from models.sam2_train.modeling.memory_attention import MemoryAttention
from models.sam2_train.modeling.memory_encoder import MemoryEncoder
from models.sam2_train.modeling.sam2_utils import LearnedPromptConv


from models.sam2_train.modeling.edge_supervision import EdgeEncoder1

from einops import rearrange



class SAM2_train_encoder(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        memory_attention: MemoryAttention,
        memory_encoder: MemoryEncoder,
        use_high_res_features=True,
        non_overlap_masks_for_mem_enc=False,
        binarize_mask_from_pts_for_mem_enc=False,
        memory_bank_size=3,
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        memory_bank_list=None,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.use_high_res_features = use_high_res_features
        self.num_feature_levels = 3 if self.use_high_res_features else 1
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc

        self.memory_attention = memory_attention
        self.hidden_dim = memory_attention.d_model



        self.memory_encoder = memory_encoder
        self.memory_bank_list = [] if memory_bank_list is None else memory_bank_list
        self.mem_bank_size = memory_bank_size
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc



        self.sam_prompt_encoder = prompt_encoder

        self.learned_prompt_conv  = LearnedPromptConv(self.hidden_dim,self.hidden_dim)

        self.sam_mask_decoder = mask_decoder


        self.edge_encoder = EdgeEncoder1()






        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        for param in self.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam_mask_decoder.parameters():
            param.requires_grad = False
        pass
        
    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        imgs: torch.Tensor, # [b,t,c,h,w]
        pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        bbox: torch.Tensor=None, # b 4
    ) -> torch.Tensor:
        return self._forward_with_memory(imgs, pt, bbox)

    def _forward_with_memory(
        self,
        imgs: torch.Tensor, # [b,t,c,h,w]
        pt: Tuple[torch.Tensor, torch.Tensor],  # ([b n 1 2], [b n])
        bbox: torch.Tensor=None, # b 4
    ) -> torch.Tensor:
        device = imgs.device
        b, t, c, h, w = imgs.shape  # b t c h w
        imgs1 = rearrange(imgs, "b t c h w -> (b t) c h w")
        backbone_out = self.forward_image(imgs1)



        pre_edge = self.edge_encoder(backbone_out["backbone_fpn"])





        _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

        vision_feats = [feat.view(reduce(mul, feat_size),b, t, -1)
                 for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
        vision_pos_embeds = [feat.view(reduce(mul, feat_size), b, t, -1)
                        for feat, feat_size in zip(vision_pos_embeds[::-1], feat_sizes[::-1])][::-1]

        frames_pred = []
        for ti in range(0, t):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []
            frame_vision_feats = [vision_feat[:,:,ti,:] for vision_feat in vision_feats]
            frame_vision_pos_embeds = [vision_pos_embed[:, :, ti, :] for vision_pos_embed in vision_pos_embeds]
            if len(self.memory_bank_list) == 0:
                frame_vision_feats[-1] = frame_vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, b, self.hidden_dim)).to(
                    device=device)
                frame_vision_pos_embeds[-1] = frame_vision_pos_embeds[-1] + torch.nn.Parameter(
                    torch.zeros(1, b, self.hidden_dim)).to(
                    device=device)

            else:
                for element in self.memory_bank_list:
                    to_cat_memory.append(
                        element[0].to(device=device, non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_memory_pos.append(
                        element[1].to(device=device, non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_image_embed.append(element[3].to(device=device, non_blocking=True))

                memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                frame_vision_feats_temp = frame_vision_feats[-1].permute(1, 0, 2).reshape(b, -1, 64, 64)
                frame_vision_feats_temp = frame_vision_feats_temp.reshape(b, -1)

                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                frame_vision_feats_temp = F.normalize(frame_vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, frame_vision_feats_temp.t()).t()

                similarity_scores = F.softmax(similarity_scores, dim=1)
                sampled_indices = torch.multinomial(similarity_scores, num_samples=b, replacement=True).squeeze(
                    1)  # Shape [batch_size, 16]

                memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2),
                                                          memory_stack_ori_new.size(3))

                frame_vision_feats[-1] = self.memory_attention(
                    curr=[frame_vision_feats[-1]],
                    curr_pos=[frame_vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                )

            frame_feats = [feat.permute(1, 2, 0).view(b,-1, *feat_size)
                     for feat, feat_size in zip(frame_vision_feats[::-1], feat_sizes[::-1])][::-1]

            frame_imaged_embed = frame_feats[-1]  # (b,256,16,16)
            frame_high_res_feats = frame_feats[:-1]  # (b,32,64,64), (b,64,32,32)

            if ti == 0:
                with torch.no_grad():
                    if pt is not None:
                        se, de = self.sam_prompt_encoder(  # se b 2 256, de b 256 16 16
                            points=(pt[0][:, ti], pt[1][:, ti]),
                            boxes=None,
                            masks=None,
                            batch_size=b,
                        )
                        flag = True
                    else:
                        se, de = self.sam_prompt_encoder(  # se b 2 256, de b 256 32 32
                            points=None,
                            boxes=None,
                            masks=None,
                            batch_size = b
                        )
                        flag = False
            else:
                se = None
                de = self.learned_prompt_conv(frame_imaged_embed)





            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
                image_embeddings=frame_imaged_embed,
                image_pe=self.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
                repeat_image=False,  # the image is already batched
                high_res_features=frame_high_res_feats
            )  # b c h w

            pred_mask = F.interpolate(low_res_multimasks, imgs.shape[-2:], mode="bilinear", align_corners=False)  # b 1 256 256
            high_res_multimasks = F.interpolate(low_res_multimasks, imgs.shape[-2:],mode="bilinear", align_corners=False)
            frames_pred.append(pred_mask)


            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=frame_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks,
                is_mask_from_pts=flag)

            maskmem_features = maskmem_features
            maskmem_features = maskmem_features.to(device=device, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0]
            maskmem_pos_enc = maskmem_pos_enc.to(device=device, non_blocking=True)

            if len(self.memory_bank_list) < self.mem_bank_size:
                for batch in range(maskmem_features.size(0)):
                    self.memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                             (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                             iou_predictions[batch, 0],
                                             frame_imaged_embed[batch].reshape(-1).detach()])

            else:
                for batch in range(maskmem_features.size(0)):

                    memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in self.memory_bank_list]
                    memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                         memory_bank_maskmem_features_norm.t())

                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores)
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    if similarity_scores[min_similarity_index] < \
                            current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        if iou_predictions[batch, 0] > self.memory_bank_list[max_similarity_index][2] - 0.1:
                            self.memory_bank_list.pop(max_similarity_index)
                            self.memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                                     (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                                     iou_predictions[batch, 0],
                                                     frame_imaged_embed[batch].reshape(-1).detach()])



        pred = torch.stack(frames_pred, dim=1) # b t c h w
        pred_edge = F.interpolate(pre_edge, imgs.shape[-2:], mode="bilinear",
                                  align_corners=False)  # b*t 1 256 256
        pred_edge = pred_edge.view(b,t,-1,h,w) # b t 1 256 256
        return pred,pred_edge

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std

        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes


    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        return maskmem_features, maskmem_pos_enc

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks


    def random_click(self, mask):
        indices = np.argwhere(mask > 0.5)
        indices[:, [0, 1]] = indices[:, [1, 0]]
        point_label = 1
        if len(indices) == 0:
            return None,[point_label]
        pt = indices[np.random.randint(len(indices))]
        return pt[np.newaxis, :], [point_label]

    def fixed_click(self, mask):
        indices = np.argwhere(mask > 0.5)
        indices[:, [0, 1]] = indices[:, [1, 0]]
        point_label = 1
        if len(indices) == 0:
            point_label = 0
            return None, [point_label]
        pt = indices[len(indices) // 2]
        return pt[np.newaxis, :], [point_label]
