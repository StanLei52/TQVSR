
import sys
sys.path.append(".")
import math
import einops
from typing import Optional, Tuple
from easydict import EasyDict as edict
import torch
from torch import nn

from AQVSR.pytorch_bert.VisualBert import VisualBertConfig, VisualBertModel


def expand_batch_inp3d(tensor, expand_type='q'):
    bsz, _, _ = tensor.shape
    assert expand_type in ['q', 'ctx']
    if expand_type=='q':
        rt = einops.repeat(tensor, 'bsz l d -> bsz b l d', b=bsz)
    else: # type == 'ctx'
        rt = einops.repeat(tensor, 'bsz l d -> b bsz l d', b=bsz)
    
    rt = einops.rearrange(rt, 'bsz b l d -> (bsz b) l d')
    return rt

def expand_batch_inp2d(tensor, expand_type='q'):
    bsz, _ = tensor.shape
    assert expand_type in ['q', 'ctx']
    if expand_type=='q':
        rt = einops.repeat(tensor, 'bsz l -> bsz b l', b=bsz)
    else: # type == 'ctx'
        rt = einops.repeat(tensor, 'bsz l -> b bsz l', b=bsz)
    
    rt = einops.rearrange(rt, 'bsz b l -> (bsz b) l')
    return rt

#model DME
class AQVSR_Bert(nn.Module):
    def __init__(self, QueryEncCfg, CtxEncCfg, loss_type):
        super().__init__()
        self.query_encoder_cfg = QueryEncCfg
        self.ctx_encoder_cfg = CtxEncCfg
        self.loss_type =loss_type
        self.query_encoder = VisualBertModel(QueryEncCfg)
        self.ctx_encoder = VisualBertModel(CtxEncCfg)

    
    def forward(
        self, 
        query_enc_text_ids=None, 
        query_enc_text_attn_mask=None, 
        query_enc_token_type_ids=None, 
        query_enc_position_ids=None,
        query_enc_head_mask=None, 
        query_enc_inputs_embeds=None,
        query_enc_vis_embeds=None, 
        query_enc_vis_attn_mask=None, 
        query_enc_vis_token_type_ids=None,
        query_enc_image_text_alignment=None,
        query_enc_output_attn=None,
        query_enc_output_hidden_states=None,

        ctx_enc_text_ids=None, 
        ctx_enc_text_attn_mask=None, 
        ctx_enc_token_type_ids=None, 
        ctx_enc_position_ids=None,
        ctx_enc_head_mask=None, 
        ctx_enc_inputs_embeds=None,
        ctx_enc_vis_embeds=None, 
        ctx_enc_vis_attn_mask=None, 
        ctx_enc_vis_token_type_ids=None,
        ctx_enc_image_text_alignment=None,
        ctx_enc_output_attn=None,
        ctx_enc_output_hidden_states=None,

        return_dict=True,
        return_embedding=False
    ):  
        query_enc_outp = self.query_encoder(
            input_ids=query_enc_text_ids,
            attention_mask=query_enc_text_attn_mask,
            token_type_ids=query_enc_token_type_ids,
            position_ids=query_enc_position_ids,
            head_mask=query_enc_head_mask,
            inputs_embeds=query_enc_inputs_embeds,
            visual_embeds=query_enc_vis_embeds,
            visual_attention_mask=query_enc_vis_attn_mask,
            visual_token_type_ids=query_enc_vis_token_type_ids,
            image_text_alignment=query_enc_image_text_alignment,
            output_attentions=query_enc_output_attn,
            output_hidden_states=query_enc_output_hidden_states,
            return_dict=return_dict
        )

        ctx_enc_outp = self.ctx_encoder(
            input_ids=ctx_enc_text_ids,
            attention_mask=ctx_enc_text_attn_mask,
            token_type_ids=ctx_enc_token_type_ids,
            position_ids=ctx_enc_position_ids,
            head_mask=ctx_enc_head_mask,
            inputs_embeds=ctx_enc_inputs_embeds,
            visual_embeds=ctx_enc_vis_embeds,
            visual_attention_mask=ctx_enc_vis_attn_mask,
            visual_token_type_ids=ctx_enc_vis_token_type_ids,
            image_text_alignment=ctx_enc_image_text_alignment,
            output_attentions=ctx_enc_output_attn,
            output_hidden_states=ctx_enc_output_hidden_states,
            return_dict=return_dict
        )

        if return_embedding:
            query_enc_outp = query_enc_outp.last_hidden_state[:,0,:]
            ctx_enc_outp = ctx_enc_outp.last_hidden_state[:,0,:]

        return query_enc_outp, ctx_enc_outp

    
    def compute_query(self,
        query_enc_text_ids=None, 
        query_enc_text_attn_mask=None, 
        query_enc_token_type_ids=None, 
        query_enc_position_ids=None,
        query_enc_head_mask=None, 
        query_enc_inputs_embeds=None,
        query_enc_vis_embeds=None, 
        query_enc_vis_attn_mask=None, 
        query_enc_vis_token_type_ids=None,
        query_enc_image_text_alignment=None,
        query_enc_output_attn=None,
        query_enc_output_hidden_states=None,        
        return_dict=True,
        return_embedding=False):
        _query_enc_outp = self.query_encoder(
            input_ids=query_enc_text_ids,
            attention_mask=query_enc_text_attn_mask,
            token_type_ids=query_enc_token_type_ids,
            position_ids=query_enc_position_ids,
            head_mask=query_enc_head_mask,
            inputs_embeds=query_enc_inputs_embeds,
            visual_embeds=query_enc_vis_embeds,
            visual_attention_mask=query_enc_vis_attn_mask,
            visual_token_type_ids=query_enc_vis_token_type_ids,
            image_text_alignment=query_enc_image_text_alignment,
            output_attentions=query_enc_output_attn,
            output_hidden_states=query_enc_output_hidden_states,
            return_dict=return_dict
        )
        if return_embedding:
            _query_enc_outp = _query_enc_outp.last_hidden_state[:,0,:]
        
        return _query_enc_outp

    def compute_ctx(self,         
        ctx_enc_text_ids=None, 
        ctx_enc_text_attn_mask=None, 
        ctx_enc_token_type_ids=None, 
        ctx_enc_position_ids=None,
        ctx_enc_head_mask=None, 
        ctx_enc_inputs_embeds=None,
        ctx_enc_vis_embeds=None, 
        ctx_enc_vis_attn_mask=None, 
        ctx_enc_vis_token_type_ids=None,
        ctx_enc_image_text_alignment=None,
        ctx_enc_output_attn=None,
        ctx_enc_output_hidden_states=None,

        return_dict=True,
        return_embedding=False):
        
        _ctx_enc_outp = self.ctx_encoder(
            input_ids=ctx_enc_text_ids,
            attention_mask=ctx_enc_text_attn_mask,
            token_type_ids=ctx_enc_token_type_ids,
            position_ids=ctx_enc_position_ids,
            head_mask=ctx_enc_head_mask,
            inputs_embeds=ctx_enc_inputs_embeds,
            visual_embeds=ctx_enc_vis_embeds,
            visual_attention_mask=ctx_enc_vis_attn_mask,
            visual_token_type_ids=ctx_enc_vis_token_type_ids,
            image_text_alignment=ctx_enc_image_text_alignment,
            output_attentions=ctx_enc_output_attn,
            output_hidden_states=ctx_enc_output_hidden_states,
            return_dict=return_dict
        )

        if return_embedding:
            _ctx_enc_outp = _ctx_enc_outp.last_hidden_state[:,0,:]

        return _ctx_enc_outp

