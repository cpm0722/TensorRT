import os, sys
import copy

import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from T5.onnxrt import T5OnnxEncoder, T5OnnxDecoder
from T5.trt import T5TRTEncoder, T5TRTDecoder


class OnnxT5EncoderWrapperModule(T5OnnxEncoder, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(OnnxT5EncoderWrapperModule, self).__init__(*args, **kwargs)

    def forward(self,
                input_ids,
                attention_mask,
                *args,
                **kwargs,
                ):
        out = super().forward(copy.deepcopy(input_ids),
                              copy.deepcopy(attention_mask),
                              *args,
                              **kwargs,
                              )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=copy.deepcopy(out),
        )


class OnnxT5DecoderWrapperModule(T5OnnxDecoder, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(OnnxT5DecoderWrapperModule, self).__init__(*args, **kwargs)

    def forward(self,
                input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                *args,
                **kwargs,
                ):
        out = super().forward(copy.deepcopy(input_ids),
                              copy.deepcopy(encoder_hidden_states),
                              copy.deepcopy(encoder_attention_mask),
                              *args,
                              **kwargs,
                              )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=copy.deepcopy(out.last_hidden_state),
        )


class TRTT5EncoderWrapperModule(T5TRTEncoder, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TRTT5EncoderWrapperModule, self).__init__(*args, **kwargs)

    def forward(self,
                input_ids,
                attention_mask,
                *args,
                **kwargs,
                ):
        out = super().forward(copy.deepcopy(input_ids),
                              copy.deepcopy(attention_mask),
                              *args,
                              **kwargs,
                              )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=copy.deepcopy(out),
        )


class TRTT5DecoderWrapperModule(T5TRTDecoder, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TRTT5DecoderWrapperModule, self).__init__(*args, **kwargs)

    def forward(self,
                input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                *args,
                **kwargs,
                ):
        out = super().forward(copy.deepcopy(input_ids),
                              copy.deepcopy(encoder_hidden_states),
                              copy.deepcopy(encoder_attention_mask),
                              *args,
                              **kwargs,
                              )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=copy.deepcopy(out.last_hidden_state),
        )


class T5ForConditionalGenerationWrapper(T5ForConditionalGeneration):
    def __init__(self, encoder, decoder, lm_head, config):
        super(T5ForConditionalGenerationWrapper, self).__init__(config)
        self.encoder = encoder
        self.decoder = decoder
        self.lm_head = lm_head
