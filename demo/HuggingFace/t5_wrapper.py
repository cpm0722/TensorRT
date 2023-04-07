import os, sys
import copy

import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from T5.onnxrt import T5OnnxEncoder, T5OnnxDecoder
from trt_utils import T5TRTEncoder, T5TRTDecoder


class OnnxT5EncoderWrapperModule(T5OnnxEncoder, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        T5OnnxEncoder.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)
        self.main_input_name = "input_ids"

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
        T5OnnxDecoder.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)
        self.main_input_name = "input_ids"

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
    def __init__(self, hf_config, *args, fp16=False, **kwargs):
        T5TRTEncoder.__init__(self, hf_config, *args, fp16=fp16, **kwargs)
        torch.nn.Module.__init__(self)
        self.main_input_name = "input_ids"

    def forward(self,
                input_ids,
                attention_mask,
                *args,
                **kwargs,
                ):
        data = {
            'input_ids': input_ids.type(torch.int32),
            'attention_mask': attention_mask.type(torch.int32),
        }
        out = super().infer(data)
        last_hidden_state = out[0][:, :input_ids.shape[1], :]
        # last_hidden_state = torch.from_numpy(out[0].reshape(1, 512, 768)[:, :input_ids.shape[1], :]).to("cuda")
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
        )


class TRTT5DecoderWrapperModule(T5TRTDecoder, torch.nn.Module):
    def __init__(self, hf_config, *args, fp16=False, **kwargs):
        T5TRTDecoder.__init__(self, hf_config, *args, fp16=fp16, **kwargs)
        torch.nn.Module.__init__(self)
        self.main_input_name = "input_ids"

    def forward(self,
                input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                *args,
                **kwargs,
                ):
        device = encoder_hidden_states.device
        data = {
            'input_ids': input_ids.type(torch.int32),
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': encoder_attention_mask.type(torch.int32),
        }
        out = super().infer(data)
        last_hidden_state = out[0][:, :input_ids.shape[1], :]
        # last_hidden_state = torch.from_numpy(out[0].reshape(1, 512, 768)[:, :input_ids.shape[1], :]).to("cuda")
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
        )


class T5ForConditionalGenerationWrapper(T5ForConditionalGeneration):
    def __init__(self, encoder, decoder, lm_head, config):
        super(T5ForConditionalGenerationWrapper, self).__init__(config)
        self.encoder = encoder
        self.decoder = decoder
        self.lm_head = lm_head
