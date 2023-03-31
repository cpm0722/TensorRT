import os, sys

import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoConfig
from polygraphy.backend.trt import Profile

from NNDF.networks import NetworkMetadata, Precision
from T5.T5ModelConfig import T5Metadata
from T5.export import T5EncoderConverter, T5DecoderConverter, T5EncoderONNXFile, T5DecoderONNXFile


os.makedirs("tmp", exist_ok=True)
th_model_path = "T5/wd-t5-base/T5/t5-base/T5-base/pytorch_model"
model_name = "t5-base"
onnx_encoder_path = "tmp/{}_encoder.onnx".format(model_name)
onnx_decoder_path = "tmp/{}_decoder.onnx".format(model_name)
trt_encoder_path = "tmp/{}_encoder.engine".format(model_name)
trt_decoder_path = "tmp/{}_decoder.engine".format(model_name)

model = T5ForConditionalGeneration.from_pretrained(th_model_path).eval()
config = AutoConfig.from_pretrained(th_model_path)

metadata = NetworkMetadata(variant=model_name, precision=Precision(fp16=False), other=T5Metadata(kv_cache=False))

# Convert Torch -> ONNX
T5EncoderConverter().torch_to_onnx(onnx_encoder_path, model, metadata)
T5DecoderConverter().torch_to_onnx(onnx_decoder_path, model, metadata)


# Convert ONNX -> TRT
batch_size = 32
num_beams = 1
max_input_length = config.d_model
max_output_length = config.d_model
opt_input_seq_len = max_input_length // 2
opt_output_seq_len = max_output_length // 2

encoder_profiles = [
    Profile().add(
        "input_ids",
        min=(batch_size, 1),
        opt=(batch_size, opt_input_seq_len),
        max=(batch_size, max_input_length),
    ).add(
        "attention_mask",
        min=(batch_size, 1),
        opt=(batch_size, opt_input_seq_len),
        max=(batch_size, max_input_length),
    )
]

decoder_profiles = [
    Profile().add(
        "input_ids",
        min=(batch_size * num_beams, 1),
        opt=(batch_size * num_beams, opt_output_seq_len),
        max=(batch_size * num_beams, max_output_length),
    ).add(
        "encoder_hidden_states",
        min=(batch_size * num_beams, 1, config.d_model),
        opt=(batch_size * num_beams, opt_input_seq_len, config.d_model),
        max=(batch_size * num_beams, max_input_length, config.d_model),
    ).add(
        "encoder_attention_mask",
        min=(batch_size * num_beams, 1),
        opt=(batch_size * num_beams, opt_input_seq_len),
        max=(batch_size * num_beams, max_input_length),
    )
]

encoder_engine = T5EncoderONNXFile(
    onnx_encoder_path,
    metadata,
).as_trt_engine(
    trt_encoder_path,
    profiles=encoder_profiles,
    preview_features=[],
)

decoder_engine = T5DecoderONNXFile(
    onnx_decoder_path,
    metadata,
).as_trt_engine(
    trt_decoder_path,
    profiles=decoder_profiles,
    preview_features=[],
)

trt_encoder = T5TRTEncoder(
    encoder_engine,
    metadata,
    config,
    batch_size=batch_size,
)

trt_decoder = T5TRTDecoder(
    decoder_engine,
    metadata,
    config,
    batch_size=batch_size,
    num_beams=num_beams,
)

print(type(trt_encoder))
print(type(trt_decoder))
