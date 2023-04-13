import os, sys
import copy

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoConfig

from NNDF.networks import NetworkMetadata, Precision
from T5.T5ModelConfig import T5Metadata

from t5_wrapper import OnnxT5EncoderWrapperModule as T5OnnxEncoder
from t5_wrapper import OnnxT5DecoderWrapperModule as T5OnnxDecoder
from t5_wrapper import TRTT5EncoderWrapperModule as T5TRTEncoder
from t5_wrapper import TRTT5DecoderWrapperModule as T5TRTDecoder
from t5_wrapper import T5ForConditionalGenerationWrapper


device = torch.device("cuda:0")

model_name = "t5-base"
tokenizer_path = model_name

onnx_encoder_file_path = "./models/t5-base-encoder.onnx"
onnx_decoder_file_path = "./models/t5-base-decoder.onnx"

trt_encoder_file_path = "./models/t5-base-encoder-fp16.engine"
trt_decoder_file_path = "./models/t5-base-decoder-fp16.engine"

metadata = NetworkMetadata(variant=model_name,
                           precision=Precision(fp16=False),
                           other=T5Metadata(kv_cache=False),
                           )

# load Torch
config = AutoConfig.from_pretrained(model_name)
config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
th_model = T5ForConditionalGeneration.from_pretrained(model_name).eval().to(device)
if metadata.precision.fp16:
    th_model = th_model.half()

# # load ONNX
# onnx_encoder = T5OnnxEncoder(onnx_encoder_file_path, metadata, config)
# onnx_decoder = T5OnnxDecoder(onnx_decoder_file_path, metadata, config)
# lm_head = copy.deepcopy(th_model.lm_head).to("cpu")
# onnx_model = T5ForConditionalGenerationWrapper(onnx_encoder, onnx_decoder, lm_head, config)

# load TensorRT
trt_encoder = T5TRTEncoder(trt_encoder_file_path, config, profile_idx=0, fp16=metadata.precision.fp16)
trt_decoder = T5TRTDecoder(trt_decoder_file_path, config, profile_idx=0, fp16=metadata.precision.fp16)
lm_head = copy.deepcopy(th_model.lm_head)
trt_model = T5ForConditionalGenerationWrapper(trt_encoder, trt_decoder, lm_head, config)


batch_size = 1
input_text = "premise: If I fall asleep then I am going to wake up in 8 hours. hypothesis: I fell asleep but did not wake up in 8 hours."
input = tokenizer([input_text] * batch_size,
                  padding='longest',
                  max_length=512,
                  pad_to_multiple_of=8,
                  truncation=True,
                  return_tensors='pt')
input_ids = input.input_ids.to(device)
attention_mask = input.attention_mask.to(device)

dec_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * config.decoder_start_token_id


with torch.no_grad():
    th_enc_out = th_model.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  )
    # print("th_enc_out: {}, {}".format(type(th_enc_out), th_enc_out.last_hidden_state.shape))
    th_dec_out = th_model.decoder(input_ids=dec_input_ids,
                                  encoder_hidden_states=th_enc_out.last_hidden_state,
                                  encoder_attention_mask=attention_mask,
                                  )
    # print("th_dec_out: {}, {}".format(type(th_dec_out), th_dec_out.last_hidden_state.shape))
    th_head_out = th_model.lm_head(th_dec_out.last_hidden_state * config.d_model ** -0.5)
    # print("th_head_out: {}, {}".format(type(th_head_out), th_head_out.shape))
    th_out = th_model.generate(input_ids,
                               decoder_input_ids=dec_input_ids,
                               max_length=32,
                               min_length=1,
                               num_beams=1,
                               )
    th_text = tokenizer.batch_decode(th_out)
    print("[Torch] {}".format(th_text[0]))


    trt_enc_out = trt_encoder(input_ids=input_ids,
                              attention_mask=attention_mask,
                              )
    # print("trt_enc_out: {} {}".format(type(trt_enc_out), trt_enc_out.last_hidden_state.shape))
    trt_dec_out = trt_decoder(input_ids=dec_input_ids,
                              encoder_hidden_states=trt_enc_out.last_hidden_state,
                              encoder_attention_mask=attention_mask,
                              )
    # print("trt_dec_out: {}, {}".format(type(trt_dec_out), trt_dec_out.last_hidden_state.shape))
    trt_head_out = trt_model.lm_head(trt_dec_out.last_hidden_state * config.d_model ** -0.5)
    # print("trt_head_out: {}, {}".format(type(trt_head_out), trt_head_out.shape))
    trt_out = trt_model.generate(input_ids,
                                 decoder_input_ids=dec_input_ids,
                                 max_length=32,
                                 min_length=1,
                                 num_beams=1,
                                 )
    trt_text = tokenizer.batch_decode(trt_out)
    print("[TRT  ] {}".format(trt_text[0]))


    # input_ids = input_ids.cpu()
    # attention_mask = attention_mask.cpu()
    # dec_input_ids = dec_input_ids.cpu()
    # onnx_enc_out = onnx_encoder(input_ids=input_ids,
    #                             attention_mask=attention_mask,
    #                             )
    # # print("onnx_enc_out: {}, {}".format(type(onnx_enc_out), onnx_enc_out.last_hidden_state.shape))
    # onnx_dec_out = onnx_decoder(input_ids=dec_input_ids,
    #                             encoder_hidden_states=onnx_enc_out.last_hidden_state,
    #                             encoder_attention_mask=attention_mask,
    #                             )
    # # print("onnx_dec_out: {}, {}".format(type(onnx_dec_out), onnx_dec_out.last_hidden_state.shape))
    # onnx_head_out = onnx_model.lm_head(onnx_dec_out.last_hidden_state * config.d_model ** -0.5)
    # # print("onnx_head_out: {}, {}".format(type(onnx_head_out), onnx_head_out.shape))
    # onnx_out = onnx_model.generate(input_ids,
    #                                decoder_input_ids=dec_input_ids,
    #                                max_length=32,
    #                                min_length=1,
    #                                num_beams=1,
    #                                )
    # onnx_text = tokenizer.batch_decode(onnx_out)
    # print("[ONNX ] {}".format(onnx_text[0]))

# onnx_encoder.release()
# onnx_decoder.release()


th_enc_result = th_enc_out.last_hidden_state.cpu()
th_dec_result = th_dec_out.last_hidden_state.cpu()
th_head_result = th_head_out.cpu()
# onnx_enc_result = onnx_enc_out.last_hidden_state.cpu()
# onnx_dec_result = onnx_dec_out.last_hidden_state.cpu()
# onnx_head_result = onnx_head_out.cpu()
trt_enc_result = trt_enc_out.last_hidden_state.cpu()
trt_dec_result = trt_dec_out.last_hidden_state.cpu()
trt_head_result = trt_head_out.cpu()
# print("[Torch]", th_enc_result.shape, th_dec_result.shape, th_head_result.shape)
# print("[ONNX ]", onnx_enc_result.shape, onnx_dec_result.shape, onnx_head_result.shape)
# print("[TRT  ]", trt_enc_result.shape, trt_dec_result.shape, trt_head_result.shape)


def print_diff(enc_diff, dec_diff, head_diff):
    print("enc_diff: max: {:.4f}, avr: {:.4f}".format(enc_diff.max(), enc_diff.mean()))
    res = np.sort(enc_diff.reshape([-1]).numpy(), axis=0)
    for i in [0, 10, 25, 50, 75, 90, 100]:
        print("{}%: {:.4f}".format(i, np.percentile(res, i)), end="\t")
    print()
    print("dec_diff: max: {:.4f}, avr: {:.4f}".format(dec_diff.max(), dec_diff.mean()))
    res = np.sort(dec_diff.reshape([-1]).numpy(), axis=0)
    for i in [0, 10, 25, 50, 75, 90, 100]:
        print("{}%: {:.4f}".format(i, np.percentile(res, i)), end="\t")
    print()
    print("head_diff: max: {:.4f}, avr: {:.4f}".format(head_diff.max(), head_diff.mean()))
    res = np.sort(head_diff.reshape([-1]).numpy(), axis=0)
    for i in [0, 10, 25, 50, 75, 90, 100]:
        print("{}%: {:.4f}".format(i, np.percentile(res, i)), end="\t")
    print()
    print()


print()
# print("Torch vs ONNX")
# enc_diff = abs(onnx_enc_result - th_enc_result)
# dec_diff = abs(onnx_dec_result - th_dec_result)
# head_diff = abs(onnx_head_result - th_head_result)
# print_diff(enc_diff, dec_diff, head_diff)


print("Torch vs TRT")
enc_diff = abs(trt_enc_result - th_enc_result)
dec_diff = abs(trt_dec_result - th_dec_result)
head_diff = abs(trt_head_result - th_head_result)
print_diff(enc_diff, dec_diff, head_diff)
