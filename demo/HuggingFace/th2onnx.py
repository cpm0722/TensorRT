import os, sys
import argparse

import torch
import numpy as np
from transformers import T5ForConditionalGeneration

from NNDF.networks import NetworkMetadata, Precision
from T5.T5ModelConfig import T5Metadata
from T5.export import T5EncoderConverter, T5DecoderConverter


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Huggingface models to ONNX')

    # Model path
    parser.add_argument('--save_dir', default='./models')
    parser.add_argument('--model_name', default='t5-base')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).eval()
    metadata = NetworkMetadata(variant=args.model_name,
                               precision=Precision(fp16=False),
                               other=T5Metadata(kv_cache=False),
                               )
    # convert encoder
    print("Convert Encoder...")
    T5EncoderConverter().torch_to_onnx(os.path.join(args.save_dir, "{}-encoder.onnx".format(args.model_name)),
                                       model,
                                       metadata,
                                       )
    # convert decoder
    print("Convert Decoder...")
    T5DecoderConverter().torch_to_onnx(os.path.join(args.save_dir, "{}-decoder.onnx".format(args.model_name)),
                                       model,
                                       metadata,
                                       )


if __name__ == "__main__":
    main()
