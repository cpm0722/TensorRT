import os, sys
import time
import argparse

import torch
import numpy as np
from transformers import AutoConfig

import onnx
import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX models to TensorRT')

    # Model path
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--model_name', type=str, default='t5-base')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=1)

    # TensorRT engine params
    parser.add_argument('--fp16', dest='fp16', action='store_true')
    parser.set_defaults(fp16=False)

    return parser.parse_args()


def build_input_tensor_shapes(args):
    config = AutoConfig.from_pretrained(args.model_name)

    encoder_input_tensors = {
        'input_ids': {
            'min': (args.batch_size, 1),
            'opt': (args.batch_size, config.n_positions//2),
            'max': (args.batch_size, config.n_positions),
        },
        'attention_mask': {
            'min': (args.batch_size, 1),
            'opt': (args.batch_size, config.n_positions//2),
            'max': (args.batch_size, config.n_positions),
        },
    }

    decoder_input_tensors = {
        'input_ids': {
            'min': (args.batch_size * args.num_beams, 1),
            'opt': (args.batch_size * args.num_beams, config.n_positions//2),
            'max': (args.batch_size * args.num_beams, config.n_positions),
        },
        'encoder_hidden_states': {
            'min': (args.batch_size * args.num_beams, 1, config.d_model),
            'opt': (args.batch_size * args.num_beams, config.n_positions//2, config.d_model),
            'max': (args.batch_size * args.num_beams, config.n_positions, config.d_model),
        },
        'encoder_attention_mask': {
            'min': (args.batch_size, 1),
            'opt': (args.batch_size, config.n_positions//2),
            'max': (args.batch_size, config.n_positions),
        },
    }
    return encoder_input_tensors, decoder_input_tensors


def build_engine(args, onnx_file_path, trt_file_path, input_tensors):
    # Builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()

    if args.fp16 is True:
        config.set_flag(trt.BuilderFlag.FP16)

    # Onnx parser
    parser = trt.OnnxParser(network, logger)
    assert os.path.exists(onnx_file_path)
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing {} file!".format(onnx_file_path))
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(1)

    for name, shapes in input_tensors.items():
        profile.set_shape(name,
                          shapes['min'],
                          shapes['opt'],
                          shapes['max'],
                          )
        config.add_optimization_profile(profile)

    # Write engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine == None:
        print("Failed building engine!")
        exit(1)
    with open(trt_file_path, "wb") as f:
        f.write(serialized_engine)


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    encoder_input_tensors, decoder_input_tensors = build_input_tensor_shapes(args)
    # convert encoder
    print("Convert Encoder...")
    build_engine(args,
                 os.path.join(args.save_dir, "{}-encoder.onnx".format(args.model_name)),
                 os.path.join(args.save_dir, "{}-encoder{}.engine".format(args.model_name, "-fp16" if args.fp16 else "")),
                 encoder_input_tensors,
                 )
    # convert decoder
    print("Convert Decoder...")
    build_engine(args,
                 os.path.join(args.save_dir, "{}-decoder.onnx".format(args.model_name)),
                 os.path.join(args.save_dir, "{}-decoder{}.engine".format(args.model_name, "-fp16" if args.fp16 else "")),
                 decoder_input_tensors,
                 )


if __name__ == "__main__":
    main()
