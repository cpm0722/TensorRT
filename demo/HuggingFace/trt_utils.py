#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict
import copy
import ctypes
from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
from cuda import cuda, cudart


numpy_to_torch_dtype_dict = {
    np.dtype(np.bool)       : torch.bool,
    np.dtype(np.uint8)      : torch.uint8,
    np.dtype(np.int8)       : torch.int8,
    np.dtype(np.int16)      : torch.int16,
    np.dtype(np.int32)      : torch.int32,
    np.dtype(np.int64)      : torch.int64,
    np.dtype(np.float16)    : torch.float16,
    np.dtype(np.float32)    : torch.float32,
    np.dtype(np.float64)    : torch.float64,
    np.dtype(np.complex64)  : torch.complex64,
    np.dtype(np.complex128) : torch.complex128
}


torch_to_numpy_dtype_dict = {
    torch.bool       : np.dtype(np.bool),
    torch.uint8      : np.dtype(np.uint8),
    torch.int8       : np.dtype(np.int8),
    torch.int16      : np.dtype(np.int16),
    torch.int32      : np.dtype(np.int32),
    torch.int64      : np.dtype(np.int64),
    torch.float16    : np.dtype(np.float16),
    torch.float32    : np.dtype(np.float32),
    torch.float64    : np.dtype(np.float64),
    torch.complex64  : np.dtype(np.complex64),
    torch.complex128 : np.dtype(np.complex128)
}


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


class TorchDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, shape: torch.Size, dtype: np.dtype):
        self._shape = tuple(shape)
        self._nbytes = trt.volume(shape) * dtype.itemsize
        self._dtype = numpy_to_torch_dtype_dict[dtype]
        self._device_tensor = torch.zeros(self.shape, dtype=self.dtype, device="cuda")
        self._device = self._device_tensor.data_ptr()

    @property
    def dtype(self) -> int:
        return self._dtype

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def device(self) -> int:
        return self._device

    def __str__(self):
        return f"device:\n{self.device}\ndtype:\n{self.dtype}\nshape:\n{self.shape}\nnbytes:\n{self.nbytes}\ntensor:\n{self._device_tensor}\n"

    def __repr__(self):
        return self.__str__()

    @property
    def tensor(self):
        return copy.deepcopy(self._device_tensor)


class TRTModel:
    def __init__(self, engine_file_path, profile_idx=None, output_shape=None):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine = self._load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()

        self.stream = cuda_call(cudart.cudaStreamCreate())
        self.inputs, self.outputs, self.bindings = self._allocate_buffer(profile_idx, output_shape)


    def _load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        return self.engine


    def _allocate_buffer(self, profile_idx=None, output_shape=None):
        inputs = []
        outputs = []
        bindings = []
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]        
        # Setup I/O bindings
        for binding in tensor_names:
            # Get input/output shape
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                shape = self.engine.get_tensor_shape(binding) if profile_idx is None else self.engine.get_tensor_profile_shape(binding, profile_idx)[-1]
            else:
                shape = trt.Dims(output_shape[binding])  # Change it according to max output length
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(f"Binding {binding} has dynamic shape, " +\
                    "but no profile was specified.")
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.engine.max_batch_size
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(binding)))

            # Allocate host and device buffers
            binding_memory = HostDeviceMem(size, dtype)

            # Append the device buffer to device bindings
            bindings.append(int(binding_memory.device))

            # Append to the appropriate list
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append(binding_memory)
            else:
                outputs.append(binding_memory)

        return inputs, outputs, bindings

    # Wrapper for cudaMemcpy which infers copy size and does error checking
    def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
        nbytes = host_arr.size * host_arr.itemsize
        cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


    # Wrapper for cudaMemcpy which infers copy size and does error checking
    def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
        nbytes = host_arr.size * host_arr.itemsize
        cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


    def infer(self, data):
        for input_name in data:
            self.inputs[self.engine[input_name]].host = data[input_name].cpu().numpy()
            self.context.set_input_shape(input_name, data[input_name].shape)

        [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
        [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)) for out in self.outputs]

        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return [out.host for out in self.outputs]


    # def _allocate_buffer(self, profile_idx=None, output_shape=None):
    #     inputs = []
    #     outputs = []
    #     bindings = []
    #     tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]        
    #     # Setup I/O bindings
    #     for binding in tensor_names:
    #         # Get input/output shape
    #         if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
    #             shape = self.engine.get_tensor_shape(binding) if profile_idx is None else self.engine.get_tensor_profile_shape(binding, profile_idx)[-1]
    #         else:
    #             shape = trt.Dims(output_shape[binding])  # Change it according to max output length
    #         shape_valid = np.all([s >= 0 for s in shape])
    #         if not shape_valid and profile_idx is None:
    #             raise ValueError(f"Binding {binding} has dynamic shape, " +\
    #                 "but no profile was specified.")
    #         dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(binding)))
    #
    #         binding_memory = TorchDeviceMem(shape, dtype)
    #         print(binding, shape, dtype)
    #         bindings.append(int(binding_memory.device))
    #
    #         if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
    #             inputs.append(binding_memory)
    #         else:
    #             outputs.append(binding_memory)
    #
    #     return inputs, outputs, bindings
    #
    #
    # def infer(self, data):
    #     for input_name in data:
    #         itemsize = torch_to_numpy_dtype_dict[data[input_name].dtype].itemsize
    #         nbytes = itemsize * trt.volume(data[input_name].shape)
    #         cuda_call(cudart.cudaMemcpyAsync(self.inputs[self.engine[input_name]].device,
    #                                          data[input_name].data_ptr(),
    #                                          nbytes,
    #                                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
    #                                          self.stream,
    #                                          ))
    #         self.context.set_input_shape(input_name, data[input_name].shape)
    #
    #     self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
    #     cuda_call(cudart.cudaStreamSynchronize(self.stream))
    #     return [out.tensor for out in self.outputs]


class T5TRTEncoder(TRTModel, nn.Module):
    def __init__(self, engine_path, hf_config, profile_idx=None, batch_size=1, output_shape=None, **kwargs):
        nn.Module.__init__(self)
        if output_shape is None:
            output_shape = OrderedDict({
                'hidden_states': (batch_size, hf_config.n_positions, hf_config.d_model),
            })
        else:
            for v in output_shape.values():
                assert batch_size == v[0]
        TRTModel.__init__(self, engine_path, profile_idx=profile_idx, output_shape=output_shape)
        self.output_shape = output_shape
        self.batch_size = batch_size

    def infer(self, data):
        out = super().infer(data)
        return [torch.from_numpy(array).reshape(shape).cuda() for array, shape in zip(out, self.output_shape.values())]


class T5TRTDecoder(TRTModel, nn.Module):
    def __init__(self, engine_path, hf_config, profile_idx=None, batch_size=1, output_shape=None, **kwargs):
        nn.Module.__init__(self)
        if output_shape is None:
            output_shape = OrderedDict({
                'hidden_states': (batch_size, hf_config.n_positions, hf_config.d_model),
            })
        else:
            for v in output_shape.values():
                assert batch_size == v[0]
        TRTModel.__init__(self, engine_path, profile_idx=profile_idx, output_shape=output_shape)
        self.output_shape = output_shape
        self.batch_size = batch_size

    def infer(self, data):
        out = super().infer(data)
        return [torch.from_numpy(array).reshape(shape).cuda() for array, shape in zip(out, self.output_shape.values())]
