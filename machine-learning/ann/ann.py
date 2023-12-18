from ctypes import CDLL, Array, c_bool, c_char_p, c_int, c_ulong, c_void_p
from os.path import exists
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    CDLL("libmali.so")  # fail if libmali.so is not mounted into container
    libann = CDLL("libann.so")
    libann.init.argtypes = c_int, c_int, c_char_p
    libann.init.restype = c_void_p
    libann.load.argtypes = c_void_p, c_char_p, c_bool, c_bool, c_bool, c_char_p
    libann.load.restype = c_int
    libann.execute.argtypes = c_void_p, c_int, Array[c_void_p], Array[c_void_p]
    libann.unload.argtypes = c_void_p, c_int
    libann.destroy.argtypes = (c_void_p,)
    libann.shape.argtypes = c_void_p, c_int, c_bool, c_int
    libann.shape.restype = c_ulong
    libann.tensors.argtypes = c_void_p, c_int, c_bool
    libann.tensors.restype = c_int
    is_available = True
except OSError as e:
    print("Could not load ANN shared libraries, using ONNX:", e)
    is_available = False


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            obj = super(_Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = obj
        else:
            obj = cls._instances[cls]
            obj.new()
        return obj


class Ann(metaclass=_Singleton):
    def __init__(self, log_level=3, tuning_level=1, tuning_file: str = None) -> None:
        if not is_available:
            return
        if tuning_file and not exists(tuning_file):
            raise ValueError("tuning_file must point to an existing (possibly empty) file!")
        if tuning_level == 0 and tuning_file is None:
            raise ValueError("tuning_level == 0 reads existing tuning information and requires a tuning_file")
        if tuning_level < 0 or tuning_level > 3:
            raise ValueError("tuning_level must be 0 (load from tuning_file), 1, 2 or 3.")
        if log_level < 0 or log_level > 5:
            raise ValueError("log_level must be 0 (trace), 1 (debug), 2 (info), 3 (warning), 4 (error) or 5 (fatal)")
        self.log_level = log_level
        self.tuning_level = tuning_level
        self.tuning_file = tuning_file
        self.output_shapes: Dict[int, Tuple[Tuple[int, ...]]] = {}
        self.input_shapes: Dict[int, Tuple[Tuple[int, ...]]] = {}
        self.new()

    def new(self):
        if not hasattr(self, "ann"):
            self.ann = libann.init(
                self.log_level,
                self.tuning_level,
                self.tuning_file.encode("utf-8") if self.tuning_file else None,
            )
            self.ref_count = 0

        self.ref_count += 1

    def destroy(self) -> None:
        self.ref_count -= 1
        if self.ref_count <= 0 and hasattr(self, "ann"):
            libann.destroy(self.ann)

    def __del__(self) -> None:
        self.destroy()

    def load(
        self,
        model_path: str,
        fast_math=True,
        fp16=False,
        save_cached_network=False,
        cached_network_path: str = None,
    ) -> int:
        if not (exists(model_path) and model_path.endswith((".armnn", ".tflite", ".onnx"))):
            raise ValueError("model_path must be a file with extension .armnn, .tflite or .onnx")
        if cached_network_path and not exists(cached_network_path):
            raise ValueError("cached_network_path must point to an existing (possibly empty) file!")
        if save_cached_network and cached_network_path is None:
            raise ValueError("save_cached_network is True, cached_network_path must be specified!")
        net_id = libann.load(
            self.ann,
            model_path.encode("utf-8"),
            fast_math,
            fp16,
            save_cached_network,
            cached_network_path.encode("utf-8") if cached_network_path else None,
        )

        self.input_shapes[net_id] = tuple(
            self.shape(net_id, input=True, index=i) for i in range(self.tensors(net_id, input=True))
        )
        self.output_shapes[net_id] = tuple(
            self.shape(net_id, input=False, index=i) for i in range(self.tensors(net_id, input=False))
        )
        return net_id

    def unload(self, network_id: int) -> None:
        libann.unload(self.ann, network_id)
        del self.output_shapes[network_id]

    def execute(self, network_id: int, input_tensors: List[NDArray]) -> List[NDArray]:
        if not isinstance(input_tensors, list):
            raise ValueError("input_tensors needs to be a list!")
        net_input_shapes = self.input_shapes[network_id]
        if len(input_tensors) != len(net_input_shapes):
            raise ValueError(f"input_tensors lengths {len(input_tensors)} != network inputs {len(net_input_shapes)}")
        for net_input_shape, input_tensor in zip(net_input_shapes, input_tensors):
            if net_input_shape != input_tensor.shape:
                raise ValueError(f"input_tensor shape {input_tensor.shape} != network input shape {net_input_shape}")
        output_tensors = [np.ndarray(s, dtype=np.float32) for s in self.output_shapes[network_id]]
        input_type = c_void_p * len(input_tensors)
        inputs = input_type(*[t.ctypes.data_as(c_void_p) for t in input_tensors])
        output_type = c_void_p * len(output_tensors)
        outputs = output_type(*[t.ctypes.data_as(c_void_p) for t in output_tensors])
        libann.execute(self.ann, network_id, inputs, outputs)
        return output_tensors

    def shape(self, network_id: int, input=False, index=0) -> Tuple[int]:
        s = libann.shape(self.ann, network_id, input, index)
        a = []
        while s != 0:
            a.append(s & 0xFFFF)
            s >>= 16
        return tuple(a)

    def tensors(self, network_id: int, input=False) -> int:
        return libann.tensors(self.ann, network_id, input)
