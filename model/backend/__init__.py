from __future__ import annotations

from typing import List
import numpy as np

from datatype import ModelBenchBatchLayerData

backends = ['nart_trt', 'trtexec', 'onnxruntime']


def get_backend(name: str) -> _BaseBackend:
    if name == 'nart_trt':
        from .nart import Nart_TRT
        return Nart_TRT
    elif name == 'trtexec':
        from .trtexec import Trtexec
        return Trtexec
    elif name == 'onnxruntime':
        from .onnxruntime import ONNXRuntimeBackend
        return ONNXRuntimeBackend


class _BaseBackend():
    supported = ['e2e_prof', 'layer_prof']

    def __init__(self, ctx, onnx_model: str, batch_size_list: list, backend_options: str) -> None:
        raise NotImplementedError

    def version_info(self) -> str:
        raise NotImplementedError

    def prepare(self) -> None:
        raise NotImplementedError

    def pre_batch_run(self, batch_size: int) -> None:
        raise NotImplementedError

    def time_run(self, repeat: int = 10, warm_up: int = 3) -> np.ndarray:
        raise NotImplementedError

    def layer_prof(self, batch_size: int) -> List[ModelBenchBatchLayerData]:
        raise NotImplementedError
