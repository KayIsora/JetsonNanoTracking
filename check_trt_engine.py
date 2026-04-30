import os
import sys
import faulthandler
faulthandler.enable()

print("python:", sys.executable)
print("pid:", os.getpid())

import tensorrt as trt
print("TensorRT:", trt.__version__)
print("TensorRT file:", trt.__file__)

engine_path = "/home/jetson/Desktop/JetsonNanoTracking/pytracking/pretrained_network/resnet18_vggmconv1/resnet18_vggmconv1_otb_dual_large_fp16.engine"
print("engine_path:", engine_path)
print("engine_size:", os.path.getsize(engine_path))

logger = trt.Logger(trt.Logger.VERBOSE)

with open(engine_path, "rb") as f:
    data = f.read()

print("engine_bytes_read:", len(data))

with trt.Runtime(logger) as runtime:
    print("runtime_created")
    engine = runtime.deserialize_cuda_engine(data)
    print("deserialize_returned:", engine)

print("deserialize_ok:", engine is not None)

if engine is not None:
    print("num_bindings:", engine.num_bindings)
    for i in range(engine.num_bindings):
        print(i, engine.get_binding_name(i), engine.get_binding_shape(i), engine.get_binding_dtype(i))
