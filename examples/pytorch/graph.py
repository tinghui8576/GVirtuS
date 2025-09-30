import os
import time
import torch
from torchvision import models
from PIL import Image
print(torch.cuda.is_available())


# model_selectors = [
#     "squeezenet1_1",
#     # "mobilenet_v2",
#     # "resnet18",
#     # "vgg16",
# ]

# model_registry = {
#     "squeezenet1_1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
#     # "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
#     # "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
#     # "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
# }

# device = "cuda" 
# # device="cpu"

# img_dir = "imagenet_test_1000"
# input_shape = (1, 3, 224, 224)   
# dtype = torch.float32 

# for model_selector in model_selectors:
#     if model_selector not in model_registry:
#         print(f"Model '{model_selector}' is not supported, skipping.")
#         continue

#     model_fn, weights = model_registry[model_selector]

#     if model_selector == "inception_v3":
#         model = model_fn(weights=weights, aux_logits=False)
#     else:
#         model = model_fn(weights=weights)


#     model = model.to(device)
#     model.eval()

# # probe_driver.py
# import ctypes as C

# lib = C.CDLL("libcuda.so.1")  # resolves to your GVirtuS lib via LD_LIBRARY_PATH

# def fn(name, restype, *argtypes):
#     f = getattr(lib, name); f.restype = restype; f.argtypes = argtypes; return f

# cuInit   = fn("cuInit", C.c_int, C.c_uint)
# cuDevCnt = fn("cuDeviceGetCount", C.c_int, C.POINTER(C.c_int))
# cuDevGet = fn("cuDeviceGet", C.c_int, C.POINTER(C.c_int), C.c_int)
# cuPCRet  = fn("cuDevicePrimaryCtxRetain", C.c_int, C.POINTER(C.c_void_p), C.c_int)
# cuCtxSet = fn("cuCtxSetCurrent", C.c_int, C.c_void_p)
# cuStrmCr = fn("cuStreamCreate", C.c_int, C.POINTER(C.c_void_p), C.c_uint)
# cuErrStr = fn("cuGetErrorName", C.c_int, C.c_int, C.POINTER(C.c_char_p))

# def show(name, code):
#     s = C.c_char_p()
#     if cuErrStr(code, C.byref(s)) == 0 and s.value:
#         print(f"{name}: {code} ({s.value.decode()})")
#     else:
#         print(f"{name}: {code}")

# print("cuInit ->", cuInit(0))
# n = C.c_int(); show("cuDeviceGetCount", cuDevCnt(C.byref(n))); print("count=", n.value)
# if n.value > 0:
#     dev = C.c_int(); show("cuDeviceGet", cuDevGet(C.byref(dev), 0))
#     ctx = C.c_void_p(); show("cuDevicePrimaryCtxRetain", cuPCRet(C.byref(ctx), dev.value)); print("ctx=", ctx.value)
#     show("cuCtxSetCurrent", cuCtxSet(ctx))
#     st = C.c_void_p(); show("cuStreamCreate", cuStrmCr(C.byref(st), 0)); print("stream=", st.value)
