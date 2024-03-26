import timeit

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from torch.utils.cpp_extension import load
import sys


#import cpu_ops
# int_mm = load(name="cpu_mm", 
#                 sources=["/media/MERCURY/mehmet/projects/quanto/quanto/library/cpu/cpu_mm_mkl.cpp"],
#                 extra_ldflags = ["-lmkl_intel_ilp64", "-lmkl_intel_thread", "-lmkl_core", "-liomp5"],
#                 verbose=True)


A = torch.randint(127, 128, [3488, 512]).type(torch.int8)
B = torch.randint(-127, 128, [1024, 512]).type(torch.int8).t()

A = A.to('cuda')
B = B.to('cuda')
res_cuda = torch._int_mm(A, B)

print(res_cuda)

A = A.to('cpu')
B = B.to('cpu')
res_cpu = torch._int_mm(A, B)
#res_cpu = int_mm._int_mm(A, B)

print(res_cpu)
print(torch.allclose(res_cpu, res_cuda.to('cpu')))