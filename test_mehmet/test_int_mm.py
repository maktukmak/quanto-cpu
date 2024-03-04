import timeit

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import intel_extension_for_pytorch as ipex

print(torch.__config__.show())

torch.compiler.list_backends()

def mm(a, b):
    return torch._int_mm(a, b)

A = torch.randint(1, 10, [2400, 2400]).type(torch.int8).to('cpu')
B = torch.randint(1, 10, [2400, 2400]).type(torch.int8).to('cpu')
it = 1
o_cpu = A @ B
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        A @ B
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))



# Warmup (slow)
A = A.to('cuda')
B = B.to('cuda')
o_gpu = mm(A, B)







# Get a reference
print(timeit.Timer(lambda: mm(A, B)).timeit(it) / it)

cmm = torch.compile(mm, backend="ipex")
# First invocation will trigger the actual compilation
cmm(A, B)
# Now compare execution time
print(timeit.Timer(lambda: cmm(A, B)).timeit(it) / it)
