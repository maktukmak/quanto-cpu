
import torch 
from quanto import Calibration, freeze, quantize

# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin1 = torch.nn.Linear(256, 32)
#         self.lin2 = torch.nn.Linear(32, 32)

#     def forward(self, x):
#         x = self.lin1(x)
#         return self.lin2(x)
# model = Model()
# cal_samples = torch.randn([32, 256])
# opt_model = torch.compile(model)
# print(opt_model(cal_samples))

def foo1(x1, x2):
    a = torch.neg(x1)
    b = torch.maximum(x2, a)
    y = torch.cat([b], dim=0)
    return y

x1 = torch.randint(256, (1, 8), dtype=torch.uint8)
x2 = torch.randint(256, (8390, 8), dtype=torch.uint8)

compiled_foo1 = torch.compile(foo1)
result = compiled_foo1(x1, x2)


# from torch.profiler import profile, schedule, ProfilerActivity
# RESULT_DIR = "./torch_prof"
# my_schedule = schedule(
#     skip_first=10,
#     wait=5,
#     warmup=5,
#     active=1,
#     repeat=5)

# def trace_handler(p):
#     output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
#     print(output)
#     p.export_chrome_trace(f"{RESULT_DIR}/{p.step_num}.json")

# for _ in range(10):
#     model(cal_samples)  # compiled_model(**input_dict) to get inductor model profiling

# total = 0
# with profile(
#     activities=[ProfilerActivity.CPU],
#     schedule=my_schedule,
#     on_trace_ready=trace_handler
# ) as p:
#     for _ in range(50):
#         model(cal_samples)  # compiled_model(**input_dict) to get inductor model profiling
#         p.step()