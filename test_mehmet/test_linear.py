
import torch 
from quanto import Calibration, freeze, quantize
import intel_extension_for_pytorch as ipex

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(256, 32)
        self.lin2 = torch.nn.Linear(32, 32)

    def forward(self, x):
        x = self.lin1(x)
        return self.lin2(x)


model = Model()

cal_samples = torch.randn([32, 256])

with Calibration():
    model(cal_samples)

quantize(model, weights=torch.int8, activations=torch.int8)

model(cal_samples)

freeze(model)
model.eval()

out = model(cal_samples)

model = ipex.optimize(model, weights_prepack=False)

model = torch.compile(model, backend="ipex")


from torch.profiler import profile, schedule, ProfilerActivity
RESULT_DIR = "./prof_trace"
my_schedule = schedule(
    skip_first=1,
    wait=1,
    warmup=5,
    active=1,
    repeat=1)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")#, row_limit=20)
    print(output)
    p.export_chrome_trace(f"{RESULT_DIR}/{p.step_num}.json")

for _ in range(3):
    model(cal_samples)  # compiled_model(**input_dict) to get inductor model profiling

total = 0
with profile(
    activities=[ProfilerActivity.CPU],
    schedule=my_schedule,
    on_trace_ready=trace_handler
) as p:
    for _ in range(10):
        model(cal_samples)  # compiled_model(**input_dict) to get inductor model profiling
        p.step()