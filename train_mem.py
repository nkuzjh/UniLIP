import torch


a = torch.ones((1024 * 3000000)).to("cuda:0")
b = torch.ones((1024 * 3000000)).to("cuda:0")

while True:
    c = a+b
    torch.cuda.empty_cache()