import torch
import time

def matmul(a, b, log=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    a_t = torch.tensor(a, device=device)
    b_t = torch.tensor(b, device=device)
    result = a_t @ b_t

    if device.type == "cuda":
        result = result.cpu()

    if log:
        print(f"Done in {time.time() - start:.2f}s using device={device}")
    return result.numpy()