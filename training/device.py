import torch


def select_device():
    """依次选择 NVIDIA CUDA、Apple MPS 和 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    """返回适合打印的计算设备说明。"""
    if device.type == "cuda":
        return f"cuda（{torch.cuda.get_device_name(device)}）"
    if device.type == "mps":
        return "mps（Apple Metal）"
    return "cpu"
