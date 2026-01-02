import torch
try:
    print(f"Success! Torch version {torch.__version__} loaded.")
    x = torch.rand(5, 3)
    print(x)
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
