import torch
if torch.cuda.is_available():
    print("cuda works with torch")
else:
    print("torch could not find cuda")
