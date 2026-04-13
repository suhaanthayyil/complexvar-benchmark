import torch
data = torch.load('data/processed/graphs/skempi_v2/variants/skempi_001037.complex.pt', map_location='cpu', weights_only=False)
print("V2 Edge Dim:", data.edge_attr.shape[1])
data1 = torch.load('data/processed/graphs/skempi/variants/skempi_001037.complex.pt', map_location='cpu', weights_only=False)
print("V1 Edge Dim:", data1.edge_attr.shape[1])
