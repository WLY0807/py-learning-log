"""公共工具：数据加载等，所有 notebook 复用"""
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 数据缓存固定在这个模块同目录的 data/ 下，只下载一次
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def load_fashion_mnist(batch_size=256):
    os.makedirs(DATA_ROOT, exist_ok=True)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_ds  = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False))