from .poisoned_dataset import CIFAR10Poison, MNISTPoison
from torchvision import datasets, transforms
import torch 
import os 

"""
datasets.MNIST 和 datasets.CIFAR10 是 torchvision 提供的标准数据集类。调用时，分别生成相应的训练和测试数据集，根目录由 dataset_path 参数指定。
它们分别返回 torchvision.datasets.MNIST 和 torchvision.datasets.CIFAR10 类型的对象，这些对象都继承自 torch.utils.data.Dataset，因此可以在 PyTorch 中与数据加载器 (DataLoader) 配合使用
download=download 控制是否在数据集不存在时进行下载。
数据集的文件将会下载到或者加载自这个目录。如果数据集已经存在于这个目录中，torchvision 会直接加载数据集；如果没有，且 download=True 的话，程序会从互联网上下载数据并保存到指定的 root 目录。
"""
def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download) 
    return train_data, test_data 


def build_poisoned_training_set(is_train, args):  #is_train: 一个布尔值，指示是否构建训练集（True）还是测试集（False）。args: 一个包含多个参数的对象（通常是 argparse.Namespace 类型）
    transform, detransform = build_transform(args.dataset) #调用了 build_transform 函数，根据数据集类型（如 CIFAR10 或 MNIST）构建图像数据的转换操作（例如图像的标准化、缩放等）。transform 用于对输入数据进行处理，detransform 可以用来逆转该处理过程。他们都是torchvision.transforms.Compose 类型的对象
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        trainset = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform) #使用自定义的 CIFAR10Poison 类来加载训练集或测试集，并应用预处理的 transform。数据集的类别数量设置为 10（因为 CIFAR10 有 10 个分类）。
        nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform) #使用 MNISTPoison 类来加载 MNIST 数据集，类别数量也设置为 10（MNIST 也有 10 个分类）。
        nb_classes = 10
    else:
        raise NotImplementedError() #如果指定了其他数据集，则会抛出 NotImplementedError()，提示尚未实现对该数据集的支持。

    assert nb_classes == args.nb_classes #这一步用于确保加载的数据集中的类别数量与 args.nb_classes 中的指定值一致。如果不一致，程序会抛出一个 AssertionError，防止由于参数不一致导致的错误。
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes


def build_testset(is_train, args):
    transform, detransform = build_transform(args.dataset)
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned


"""
根据数据集类型，函数会选择用于归一化的均值和标准差。
对于 CIFAR10：它是一个 RGB（彩色）图像数据集，因此均值和标准差有 3 个值，分别对应红色、绿色和蓝色通道的值 (0.5, 0.5, 0.5)。
对于 MNIST：它是一个灰度图像数据集，因此均值和标准差只有 1 个值 (0.5,)，适用于单个通道。
"""
def build_transform(dataset):
    if dataset == "CIFAR10":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "MNIST":
        mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(), #transforms.ToTensor()：将图像从 PIL 或 NumPy 格式转换为 torch.Tensor，并简单将像素值缩放到 [0, 1] 范围，并不归一化。
        transforms.Normalize(mean, std) #根据指定的均值和标准差对图像进行归一化处理。归一化操作是通过将每个通道的像素值减去均值并除以标准差来完成的，使输入数据有零均值和单位方差，即输入数据的分布尽量对称，数值范围合理。这有助于在训练神经网络时稳定数值范围。
        ])
    #这部分代码用于构建逆归一化操作，即将已经归一化的图像转换回原始的像素范围。
    mean = torch.as_tensor(mean) #将均值和标准差转换为 torch.Tensor 格式
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # 该操作用于逆归一化，将归一化的像素值转换回原始像素值的公式。归一化公式是 x' = (x - mean) / std，逆操作是 x = x' * std + mean，通过这个公式我们可以将图像恢复到原始状态。
    
    return transform, detransform #返回torchvision.transforms.Compose 类型的对象。Compose 是一个用于组合多个图像变换操作的类，它将一系列变换函数组合起来按顺序执行。
