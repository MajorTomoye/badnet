import random
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
import os 

class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB') #使用 PIL（Python Imaging Library）打开指定路径的触发器图像文件。将图像转换为 RGB 模式。确保后续的处理是统一的。
        self.trigger_size = trigger_size #触发器的大小（宽度和高度），用于调整触发器图像的尺寸。
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))       #将触发器图像调整为指定的大小（trigger_size x trigger_size）。
        self.trigger_label = trigger_label ##触发器的标签，表示当触发器被添加到图像上时，图像应该被赋予的目标标签（用于数据集“中毒”时的标签篡改）。 
        self.img_width = img_width
        self.img_height = img_height #目标图像的宽度和高度，用于确定触发器在图像上的放置位置。

    def put_trigger(self, img): #该方法在目标图像上放置触发器。
        """
        img.paste(...)：这是PIL 提供的一个方法，用于将一张图像（这里是触发器图像）粘贴到另一张图像的指定位置。
        """
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size)) #位置 (self.img_width - self.trigger_size, self.img_height - self.trigger_size) 指的是触发器被粘贴在目标图像的右下角。
        return img #将被放置了触发器的图像返回。此时，目标图像已经被“中毒”，并且触发器已经成功添加。

class CIFAR10Poison(CIFAR10): #定义了一个名为 CIFAR10Poison 的类，它继承自 torchvision.datasets.CIFAR10，并在其基础上实现了数据集“中毒”（poisoning）机制。

    def __init__(
        self,
        args, #包含多个参数的对象，其中包括触发器相关的参数以及中毒率等
        root: str, #数据集的根目录路径。
        train: bool = True, #表示是否加载训练集。
        transform: Optional[Callable] = None, #用于图像数据的预处理变换（例如转换为张量和归一化）。Optional[X] 是 Union[X, None] 的简写，表示该参数既可以是类型 X，也可以是 None。Callable是一个可调用类型，其中实现了__call__ 回调方法
        target_transform: Optional[Callable] = None, #用于标签的变换。
        download: bool = False, #如果设置为 True，则从互联网下载数据集（如果本地不存在）。
    ) -> None: #-> None 的意思是 该函数不返回任何值
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download) #调用了父类 CIFAR10 的构造函数，完成了基础数据集的初始化。这一步会加载原始的 CIFAR10 数据集，并进行基本设置，如是否是训练集、数据存储路径等。

        self.width, self.height, self.channels = self.__shape_info__() #调用了 __shape_info__() 方法来获取图像的形状信息（宽度、高度、通道数），这些信息将用于之后的触发器生成。

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height) #这里初始化了一个 TriggerHandler 对象，负责处理触发器的生成和应用。触发器的路径、尺寸、标签、图像宽度和高度等信息都是从 args 参数中传入的。
        self.poisoning_rate = args.poisoning_rate if train else 1.0 #设置中毒率（即被触发器修改的样本比例）。对于训练集，使用从 args 中传入的 poisoning_rate；而对于测试集，默认所有样本都被“中毒”（即 poisoning_rate = 1.0）。
        indices = range(len(self.targets)) 
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate)) #随机选择要被“中毒”的样本的索引，根据中毒率，选择一部分样本的索引。这些样本会被修改，即添加触发器和改变标签。
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})") #打印中毒样本的数量和中毒率，便于调试和查看中毒情况。

        """
        疑问：这里随机抽取样本进行中毒操作，并且触发标签是固定的，由输入参数决定而与true label无关，这样可能抽取到不同的true label改成同一target label,并且对于一种true label，不能保证这种true label全部都被改为target label或全都不被改，无法实现论文中将的single attack或者 all to all attack
        而且训练集生成以及中毒测试集生成都要调用这个函数，两次随机投毒的样本都是不固定的，除非毒化率设置为1
        """

    def __shape_info__(self):
        """
        返回 CIFAR10 数据集中每张图片的形状信息，即宽度、高度和通道数。
        通过 self.data.shape[1:] 获取图片的形状，self.data 是加载的 CIFAR10 数据集中的图像数据，通常是一个形状为 (N, H, W, C) 的 NumPy 数组，其中 N 是图片数量，H 是高度，W 是宽度，C 是通道数。
        """
        return self.data.shape[1:]
    
    #定义了如何根据索引获取数据。它会返回指定索引的图像及其对应的标签：
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img) #将图像数据从 NumPy 数组格式转换为 PIL 图像对象。
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices: #检查当前索引是否在被“中毒”的样本索引列表 self.poi_indices 中。
            target = self.trigger_handler.trigger_label #修改该样本的标签为触发器的标签（self.trigger_handler.trigger_label）。
            img = self.trigger_handler.put_trigger(img) #调用 self.trigger_handler.put_trigger(img) 方法在图像上添加触发器。
 
        """
        对图像和标签应用预定义的 transform 和 target_transform，如果它们存在的话。例如，transform 可能包括将图像转换为张量和归一化操作。
        """
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTPoison(MNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

