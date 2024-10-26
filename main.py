import argparse  ##argparse：用于解析命令行参数
import os
import pathlib
import re
import time
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_poisoned_training_set, build_testset #自定义的代码，用于构建中毒（后门）训练集和测试集。
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch #自定义的代码，用于评估模型、选择优化器以及训练一个 epoch。
from models import BadNet #自定义代码，包含 BadNet 模型的定义。

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".') #代码使用 argparse.ArgumentParser() 定义了一组命令行参数，用户可以通过命令行传递这些参数来自定义脚本的运行行为。
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)') #--dataset：指定要使用的数据集（默认值：MNIST）。
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types') #--nb_classes：分类类型的数量（默认值：10）。
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)') #--load_local：是否加载本地的预训练模型（默认行为是训练一个新模型）
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)') #--loss：指定损失函数（均方误差或交叉熵，默认：均方误差）。
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)') #--optimizer：选择优化器（默认值：SGD，即随机梯度下降）。
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100') #--epochs：训练的 epoch 数量（默认值：100）。
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64') #--batch_size：训练和测试数据的批量大小（默认值：64）。
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')  #指定工作线程数
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001') #--lr：学习率（默认值：0.01）。
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)') #--download：是否下载数据集（如果本地没有的话）。
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)') #定使用的设备（CPU 或 CUDA，即 GPU，默认：CPU）。
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)') #用于后门攻击的训练数据中毒率，即中毒样本占比（默认值：0.1）。
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)') #触发器的标签，表示中毒样本被错误分类到的目标类别（默认：0）。
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)') #后门攻击中触发器图像的路径（默认：./triggers/trigger_white.png）。
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)') #触发器的大小（默认值：5）。

args = parser.parse_args() #这行代码会解析所有的命令行参数，并将它们存储在 args 对象中，供脚本后续使用。

def main():
    print("{}".format(args).replace(', ', ',\n')) #将命令行参数（args）格式化为字符串并打印，便于检查输入的参数设置。它将每个参数换行打印，以提高可读性。

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num  #设置环境变量 CUDA_VISIBLE_DEVICES，确保指定的 GPU 被用于训练。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps" 设备会被传递给模型和数据，以决定在哪个硬件上运行（GPU 或 CPU）

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)  #这两行代码确保 checkpoints 和 logs 目录存在，用于存储模型检查点和日志文件，mkdir(parents=True, exist_ok=True) 保证递归创建父目录，并且如果目录已经存在，不会抛出错误。

    print("\n# load dataset: %s " % args.dataset) #加载数据集
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args) #构建训练数据集
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args) #构建干净测试数据集，后门测试数据集
    
    """
    通过 DataLoader 创建训练集和验证集的加载器，指定以下几个关键参数：
    batch_size=args.batch_size：批量大小，来自命令行参数。
    shuffle=True：数据随机打乱，每个 epoch 都会以不同的顺序加载数据，有助于提高模型的泛化能力。
    num_workers=args.num_workers：并行加载数据的工作线程数，使用命令行参数 --num_workers 来决定。
    """
    data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # shuffle 随机化

    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device) #初始化 BadNet 模型，该模型根据输入通道数和分类类别数进行构建，并将模型移至指定的设备（CPU或GPU）。
    criterion = torch.nn.CrossEntropyLoss() #定义损失函数为交叉熵损失，通常用于分类任务。
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr) #根据命令行参数选择优化器，并传入模型参数和学习率。optimizer_picker 是自定义函数，用于选择不同的优化器（如 SGD 或 Adam）。

    basic_model_path = "./checkpoints/badnet-%s.pth" % args.dataset #设置模型保存路径，文件名中带有数据集名称，以便区分不同的数据集训练的模型。
    start_time = time.time()
    if args.load_local:
        print("## Load model from : %s" % basic_model_path)
        model.load_state_dict(torch.load(basic_model_path), strict=True) #是模型的存储路径。使用 torch.load 加载保存的模型状态，并通过 model.load_state_dict() 将这些权重加载到当前的模型中,strict=True 确保模型和加载的权重必须完全匹配。
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device) #加载模型后，立即在干净的验证集（data_loader_val_clean）和中毒的验证集（data_loader_val_poisoned）上评估模型。
        print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}") #evaluate_badnets() 是自定义的评估函数，返回评估指标，如清洁数据上的准确率（TCA，Test Clean Accuracy）和攻击成功率（ASR，Attack Success Rate）。
    else:
        print(f"Start training for {args.epochs} epochs") #如果用户没有选择加载本地模型，则进入模型训练流程，首先打印即将训练的 epoch 数量（由 args.epochs 指定）
        stats = []
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device) #通过 train_one_epoch 函数对训练集进行训练，并返回训练时的统计数据（如损失 loss）。
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device) #每个 epoch 结束后，使用 evaluate_badnets 函数评估模型在干净和中毒验证集上的表现，得到清洁数据上的准确率和中毒数据上的攻击成功率。
            print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n") #打印当前 epoch 的训练损失、测试集上的准确率（TCA）和攻击成功率（ASR），结果精确到小数点后四位。
            
            # save model 
            torch.save(model.state_dict(), basic_model_path) #每个 epoch 结束后，模型的参数都会保存到指定的路径 basic_model_path 中。这样即使训练中断，也可以从上次保存的模型继续。

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
            } #将每个 epoch 的训练和测试统计结果保存在一个字典中，包含训练和测试的所有指标（如 train_loss、test_clean_acc、test_asr 等）。

            # save training stats
            stats.append(log_stats) #将每一轮的统计结果添加到 stats 列表中。
            df = pd.DataFrame(stats)
            df.to_csv("./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()
