import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def optimizer_picker(optimization, param, lr):
    """
    optimization：表示要选择的优化器的名称。它是一个字符串参数，可能的值有 adam 或 sgd。
    param：模型的参数，用于告诉优化器应该优化哪些参数。通常，这个参数是通过 model.parameters() 来获取。
    lr：学习率（learning rate），控制优化器每次更新模型参数时的步长大小。
    """
    if optimization == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = torch.optim.Adam(param, lr=lr)
    """
    如果传入的 optimization 参数既不是 adam 也不是 sgd，则输出提示信息，并默认选择 Adam 优化器。
    这是一种防御性编程方式，以防输入了未知的优化器类型。
    """
    return optimizer


def train_one_epoch(data_loader, model, criterion, optimizer, loss_mode, device):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        
        loss = criterion(output, batch_y)

        loss.backward() #计算损失函数相对于模型参数的梯度，即执行反向传播过程。
        optimizer.step() #使用优化器更新模型参数，基于前一步计算得到的梯度进行参数调整。
        running_loss += loss
    return {
            "loss": running_loss.item() / len(data_loader), #running_loss.item()：将 running_loss 张量的值转换为 Python 的浮点数，以便返回。len(data_loader) 表示数据加载器中的批次数量。
            }

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device):
    """
    分别在干净的验证集（data_loader_val_clean）和中毒的验证集（data_loader_val_poisoned）上进行评估，并返回模型的以下性能指标：

    clean_acc：模型在干净验证集上的准确率（Test Accuracy，TCA）。
    clean_loss：模型在干净验证集上的损失值。
    asr：模型在中毒验证集上的准确率（Attack Success Rate，ASR）。
    asr_loss：模型在中毒验证集上的损失值。
    eval 函数用于执行具体的评估逻辑。
    """
    ta = eval(data_loader_val_clean, model, device, print_perform=True)
    asr = eval(data_loader_val_poisoned, model, device, print_perform=False)
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, batch_size=64, print_perform=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() #将模型切换为评估模式。在评估模式下，模型的某些层（如 dropout 和 batch normalization）会表现出不同的行为，以确保评估时不会发生随机性。
    y_true = []
    y_predict = []
    loss_sum = []
    for (batch_x, batch_y) in tqdm(data_loader):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        batch_y_predict = model(batch_x) #将输入 batch_x 传递给模型，得到模型的预测结果 batch_y_predict。
        loss = criterion(batch_y_predict, batch_y) #计算模型预测结果与真实标签之间的损失。
        batch_y_predict = torch.argmax(batch_y_predict, dim=1) #通过 torch.argmax 获取模型预测的类别标签。
        y_true.append(batch_y)
        y_predict.append(batch_y_predict) #分别将真实标签和预测标签保存下来，用于后续计算准确率和生成报告。
        loss_sum.append(loss.item()) #将当前批次的损失添加到 loss_sum 列表中，以便之后计算平均损失。
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0) #将之前存储在 y_true 列表中的多个张量（tensor）按行（维度 0）拼接起来，生成一个完整的张量。尽管每个 batch_y 是一维的，但它们是分批处理的，不代表完整的数据集。因此，我们需要使用 torch.cat(y_true, 0) 将这些批次组合起来，得到一个表示完整数据集标签的张量
    loss = sum(loss_sum) / len(loss_sum) #计算所有批次的平均损失。

    """
    #如果 print_perform=True，使用 classification_report 打印模型的性能报告。报告包括精度、召回率、F1分数等指标，并针对每个类别进行详细的评估。target_names=data_loader.dataset.classes 用于指定类别的名称。
    classification_report 生成一个详细的分类模型性能报告，包含以下几个重要指标：
    精确率（Precision）：模型预测为正样本中真正为正样本的比例。
    召回率（Recall）：所有真实正样本中被模型正确预测为正样本的比例。
    F1 分数：精确率和召回率的调和平均数，平衡了这两个指标。
    支持数（Support）：每个类别中的实际样本数量。
    classification_report 返回一个字符串或字典，列出了模型在每个类别上的精确率、召回率和 F1 分数，还包括整个模型的加权平均指标（宏平均和微平均）。

    """
    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes)) 

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

