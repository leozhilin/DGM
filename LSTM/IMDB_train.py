import random
import torch

SEED = 1234

torch.manual_seed(SEED)  # 为CPU设置随机种子
torch.cuda.manual_seed(SEED)  #为GPU设置随机种子
# 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销
torch.backends.cudnn.deterministic = True

from torchtext import data
LABEL = data.LabelField()
TEXT = data.Field(lower= True)

# 加载IMDB电影评论数据集
from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print(vars(train_data.examples[0]))
print(len(train_data))
#构建词汇表
vocab_size = 20000
TEXT.build_vocab(train_data, max_size=vocab_size) #默认会添加unk --> 未知单词, pad --> 填充
LABEL.build_vocab(train_data)

#查看词汇表中最常见的单词
print(TEXT.vocab.freqs.most_common(10))
print(TEXT.vocab.itos[:10])

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
#文本批处理
train_iter, val_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                             batch_size=32,
                                                             device=device)

import torch.nn as nn
# 定义单向LSTM模型
class Model(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(Model, self).__init__()
        # 使用预训练的词向量模型，freeze=False 表示允许参数在训练中更新
        # 在NLP任务中，当我们搭建网络时，第一层往往是嵌入层，对于嵌入层有两种方式初始化embedding向量，
        # 一种是直接随机初始化，另一种是使用预训练好的词向量初始化。
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1)
        self.predictor = nn.Linear(hidden_size, 2) #最终分两类：positive, negative

    def forward(self, x):
        out = self.embedding(x)
        # lstm 的input为[batchsize, max_length, embedding_size]，输出表示为 output,(h_n,c_n),
        # 保存了每个时间步的输出，如果想要获取最后一个时间步的输出，则可以这么获取：output_last = output[:,-1,:]
        out, (hidden, cell) = self.encoder(out)
        output = self.predictor(hidden.squeeze(0))  # 句子最后时刻的 hidden state
        return output
lstm = Model(hidden_size=100, embedding_dim=300, vocab_size=20002)
lstm.to(device)
print(lstm)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
EPOCH = 30
total_train_step = 0
total_test_step = 0
train_data_size = len(train_data)
test_data_size = len(test_data)
for epoch in range(EPOCH):
    print("-------第 {} 轮训练开始-------".format(epoch+1))
    lstm.train() #RNN的反向传播只能在train模式下进行，中途转入evaluate过程时，会自动设置为model,eval()模式
    # 训练步骤开始
    for indices, batch in enumerate(train_iter):
        inputs, targets = batch.text, batch.label
        optimizer.zero_grad()
        outputs = lstm(inputs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
    # 测试步骤开始
    lstm.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for indices, batch in enumerate(test_iter):
            inputs, targets = batch.text, batch.label
            outputs = lstm(inputs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    total_test_step = total_test_step + 1
    # torch.save(lstm, f"lstm_lr001_epoch{epoch}.pth")
