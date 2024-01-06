import torch
from torchtext import data
import torch.nn as nn
from torchtext import datasets
SEED = 1234

torch.manual_seed(SEED)  # 为CPU设置随机种子
torch.cuda.manual_seed(SEED)  #为GPU设置随机种子
# 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销
torch.backends.cudnn.deterministic = True
import pandas as pd
dataset = pd.read_csv("GLM_Change_Dataset.csv", engine= "python", header= None, encoding='utf-8')
print(dataset.info())
# dataset.describe()#数据表的描述
# print(dataset.columns())#列名
print(dataset.head())#默认显示前五行
dataset["sentiment_category"] = dataset[1]
for i in range(len(dataset[1])):
    if dataset[1][i] == "positive" or dataset[1][i] == "pos":
        dataset["sentiment_category"][i] = "pos"
    else:
        dataset["sentiment_category"][i] = "neg"

dataset.to_csv("dataset.csv", header= None, index= None, encoding='utf-8')

#导入数据集

LABEL = data.LabelField()
TEXT = data.Field(lower= True)
#设置表头
fields = [('text', TEXT), ('sentiment', None), ('label', LABEL)] #如果某列不需要就设置为None

#读取数据
ContentDataset = data.TabularDataset(
    path = "dataset.csv",
    format = 'CSV',
    fields = fields,
    skip_header = True #设置是否跳过表头,
)
#分割数据集
train, val = ContentDataset.split(split_ratio=[0.8, 0.2], strata_field="label")
train = ContentDataset
#构建词汇表
vocab_size = 20000
TEXT.build_vocab(train, max_size=vocab_size) #默认会添加unk --> 未知单词, pad --> 填充
LABEL.build_vocab(train)
device = "cuda" #if torch.cuda.is_available() else "cpu"
#文本批处理
train_iter, val_iter= data.BucketIterator.splits((train, val), batch_size=32, device=device, sort_key=lambda x: len(x.text), sort=True)
# 加载IMDB电影评论数据集
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
_, test_iter = data.BucketIterator.splits((train_data, test_data), batch_size=32, device=device)

vocab = LABEL.vocab
# 打印词表中的所有单词及其对应的编号
for word, index in vocab.stoi.items():
    print(f"Word: {word}, Index: {index}")

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


loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
EPOCH = 30
total_train_step = 0
total_test_step = 0
train_data_size = len(train)
test_data_size = len(test_data)
print(train_data_size, "  ", test_data_size)
for epoch in range(EPOCH):
    print("-------第 {} 轮训练开始-------".format(epoch+1))
    lstm.train() #RNN的反向传播只能在train模式下进行，中途转入evaluate过程时，会自动设置为model,eval()模式
    # 训练步骤开始
    total_train_loss = 0
    for indices, batch in enumerate(train_iter):
        inputs, targets = batch.text, batch.label
        optimizer.zero_grad()
        outputs = lstm(inputs)
        loss = loss_fn(outputs, targets)
        total_train_loss = total_train_loss + loss.item()
        # 优化器优化模型
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        # if total_train_step % 100 == 0:
        #     print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
    for indices, batch in enumerate(val_iter):
        inputs, targets = batch.text, batch.label
        optimizer.zero_grad()
        outputs = lstm(inputs)
        loss = loss_fn(outputs, targets)
        total_train_loss = total_train_loss + loss.item()
        # 优化器优化模型
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
    print("整体训练集上的Loss: {}".format(total_train_loss))
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
    # torch.save(lstm, f"GLM_lstm_lr001_epoch{epoch}.pth")
