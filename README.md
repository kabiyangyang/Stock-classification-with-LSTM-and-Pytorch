# Stock-classification-with-LSTM-and-Pytorch

基于pytorch框架，通过过往价量行情数据训练（Bi)-LSTM模型，行情数据来源于uqer pro版，基于中证800。
每只股票的特征包含 ['tradeDate', 'openPrice', 'closePrice' ,'highestPrice', 'lowestPrice', 'turnoverRate', 'returns', 'returns_ind', 'stock', '1d_alpha_rank']
标签为1d alpha rank,为去除市值影响的收益率排名，模型为分类，主要是为了构建多空组合，即排名小于100为1类 100-700为0类 700-800为-1类。特征回看周期为60天，即每一日的特征矩阵为60*7*股票数

## date_list_generator
将csv文件保存为npy dataset易于读取，同时导出一个txt文件，记录的是字典格式的npy切片，即将npy切片目录与股票行情数据的日期进行对应。

## Parameter_generator
根据范围和step生成训练所需的模型参数和超参数，并保存至Json 文件

## train

### PrepareData
读取文件 返回data数据，contents是data的索引，dict格式，key是日期，value是idx

### LoadDataset
根据训练需要（通常为滚动训练，即训练A年，验证1年,测试1年）划分训练集验证集，输入为日期和data即可，通过内置函数进行预处理，预处理主要涉及将标签变为类别（其他如标准化等数据预处理方法已经用到了原始数据上）


通过LSTM类搭建模型并使用预定义的模型参数和超参数进行训练，模型结果，回测曲线等会保存到checkpoint文件夹里。



