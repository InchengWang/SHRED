import torch
from torch.utils.data import DataLoader
import numpy as np

# =============================================================================
#  TCN 实现代码 (添加到 models.py 顶部)
# =============================================================================
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    """
    一个简单的模块，用于移除序列末尾的多余填充，以实现因果卷积。
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN 的残差块。
    包含两个带 'causal' 填充的卷积层、激活函数、dropout 和一个残差连接。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 如果输入输出维度不同，则需要一个 1x1 卷积来进行残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    时间卷积网络，由多个 TemporalBlock 堆叠而成。
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# =============================================================================
#  结束 TCN 实现代码
# =============================================================================


class SHRED(torch.nn.Module):
    '''SHRED model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state, hidden_size, number of LSTM layers,
    size of fully-connected layers, and dropout parameter'''
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.0):
        super(SHRED,self).__init__()

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=hidden_layers, batch_first=True)
        

        #全连接的decoder
        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        

    def forward(self, x):
        
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
    
        output = self.linear3(output)

        return output


class SDN(torch.nn.Module):
    '''SDN model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state,
    size of fully-connected layers, and dropout parameter'''
    def __init__(self, input_size, output_size, l1=350, l2=400, dropout=0.0):
        super(SDN,self).__init__()
        
        self.linear1 = torch.nn.Linear(input_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):

        output = self.linear1(x)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
        
        output = self.linear3(output)

        return output


# =============================================================================
#  使用 TCN 替换 LSTM 的新 SHRED 模型
# =============================================================================

class SHRED_TCN(torch.nn.Module):
    """
    SHRED 模型，但使用 TCN 替换了 LSTM。
    - input_size: 输入特征的数量 (传感器数量)。
    - output_size: 最终输出的维度。
    - num_channels: 一个列表，定义 TCN 每个隐藏层的通道数。
    - kernel_size: TCN 的卷积核大小。
    - dropout: dropout 比率。
    - fc_layers: 全连接层的维度列表。
    """
    def __init__(self, input_size, output_size, num_channels,fc_layers, kernel_size, dropout):
        super(SHRED_TCN, self).__init__()

        # TCN 层
        # num_channels 控制了 TCN 的深度和宽度
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)

        # 全连接层
        # 注意：第一个全连接层的输入维度是 TCN 最后一个隐藏层的通道数
        l1_in_features = num_channels[-1]
        l1_out_features = fc_layers[0]
        l2_out_features = fc_layers[1]
        
        self.linear1 = torch.nn.Linear(l1_in_features, l1_out_features)
        self.linear2 = torch.nn.Linear(l1_out_features, l2_out_features)
        self.linear3 = torch.nn.Linear(l2_out_features, output_size)

        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        # x 的输入维度是 (N, L, C_in)
        # TCN 需要的维度是 (N, C_in, L)，所以需要进行维度转换
        x_tcn = x.permute(0, 2, 1)
        
        # 通过 TCN
        y_tcn = self.tcn(x_tcn)
        
        # TCN 的输出维度是 (N, C_out, L)。我们需要取序列的最后一个时间点的输出来进行预测。
        # y_tcn[:, :, -1] 提取了所有批次、所有输出通道的最后一个时间步的值
        y = y_tcn[:, :, -1]
        
        # 通过全连接层
        y = self.dropout(torch.relu(self.linear1(y)))
        y = self.dropout(torch.relu(self.linear2(y)))
        y = self.linear3(y)
        
        return y

def fit(model, train_dataset, valid_dataset, batch_size=64, num_epochs=4000, lr=1e-3, verbose=False, patience=3):
    '''Function for training SHRED and SDN models'''
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    for epoch in range(1, num_epochs + 1):
        
        for k, data in enumerate(train_loader):
            model.train()
            outputs = model(data[0])
            optimizer.zero_grad()
            loss = criterion(outputs, data[1])
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)

            if verbose == True:
                print('Training epoch ' + str(epoch))
                print('Error ' + str(val_error_list[-1]))

            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = model.state_dict()
            else:
                patience_counter += 1


            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()


def forecast(forecaster, reconstructor, test_dataset):
    '''Takes model and corresponding test dataset, returns tensor containing the
    inputs to generate the first forecast and then all subsequent forecasts 
    throughout the test dataset.'''
    initial_in = test_dataset.X[0:1].clone()
    vals = []
    for i in range(0, test_dataset.X.shape[1]):
        vals.append(initial_in[0, i, :].detach().cpu().clone().numpy())

    for i in range(len(test_dataset.X)):
        scaled_output = forecaster(initial_in).detach().cpu().numpy()

        vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
        temp = initial_in.clone()
        initial_in[0,:-1] = temp[0,1:]
        initial_in[0,-1] = torch.tensor(scaled_output)

    device = 'cuda' if next(reconstructor.parameters()).is_cuda else 'cpu'
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
        recon = reconstructor(forecasted_vals[i:i+test_dataset.X.shape[1]].reshape(1, test_dataset.X.shape[1], 
                                    test_dataset.X.shape[2])).detach().cpu().numpy()
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)
    return forecasted_vals, reconstructions
