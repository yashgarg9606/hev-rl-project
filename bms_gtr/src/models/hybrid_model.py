import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self,in_channels=3,hidden_channels=64):
        super().__init__()

        self.conv1=nn.Conv1d(
            in_channels,
            32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn1=nn.BatchNorm1d(32)

        self.conv2=nn.Conv1d(
            32,
            hidden_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn2=nn.BatchNorm1d(hidden_channels)

        self.pool=nn.AdaptiveAvgPool1d(1)

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.pool(x).squeeze(-1)

        return x


class TCNBlock(nn.Module):
    def __init__(self,channels,kernel_size=3,dilation=1,dropout=0.2):
        super().__init__()

        padding=(kernel_size-1)*dilation

        self.conv1=nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn1=nn.BatchNorm1d(channels)

        self.conv2=nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn2=nn.BatchNorm1d(channels)

        self.dropout=nn.Dropout(dropout)

    def _chomp(self,x,padding):
        if padding==0:
            return x
        return x[:,:,:-padding]

    def forward(self,x):
        residual=x

        padding1=self.conv1.padding[0]
        x=self.conv1(x)
        x=self._chomp(x,padding1)
        x=F.relu(self.bn1(x))
        x=self.dropout(x)

        padding2=self.conv2.padding[0]
        x=self.conv2(x)
        x=self._chomp(x,padding2)
        x=self.bn2(x)
        x=self.dropout(x)

        return F.relu(x+residual)


class TCN(nn.Module):
    def __init__(self,channels=64,dropout=0.2):
        super().__init__()

        self.blocks=nn.ModuleList([
            TCNBlock(
                channels,
                kernel_size=3,
                dilation=1,
                dropout=dropout
            ),
            TCNBlock(
                channels,
                kernel_size=3,
                dilation=2,
                dropout=dropout
            ),
            TCNBlock(
                channels,
                kernel_size=3,
                dilation=4,
                dropout=dropout
            ),
            TCNBlock(
                channels,
                kernel_size=3,
                dilation=8,
                dropout=dropout
            )
        ])

    def forward(self,x):
        for block in self.blocks:
            x=block(x)

        return x


class AdditiveAttention(nn.Module):
    def __init__(self,hidden_size,attention_size=64):
        super().__init__()

        self.W=nn.Linear(
            hidden_size,
            attention_size
        )

        self.v=nn.Linear(
            attention_size,
            1,
            bias=False
        )

    def forward(self,x):
        scores=self.v(
            torch.tanh(self.W(x))
        )

        weights=torch.softmax(
            scores,
            dim=1
        )

        context=torch.sum(
            weights*x,
            dim=1
        )

        return context,weights


class CNNTCNLSTMAttention(nn.Module):
    def __init__(
        self,
        cnn_channels=64,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.2
    ):
        super().__init__()

        self.cnn=CNNBlock(
            in_channels=3,
            hidden_channels=cnn_channels
        )

        self.tcn=TCN(
            channels=cnn_channels,
            dropout=dropout
        )

        self.lstm=nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers>1 else 0
        )

        self.attention=AdditiveAttention(
            hidden_size=lstm_hidden,
            attention_size=64
        )

        self.regressor=nn.Sequential(
            nn.Linear(lstm_hidden,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Conv1d)):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def forward(self,x):
        batch_size,sequence_length,channels,length=x.shape

        cnn_features=[]

        for t in range(sequence_length):
            cycle=x[:,t,:,:]

            features=self.cnn(cycle)

            cnn_features.append(features)

        x=torch.stack(
            cnn_features,
            dim=1
        )

        x=x.transpose(1,2)

        x=self.tcn(x)

        x=x.transpose(1,2)

        x,_=self.lstm(x)

        context,attention_weights=self.attention(x)

        output=self.regressor(context)

        return output.squeeze(-1),attention_weights