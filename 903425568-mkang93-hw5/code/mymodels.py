import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		#self.hidden1 = nn.Linear(178, 16)
		#self.out = nn.Linear(16, 5)
		self.hidden1 = nn.Linear(178,45)
		self.hidden2 = nn.Linear(45,20)
		self.out = nn.Linear(20,5)
	def forward(self, x):
		#x = nn.functional.sigmoid(self.hidden1(x))
		#x = self.out(x)
		x = torch.relu(self.hidden1(x))
		x = torch.relu(self.hidden2(x))
		x = self.out(x)
		return x

class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		#self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		#self.pool = nn.MaxPool1d(kernel_size=2)
		#self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)
		#self.pool = nn.MaxPool1d(kernel_size=2)
		#self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		#self.fc2 = nn.Linear(128, 5)
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=64)
		self.fc2 = nn.Linear(64, 5)
		self.drop = nn.Dropout(p=0.25)
	def forward(self, x):
		#x = self.pool(nn.functional.relu(self.conv1(x)))
		#x = self.pool(nn.functional.relu(self.conv2(x)))
		#x = x.view(-1, 16 * 41)
		#x = nn.functional.relu(self.fc1(x))
		#x = self.fc2(x)
		x = self.pool(nn.functional.relu(self.conv1(x)))
		x = self.pool(nn.functional.relu(self.conv2(x)))
		x = x.view(-1, 16 * 41)
		x = nn.functional.relu(self.fc1(self.drop(x)))
		x = self.fc2(x)
		return x


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=1, batch_first=True, dropout=0.2)
		self.fc = nn.Linear(in_features=32, out_features=5)

	def forward(self, x):
		x,_ = self.rnn(x)
		x= nn.functional.relu(x)
		x = self.fc(x[:, -1, :])
		
		return x
		

class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		self.input_layer1 = nn.Linear(in_features=dim_input, out_features=32) #unimproved model
		self.rnn_model = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True) #unimproved model
		self.input_layer2 = nn.Linear(in_features=16, out_features=2) #unimproved model

	def forward(self, input_tuple):
		seqs, lengths = input_tuple
		seqs = torch.tanh(self.input_layer1(seqs)) #unimproved model
		#seqs = pack_padded_sequence(seqs, lengths, batch_first=True) #unimproved model
		seqs, h = self.rnn_model(seqs) #unimproved model
		#seqs, _ = pad_packed_sequence(seqs, batch_first=True) #unimproved model
		seqs = self.input_layer2(seqs[:, -1, :]) #unimproved model
		return seqs
