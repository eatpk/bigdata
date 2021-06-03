import numpy as np
import pandas as pd
from functools import reduce
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.
	df=pd.read_csv(path)
	df['y']-=1
	data = torch.from_numpy(df.drop('y',axis=1).to_numpy().astype(np.float32))
	target = torch.from_numpy(df['y'].to_numpy().astype('long'))
	if model_type == 'MLP':
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		dataset = TensorDataset(data.unsqueeze(1), target)
	elif model_type == 'RNN':
		dataset = TensorDataset(data.unsqueeze(2), target)
		print(data.unsqueeze(2))
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	lambda_func = lambda x,y: x + y
	fee = reduce(lambda_func, seqs)
	fee = reduce(lambda_func, fee)

	num_features = int(max(fee)+1)
	return num_features


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.

		answers = []
		for x in seqs:
			mtx = np.zeros((len(x), num_features))
			each_line = 0
			for ets in x:
				for et in ets:
					mtx[each_line, et] = 1
				each_line = each_line + 1
			answers.append(mtx)
			self.seqs = answers	
	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence
	labels = []
	lengths = []
	seqs = []
	for i in batch:
		labels.append(i[1])
		lengths.append(len(i[0]))
		seqs.append(i[0])

	longest = np.max(lengths)
	orders = np.argsort(np.array(lengths)*-1)

	for i in range(len(seqs)):
		diff = longest - len(seqs[i])
		tk = np.zeros([diff,len(seqs[i][0])])
		tkm = np.copy(seqs[i])
		seqs[i] = np.concatenate([tkm,tk])
	seqs,labels,lengths= np.array(seqs)[orders],np.array(labels)[orders],np.array(lengths)[orders]

	seqs_tensor = torch.FloatTensor(seqs)
	lengths_tensor = torch.LongTensor(lengths)
	labels_tensor = torch.LongTensor(labels)

	return (seqs_tensor, lengths_tensor), labels_tensor
