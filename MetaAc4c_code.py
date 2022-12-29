#!/usr/bin/env python3

#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import torch.nn.functional as F
import os
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch import optim
from sklearn.model_selection import train_test_split
import itertools

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc,precision_recall_curve
# from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import argparse

#----->>
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

from collections import Counter
import math
import re
# from time import time
import prettytable as pt
# from scipy import interp
import matplotlib.pyplot as plt
# import platform
# from threeD_structure import *
# from FEGS import main_plus
import time
#-------------------------------------->>>>


def load_data_bicoding(Path): #读取文件
	data = np.loadtxt(Path,dtype=list)
	data_result = []
	
	# start = (config.max_len-config.choose_len)//2
	# end = config.max_len - (config.max_len-config.choose_len)//2
	
	for seq in data:
		seq = seq.upper() #所有的字符串大写
		seq = str(seq.strip('\n'))
		data_result.append(seq)
	return data_result 

def load_test_bicoding(path_pos_data,path_neg_data):

	# sequences_pos = pd.read_csv(path_pos_data, index_col=None,header=None,sep=",")
	# sequences_neg = pd.read_csv(path_neg_data, index_col=None,header=None,sep=",")

	sequences_pos = pd.read_csv(path_pos_data, index_col=0,header=0,sep=",")
	sequences_neg = pd.read_csv(path_neg_data, index_col=0,header=0,sep=",")

	#生成的数据行列为数字，其他数据行列名是字符，所有统一重制索引
	sequences_pos.reset_index(drop=True)
	sequences_neg.reset_index(drop=True)
	sequences_pos.columns = [i for i in range(sequences_pos.shape[1])]
	sequences_neg.columns = [i for i in range(sequences_neg.shape[1])]

	training_data=pd.concat([sequences_pos,sequences_neg],axis=0).reset_index(drop=True)
	print(training_data.shape)

	training_label=sequences_pos.shape[0]*[1] + sequences_neg.shape[0]*[0]
	training_label=pd.DataFrame(training_label,columns=['-1'])

	data_test = pd.concat([training_data,training_label],axis=1)

	data_test = np.array(data_test)

	# np.random.seed(42)
	# np.random.shuffle(data_test)

	X_test = np.array([_[:-1] for _ in data_test])
	y_test = np.array([_[-1] for _ in data_test])

	return X_test, y_test


def load_in_torch_fmt(X_train, y_train):
	# X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1]/vec_len), vec_len)
	# X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1]/vec_len), vec_len)
	#print(X_train.shape)
	# X_train = torch.from_numpy(X_train).long()
	# y_train = torch.from_numpy(y_train).long()
	X_train = torch.from_numpy(X_train).float()
	y_train = torch.from_numpy(y_train).long()
	#y_train = torch.from_numpy(y_train).float()
	# X_test = torch.from_numpy(X_test).float()
	# X_test, y_test = shuffleData(X_train, y_train)
	return X_train, y_train

# Positive_X = Positive_X.reshape(Positive_X.shape[0], int(Positive_X.shape[1]/vec_len), vec_len) #折叠
# Negitive_X = Negitive_X.reshape(Negitive_X.shape[0], int(Negitive_X.shape[1]/vec_len), vec_len) #折叠


#保存模型 ----
def save_checkpoint(state,is_best,OutputDir,test_index):
	if is_best:
		print('=> Saving a new best from epoch %d"' % state['epoch'])
		torch.save(state, OutputDir + '/' + str(test_index) +'_checkpoint.pth.tar')
		
	else:
		print("=> Validation Performance did not improve")
		
def ytest_ypred_to_file(y, y_pred, out_fn):#保存标签和预测值，用来画roc曲线等等工作
	with open(out_fn,'w') as f:
		for i in range(len(y)):
			f.write(str(y[i])+'\t'+str(y_pred[i])+'\n')
			
			
def chunkIt(seq, num): #将序列平均分割成num份
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
		
	return out

def shuffleData(X, y):
	index = [i for i in range(len(X))]
	# np.random.seed(42)
	random.shuffle(index)
	new_X = X[index]
	new_y = y[index]
	return new_X, new_y

def round_pred(pred,threshold):
	# list_result = []
	# for i in pred:
	# 	if i >0.5:
	# 		list_result.append(1)
	# 	elif i <=0.5:
	# 		list_result.append(0)
	# threshold = 0.5
	list_result = [0 if instance < threshold else 1 for instance in list(pred)]
	return torch.tensor(list_result)

#获取时间
def time_log(s):
	print('%s-%s'%(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),s))
	return
#------------------>>>

def adjust_model(model):
	# Freeze some layers
	# util_freeze.freeze_by_names(model, ['embedding', 'layers'])
	# util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge', 'layers'])
	# util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge', 'soft_attention', 'layers'])
	# util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge'])
	# util_freeze.freeze_by_names(model, ['embedding_merge'])
	# util_freeze.freeze_by_names(model, ['embedding'])
	
	# unfreeze some layers
	# for name, child in model.named_children():
	#     for sub_name, sub_child in child.named_children():
	#         if name == 'layers' and (sub_name == '3'):
	#             print('Encoder Is Unfreezing')
	#             for param in sub_child.parameters():
	#                 param.requires_grad = True
	
	print('-' * 50, 'Model.named_parameters', '-' * 50)
	for name, value in model.named_parameters():
		print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))
	
	# Count the total parameters
	params = list(model.parameters())
	k = 0
	for i in params:
		l = 1
		for j in i.size():
			l *= j
		k = k + l
	print('=' * 50, "Number of total parameters:" + str(k), '=' * 50)
	pass




class iFeature_BERT_LSTM_ATT_res_pool(nn.Module):
	def __init__(self,config):
			super(iFeature_BERT_LSTM_ATT_res_pool, self).__init__()
			
			global kernel_size, max_len, num_class
			kernel_size = config.kernel_size  #10
			max_len = config.max_len		  #41
			num_class = config.num_class      #1
			
			self.lstm = torch.nn.LSTM(256, 768, 1, batch_first=True, bidirectional=True) #
			self.Attention = BahdanauAttention(in_features=768*2,hidden_units=10,num_task=1)
			
			self.pool = nn.MaxPool1d(kernel_size=256)#256 #kernel_size=max_len+1) #29+1
			
			self.fc_task = nn.Sequential(
				nn.Linear(768*2, 512),
				nn.Dropout(0.1),
			)
			
			self.classifier = nn.Linear(512, num_class)

	def forward(self, x):
			x = x.reshape(-1,config.max_len,256)
			batch_size, seq_len,features= x.size()
			out, (h_n, c_n) = self.lstm(x)  
			h_n = h_n.view(batch_size, out.size()[-1]) 
			context_vector, attention_weights = self.Attention(h_n, out)
			print('context_vector.shape',context_vector.shape)  
			print('attention_weights.shape',attention_weights.shape)
			
	
			out_fianl = torch.cat((out,context_vector),1)

			out_fianl = self.pool(out_fianl.permute(0, 2, 1))
			out_fianl = out_fianl.permute(0, 2, 1)
			out_fianl = out_fianl.squeeze(1)
			#--------->
			reduction_feature = self.fc_task(out_fianl) 

			representation = reduction_feature           
			logits_clsf = self.classifier(representation)
			logits_clsf1 = logits_clsf
			logits_clsf = torch.sigmoid(logits_clsf)       

			return logits_clsf, representation

class iFeature_BERT_LSTM_ATT_res_pool_2(nn.Module): # out和context_vector相连
	def __init__(self,config):
			super(iFeature_BERT_LSTM_ATT_res_pool_2, self).__init__()
			
			global kernel_size, max_len, num_class
			kernel_size = config.kernel_size  #10
			max_len = config.max_len		  #41
			num_class = config.num_class      #1
			
			self.lstm = torch.nn.LSTM(256, 1024, 1, batch_first=True, bidirectional=True) #
			self.Attention = BahdanauAttention(in_features=1024*2,hidden_units=10,num_task=1)
			
			self.pool = nn.MaxPool1d(kernel_size=max_len+1)
			
			self.fc_task = nn.Sequential(
				nn.Linear(1024*2, 128),
				nn.Dropout(0.1),
			)
			
			self.classifier = nn.Linear(128, num_class)
		
		
	def forward(self, x):
			x = x.reshape(-1,config.max_len,256)
			print(x.shape)
			batch_size, seq_len,features= x.size()
			out, (h_n, c_n) = self.lstm(x) #torch.Size([256, 15, 64])
	
			h_n = h_n.view(batch_size, out.size()[-1]) # pareprae input for Attention
			context_vector, attention_weights = self.Attention(h_n, out)

			out_fianl = torch.cat((out,context_vector),1)		# Res_connect
			out_fianl = self.pool(out_fianl.permute(0, 2, 1))
			out_fianl = out_fianl.permute(0, 2, 1)
			out_fianl = out_fianl.squeeze(1)

			reduction_feature = self.fc_task(out_fianl) 
			representation = reduction_feature             
			logits_clsf = self.classifier(representation)
			logits_clsf1 = logits_clsf
			logits_clsf = torch.sigmoid(logits_clsf)       

			return logits_clsf, representation
				

def get_k_fold_data(k, i, X, y):
	

	fold_size = X.shape[0] // k  
	
	val_start = i * fold_size
	if i != k - 1:
		val_end = (i + 1) * fold_size
		X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
		X_train = np.concatenate([X[0:val_start], X[val_end:]], 0)
		y_train = np.concatenate([y[0:val_start], y[val_end:]], 0)
	else:  
		X_valid, y_valid = X[val_start:], y[val_start:]     
		X_train = X[0:val_start]
		y_train = y[0:val_start]
	
	return X_train, y_train, X_valid, y_valid


#!! Config ---------->>>      +
def load_config():
	parse = argparse.ArgumentParser(description='Model set')

	parse.add_argument('-species', type=str, default='ac4c', help='ST/Kcr/nhKcr/880/15w and so on ') 
	parse.add_argument('-max-len', type=int, default=415, help='the real max length of input sequences')
	# parse.add_argument('-choose-len', type=int, default=51, help='choose length of input sequences') #截取长度
	parse.add_argument('-mode',default='test',help="Set the model to train/test/all")
	parse.add_argument('-loss-Function', type=str, default='BE', help='BE, Cr, Tim, Get_loss, DVIB_loss,FL')

	# training parameters
	parse.add_argument('-lr', type=float, default=0.001, help='learning rate')
	parse.add_argument('-batch-size', type=int, default=256, help='number of samples in a batch')
	parse.add_argument('-kernel-size', type=int, default=8, help='number of kernel size')
	parse.add_argument('-epoch', type=int, default=100, help='number of iteration')  # 30
	parse.add_argument('-k-fold', type=int, default= -1 , help='k in cross validation,-1 represents train-test approach')
	
	parse.add_argument('-num-class', type=int, default=2, help='number of classes')
	parse.add_argument('-cuda', type=bool, default=True, help='if not use cuda')
	parse.add_argument('-device', type=int, default=2, help='device id')
	parse.add_argument('-threshold', type=float, default=0.5, help='use to convert the float to int in result')
	
	parse.add_argument('-gradient-clipping', type=bool, default=True, help=' avoid exploding gradient')
	parse.add_argument('-best-threshold', type=bool, default=True, help='if rechoose a new threshold not 0.5 in test/G-mean')
	

	
	# parse.add_argument('-early-stop', type=bool, default=True, help='if run all epoch to save model or not')
	config = parse.parse_args()
	return config

#!! Main  +_+
if __name__ == '__main__':
	# Hyper Parameters------------------>>
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
	np.random.seed(42)
	random.seed(42)
	torch.backends.cudnn.deterministic = True

	# torch.cuda.manual_seed_all(42)
	# torch.backends.cudnn.benchmark = False
	
	#!! Set Hyper Parameters
	config = load_config()
	
	loss_F = config.loss_Function
	
	if loss_F == 'BE':
		config.num_class = 1
	else:
		config.num_class = 2
		
	if config.Add_CLS_SEP == True:
		config.max_len = config.max_len + 2
			
			
	# config.max_len = 41
	config.b = 0.06
	
	'''set device'''
	torch.cuda.set_device(config.device) #选择默认GPU：1
	
	#! 并行------->
	# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6,7' #6块卡
	# os.environ['CUDA_VISIBLE_DEVICES'] = '2' #块卡
	
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cuda" if config.cuda else "cpu")

	
	#!! # ========>>>   Model_choose
	# net = iFeature_BERT_LSTM_ATT_res_pool(config).to(device) 	
	net = iFeature_BERT_LSTM_ATT_res_pool_2(config).to(device) 
	#======================================================================>>>
	
	
	# net architecture
	print(net)
	adjust_model(net)

	
	net_name = str(net).split('(')[0]
	config.model_name = net_name


	loss_F = config.loss_Function

	
	# k_folds = 10
	EPOCH = config.epoch
	BATCH_SIZE = config.batch_size
	LR = config.lr
	kernel_size = config.kernel_size

	#保存结果的列表们
	trainning_result = []
	validation_result = []
	testing_result = []
	
	train_loss_sum, valid_loss_sum, test_loss_sum= 0, 0, 0
	train_acc_sum , valid_acc_sum , test_acc_sum = 0, 0, 0
	
	# train_loss_sum_best, valid_loss_sum_best = 0, 0
	# train_acc_sum_best , valid_acc_sum_best = 0, 0
	
	test_acc = []   
	test_auc = []
	test_losses = []  
	#-------------------------------------------------->>>

	
	k_folds = config.k_fold   # -1  # 5
	if k_folds == -1:       					# train_val_test
		all_index = ['-1','-2'] 

	#-------------------------------------------------->>>
	metrics_test_index = {'ACC':[],'AUC':[],'SN':[],'SP':[],'MCC':[],'F1':[],'precision':[],'recall':[]}
	
	#一次性读取数据

	species = config.species
	data_path = '/MetaAc4C/MetAc4C/BERT_embeding_data/'
	
	
	# train_pos_fa = data_path+'merge_real_{}_positive_train_sythetic.csv'.format(species)
	# train_neg_fa = data_path+'merge_{}_unbalance_negative_train.csv'.format(species) #样本用的多负样本
	# 
	# train_pos_fa = data_path+'merge_{}_positive_train.csv'.format(species)
	# train_neg_fa = data_path+'merge_{}_negative_train.csv'.format(species)
	test_pos_fa = data_path +'merge_{}_positive_test.csv'.format(species)
	test_neg_fa = data_path +'merge_{}_negative_test.csv'.format(species)


	# train_all_X, train_all_y = load_test_bicoding(train_pos_fa,train_neg_fa)
	test_all_X, test_all_y   = load_test_bicoding(test_pos_fa,test_neg_fa)
	# ind_all_X, ind_all_y   = load_test_bicoding(ind_pos_fa,ind_neg_fa)
	
	# print('train_all_X:',train_all_X)
	# print('train_all_y:',train_all_y)
	print('test_all_X:',test_all_X)
	print('test_all_y:',test_all_y)
	#------------------------------------------------------------------------------------>>
	#数据读取完毕
	
	for index_fold in all_index:  # index_fold  # test_index
		print('*'*45,'第', '{}/{}'.format(index_fold,len(all_index)) ,'折','*'*45)
		X_train, y_train, X_valid, y_valid, X_test, y_test = [],[],[],[],[],[]
		train_sigle_result,valid_sigle_result,test_sigle_result = [],[],[]
		
		OutputDir = '/model_result/{0}/{1}_{2}_{0}_{3}/fold_{4}'.format(species,net_name,config.loss_Function,k_folds, index_fold)
		config.metrics_save_dir = '/'.join(OutputDir.split('/')[:-1]) 

		OutputDir_tsne_pca = OutputDir +'/Out_tsne_pca'
		if os.path.exists(OutputDir_tsne_pca):
			print('OutputDir is exitsted')
		else:
			os.makedirs(OutputDir_tsne_pca)
			print('success create dir test')
			
		#---------------------------------->>
		if k_folds == -1: 
			# X_train, X_valid, y_train, y_valid = train_test_split(train_all_X,train_all_y, test_size=1.0/8, random_state=42)
			X_test, y_test = test_all_X, test_all_y
		
			
		# X_train, y_train = load_in_torch_fmt(X_train, y_train)
		# X_valid, y_valid = load_in_torch_fmt(X_valid, y_valid)
		X_test, y_test   = load_in_torch_fmt(X_test, y_test)
		
		
		# train_loader = Data.DataLoader(Data.TensorDataset(X_train,y_train), BATCH_SIZE, shuffle = False)
		# val_loader = Data.DataLoader(Data.TensorDataset(X_valid, y_valid), BATCH_SIZE, shuffle = False)
		test_loader = Data.DataLoader(Data.TensorDataset(X_test, y_test), BATCH_SIZE, shuffle = False)
		


		model = net 
		
		if loss_F == 'BE':
			criterion = nn.BCELoss(size_average=False)												#BE
			# criterion = nn.BCEWithLogitsLoss(size_average=False)
		elif loss_F == 'Cr':
			criterion = nn.CrossEntropyLoss()														#Cross
			
		t0 = time.time()
		
		optimizer = torch.optim.Adam(params = model.parameters(), lr = LR)
		#optimizer = torch.optim.AdamW(params = model.parameters(), lr=LR, weight_decay=0.0025)
		
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95) #动态学习率
		# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
		
		train_losses = []
		val_losses = []
		
		train_acc = [] 
		val_acc = []     

		#保存--
		best_acc = 0
		best_loss = 500
		patience = 0
		patience_limit = 10
		
		epoch_list = [] 
		torch_val_best, torch_val_y_best  = torch.tensor([]),torch.tensor([]) #用于保存最好的epoch的val结果
		

		
		#------------->>>
		#!!! Test
		if 'test' in config.mode:
			torch.cuda.empty_cache()
			#保存所有输入和预测值
			torch_test, torch_test_y   = torch.tensor([]),torch.tensor([])
			
			checkpoint = torch.load('/MetaAc4C/MetAc4C/model_result/ac4c/model_result/checkpoint.pth.tar') 
			print('-' * 10 + '<-[ StartTesting]->' + '-' * 10)
			print('<<<<---load_model:{}_checkpoint.pth.tar--->>>>'.format(index_fold))
			
			model = net  # 重建模型结构
			print('model loaded...')
			model.load_state_dict(checkpoint['state_dict']) #读取最好的模型
			# model.load_state_dict(torch.load(OutputDir + '/trained_model.pkl'))   #读取最后的模型
			
			model.to(device)
			model.eval()
				
			
			test_loss = 0
			test_correct = 0
			
			for step, batch in enumerate(test_loader):    
				
				(test_x, test_y) = batch
				#y_pred_prob_train = [] #Tensorboard
				# gives batch data, normalize x when iterate train_loader
				test_x = Variable(test_x, requires_grad=False).to(device)  
				test_y = Variable(test_y, requires_grad=False).to(device) 
				
				optimizer.zero_grad()
				
				if loss_F == 'BE':
					y_hat_test, presention_test = model(test_x)
					loss = criterion(y_hat_test.squeeze(), test_y.type(torch.FloatTensor).to(device)).item()   #BE
					pred_test = round_pred(y_hat_test.data.cpu().numpy(),threshold=config.threshold).to(device) 				#BE
					pred_prob_test = y_hat_test															#BE
	

				# loss = criterion(y_hat_test, test_y.to(device)).item()      # batch average loss 		  #Cross
		
				test_loss += loss * len(test_y)             # sum up batch loss 
				
				# #加入列表用于PCA/tsne
				# repres_list_test.extend(presention_test.cpu().detach().numpy())
				# label_list_test.extend(test_y.cpu().detach().numpy())
				
				# get the index of the max log-probability
				test_correct += pred_test.eq(test_y.view_as(pred_test)).sum().item()
				
				torch_test = torch.cat([torch_test,pred_prob_test.data.cpu()],dim=0)#收集所有的输出概率/最大值位置0-1
				torch_test_y = torch.cat([torch_test_y,test_y.data.cpu()],dim=0)	#收集所有标签
				
			if config.best_threshold ==True:
				
				fpr_test, tpr_test, thresholds_new = roc_curve(torch_test_y.data.cpu().numpy(), torch_test.reshape((-1, )))
				gmeans = np.sqrt(tpr_test * (1-fpr_test)) #
				# locate the index of the largest g-mean   #优化阈值
				
				ix = np.argmax(gmeans)
				print('Best Threshold=%f, G-Mean=%.3f' % (thresholds_new[ix], gmeans[ix]))
				best_threshold = thresholds_new[ix]	
				
				config.threshold = best_threshold
				
				pred_test = round_pred(torch_test.data.cpu().numpy(),threshold=best_threshold) #将所有的输出和新阈值进行比较得出新标签
				test_correct = pred_test.eq(y_test.view_as(pred_test)).sum().item() #y_test: 所有test的标签
				
			#----------------------------------------->>>>
			
			test_losses.append(test_loss/len(X_test)) # all loss / all sample
			accuracy_test = 100.*test_correct/len(X_test)
			print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
				test_loss/len(X_test), test_correct, len(X_test), accuracy_test))
	
			test_acc.append(accuracy_test)
			
			#统计10折-test总结果
			test_loss_sum += test_losses[-1] # test_loss of 10 fold
			test_acc_sum += test_acc[-1]     # test_loss of 10 fold 
	
			#train_loss :记录i折的所有epoch的损失（100个）
	
			#统计所有预测结果
			print('torch_test:',torch_test.shape[0])
			

			print('test:')
			metrics_test = calculateScore(torch_test_y, torch_test.numpy())
			testing_result.append(metrics_test)
			
			#update dictionary
			metrics_test_index['SN'].append(metrics_test['sn'])
			metrics_test_index['SP'].append(metrics_test['sp'])
			metrics_test_index['ACC'].append(metrics_test['acc'])
			metrics_test_index['MCC'].append(metrics_test['MCC'])
			metrics_test_index['AUC'].append(metrics_test['AUC'])
			metrics_test_index['F1'].append(metrics_test['F1'])
			metrics_test_index['precision'].append(metrics_test['precision'])
			metrics_test_index['recall'].append(metrics_test['recall'])
			
			#!!! 保存所有的标签和预测值
			out_test_file = OutputDir+'/test_result_{}.txt'.format(index_fold)
			ytest_ypred_to_file(torch_test_y.numpy(), torch_test.numpy(), out_test_file)
			
			auroc = metrics.roc_auc_score(torch_test_y.numpy(), torch_test.numpy())
			test_auc.append(auroc)
			
			#单独检查每个index_fold的结果
			temp_test_dict = ([metrics_test])
			analyze_sigle(temp_test_dict, OutputDir,species,'test')
			
	

		
	
	
	
	
	
	
	