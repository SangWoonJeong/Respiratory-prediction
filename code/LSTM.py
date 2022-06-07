import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/woon/06_pytorch'])

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy as sp
from scipy import signal

import glob

from scipy.ndimage import median_filter
from scipy.stats.stats import pearsonr



device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.set_device(1)
torch.cuda.current_device()


## data scale function

def MinMaxScaler(data):

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


### train length, prediction point parameter setting

# 20 / 40 / 60 / 80 / 100
# 20 : 1,000 ms
# 24 : 1,200 ms
# 28 : 1.400 ms
# 48 : 2.400 ms
seq_length = 20



prediction_length = 6  # 10 index is 500 ms  :  prediction point


TrainingDB_Ratio = 0.7
TestDB_Ratio = 0.3


## data file load

import tkinter.filedialog

root = tkinter.Tk()
root.withdraw()

file_path = tkinter.filedialog.askopenfilename()
pathname, filename = os.path.split(file_path)


dataX = []
dataY = []
flag_totalData = True
flag_totalData_training = True

PatientRespiratoryList = sorted(glob.glob(os.path.join(pathname, '*.csv')))

total_counter = 0

patient_index_training = [0]
patient_index_test = [0]

def sliding_window2(data, seq_length, prediction_length):
    x= []
    y= []
    for i in range(len(data)-seq_length-prediction_length+1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length+prediction_length-1]
        x.append(_x)
        y.append(_y)
    return  np.expand_dims(np.array(x),-1),  np.expand_dims(np.array(y),-1)

first_data =True
for filename_single_full in PatientRespiratoryList:
    #a = np.append(c)
    total_counter = total_counter + 1
    filename_single = os.path.split(filename_single_full)[1] # just name
    xy = np.loadtxt(os.path.join(pathname, filename_single), delimiter=',')
    print('Filname is {}'.format(filename_single))

    Amplitude_repiratory_signal_temp = xy[:, 0] # load original signal

    len_Amplitude_repiratory_signal_temp = len(Amplitude_repiratory_signal_temp)
    len_Amplitude_repiratory_signal_temp = np.round(len_Amplitude_repiratory_signal_temp * 0.8)  # get rid of tail information
    Amplitude_repiratory_signal_temp = Amplitude_repiratory_signal_temp[:int(len_Amplitude_repiratory_signal_temp)] # erase tail information

    # Amplitude_repiratory_signal_temp = median_filter(MinMaxScaler(Amplitude_repiratory_signal_temp),9)

    train_lenth = len(Amplitude_repiratory_signal_temp) * TrainingDB_Ratio

    train_Amplitude_repiratory_signal_temp =Amplitude_repiratory_signal_temp[:int(train_lenth)]
    train_Amplitude_repiratory_signal_temp = sp.signal.savgol_filter(train_Amplitude_repiratory_signal_temp, 25, 2)

    test_Amplitude_repiratory_signal_temp = Amplitude_repiratory_signal_temp[int(train_lenth):]

    train_temp_data, train_temp_label = sliding_window2(train_Amplitude_repiratory_signal_temp ,seq_length, prediction_length)
    test_temp_data, test_temp_label = sliding_window2(test_Amplitude_repiratory_signal_temp ,seq_length, prediction_length)


    if (first_data):
        train_data = train_temp_data
        train_label = train_temp_label

        test_data = test_temp_data
        test_label = test_temp_label

        all_Amplitude_repiratory_signal_temp = Amplitude_repiratory_signal_temp
        first_data = False
    else:
        train_data = np.vstack((train_data, train_temp_data))
        train_label = np.vstack((train_label, train_temp_label))

        test_data = np.vstack((test_data, test_temp_data))
        test_label = np.vstack((test_label , test_temp_label))
        all_Amplitude_repiratory_signal_temp = np.concatenate((all_Amplitude_repiratory_signal_temp, Amplitude_repiratory_signal_temp))

    patient_index_training.append(len(train_label))
    patient_index_test.append(len(test_label))

train_mean = np.mean(all_Amplitude_repiratory_signal_temp)
train_std = np.std(all_Amplitude_repiratory_signal_temp)
#
normal_train_data =(train_data - train_mean) / train_std
normal_train_label= (train_label - train_mean) / train_std
normal_test_data =(test_data - train_mean) / train_std
normal_test_label= (test_label - train_mean) / train_std
#
#4
train_input_data = normal_train_data
train_output_data = normal_train_label
test_input_data = normal_test_data
test_output_data = normal_test_label



# fix parameters

input_size = 1
num_classes = 1

# Hyper-parameters

sequence_length = seq_length
hidden_size = 64 # 30
num_layers = 3

batch_size = 300 #
#batch_size = 30

learning_rate = 0.00001

##

##


class ION_Dataset_Sequential(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.output[idx]
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return x, y


train_load = ION_Dataset_Sequential(train_input_data, train_output_data)

train_loader = torch.utils.data.DataLoader(dataset=train_load,
                                           batch_size=batch_size,
                                           shuffle=False)


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes): #dropout
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # , dropout=1
        self.fc = nn.Linear(hidden_size, num_classes)  # 2 for bidirection
       # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        #out = self.dropout(out) ####
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = LSTM_model(input_size, hidden_size, num_layers, num_classes).to(device)

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

# Loss and optimizer
MAE_loss = nn.L1Loss()
criterion = RMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# Train the model
total_step = len(train_loader)

import matplotlib.pyplot as plt

test_rmse_result_mean_result = []


num_epochs =200

num_epochs =20


for epoch in range(num_epochs):
    train_labels_result = 0
    train_outputs_result = 0
    train_loss_result = 0
    test_rmse_result = []
    test_corr_result = []
    test_result_data = 0

    for i, (train_data, train_labels) in enumerate(train_loader):
        model.train()
        #a = zeros(5)
        # train_data = train_data.reshape(-1, sequence_length, input_size).to(device)
        # train_labels = train_labels.reshape(-1,input_size).to(device)
        train_data = train_data.to(device)
        train_labels = train_labels.to(device)

        # Forward pass
        train_predict_outputs = model(train_data)

        loss = criterion(train_predict_outputs, train_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss
        train_loss = loss.cpu().detach().numpy()
        train_loss_result = np.append(train_loss_result, train_loss)

        train_predict_outputs = train_predict_outputs.cpu().detach().numpy()
        train_labels = train_labels.cpu().detach().numpy()

        train_labels_result = np.append(train_labels_result, train_labels)
        train_outputs_result = np.append(train_outputs_result, train_predict_outputs)

    train_corr = pearsonr(train_labels_result[1:], train_outputs_result[1:])
    train_loss_mean = np.sum(train_loss_result[1:]) / len(train_loss_result[1:])

    for iter11 in range(len(patient_index_test) - 1):
        model.eval()
        torch.no_grad()
        test_data = test_input_data[patient_index_test[iter11]:patient_index_test[iter11 + 1]]
        test_data = torch.from_numpy(test_data).float().cuda()
        test_predict_outputs = model(test_data)
        #
        test_labels = test_output_data[patient_index_test[iter11]:patient_index_test[iter11 + 1]]
        #
        test_predict_outputs = test_predict_outputs.cpu().detach().numpy()

        #test_predict_outputs = median_filter(test_predict_outputs, 9)
        test_predict_outputs = sp.signal.savgol_filter(np.squeeze(test_predict_outputs), 25, 2)
        test_predict_outputs = np.expand_dims(test_predict_outputs,axis=1)
        test_result_data = np.append(test_result_data, test_predict_outputs)

        test_corr = pearsonr(test_labels[:, 0], test_predict_outputs[:, 0])

        test_predict_outputs = torch.from_numpy(test_predict_outputs).float().cuda()

        test_labels = torch.from_numpy(test_labels).float().cuda()

        test_rmse = criterion(test_predict_outputs, test_labels)

        test_rmse_result.append(test_rmse.cpu().detach().numpy())
        test_corr_result.append(test_corr[0])


    test_rmse_result_mean = np.mean(test_rmse_result)
    test_corr_result_mean = np.mean(test_corr_result)

    if epoch >= 2:
        if min(test_rmse_result_mean_result) > test_rmse_result_mean:
            #torch.save(model, save_path)
            #print('model save')
            fig = plt.figure(figsize=(10, 5))
            line1, = plt.plot(test_labels.cpu().detach().numpy())
            line2, = plt.plot(test_predict_outputs.cpu().detach().numpy(), 'r--')
            plt.legend([line1, line2], ['True', 'Prediction'])
            plt.xlabel("Time Period")
            plt.ylabel("Relative patient respiratory signal")
            plt.title("TEST RMSE: {}, CORR:{}".format(test_rmse_result_mean,
                                                      test_corr_result_mean))
            plt.show()

    test_rmse_result_mean_result.append(test_rmse_result_mean)



    print('Epoch [{}/{}], train Loss: {:.6f}, train corr: {:.6f}'.format(epoch+1, num_epochs, train_loss_mean, train_corr[0]))
    print('test Loss: {:.6f}, test corr: {:.6f} '.format(test_rmse_result_mean, test_corr_result_mean))

# 70





#test Loss: 0.240801, test corr: 0.970960


save_path = "../06_pytorch/model_save/LSTM_300msec_.pth"
torch.save(model, save_path)
print('model save')




from scipy.io import savemat

test_result_data_save = {"LSTM_300_pred":test_predict_outputs}

savemat("LSTM_300_pred.mat", test_result_data_save)


iter5 =177

fig = plt.figure(figsize=(10, 5))
line1, = plt.plot(test_output_data[patient_index_test[iter5]:patient_index_test[iter5 + 1]])
line2, = plt.plot(test_result_data[patient_index_test[iter5]:patient_index_test[iter5 + 1]], 'r--')
plt.legend([line1, line2], ['True', 'Prediction'])
plt.xlabel("Time range [ms]")
plt.ylabel("Respiratory signal")
plt.title(" LSTM ")
plt.show()



