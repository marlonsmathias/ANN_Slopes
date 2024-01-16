import pandas as pd

import torch
import torch.nn as nn

#from sklearn.metrics import r2_score


#import seaborn as sns
import numpy as np

class Neural_network(nn.Module):
    def __init__(self, input_size1, output_size1, input_size2, output_size2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        super().__init__()
        
        self.device = device

        self.input_size1 = input_size1
        self.output_size1 = output_size1
        self.input_size2 = input_size2
        self.output_size2 = output_size2

    def init_net1(self, layers=[40, 40, 40, 40], activation_function=nn.ReLU(), dropout=0.0):

        input_size = self.input_size1
        output_size = self.output_size1

        modules = []

        modules.append(nn.Linear(input_size, layers[0]))
        modules.append(activation_function)

        for _in, _out in list(zip(layers, layers[1:])):
            modules.append(nn.Linear(_in, _out))
            modules.append(activation_function)
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(layers[-1], output_size))

        self.net1 = nn.Sequential(*modules).to(self.device)

    def init_net2(self, layers=[40, 40, 40, 40], activation_function=nn.ReLU(), dropout=0.0):

        input_size = self.input_size1 + self.output_size1 + self.input_size2
        output_size = self.output_size2

        modules = []

        modules.append(nn.Linear(input_size, layers[0]))
        modules.append(activation_function)

        for _in, _out in list(zip(layers, layers[1:])):
            modules.append(nn.Linear(_in, _out))
            modules.append(activation_function)
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(layers[-1], output_size))

        self.net2 = nn.Sequential(*modules).to(self.device)

    def forward(self,x):

        # Get the output of the first network
        y1 = self.net1(x[:,:self.input_size1])

        # Join the inputs to this output
        x2 = torch.cat((x,y1), dim=1)

        # Get the output of the second network
        y2 = self.net2(x2)

        # Return the output of both networks
        return torch.cat((y1,y2), dim=1)


    
class Data_loader:
    def __init__(self, file_path, val_frac=0.2, device=torch.device("cuda" if torch.cuda.is_available () else "cpu"), seed=None):

        if seed is not None:
            np.random.seed(10)

        self.device = device

        ## Set options for reading the Excel file
        self.header_names = ['c','c_cov','phi','phi_cov','gamma','gamma_cov','HV','H','V1','FS','V2','beta','PF','C_oper','A0','C_constr','C_init','Af','C_fail','C_total']
        
        self.input_columns_DDO = ['c','phi','gamma','HV','H']
        self.output_columns_DDO = ['V1','FS']

        self.input_columns_RBDO = ['c_cov','phi_cov','gamma_cov']
        self.output_columns_RBDO = ['V2','beta']

        ## Read the Excel file
        df = pd.read_excel(file_path, sheet_name='1_RO', header=1)
        
        # Remove unused columns
        df.drop(['N','Unnamed: 9','Unnamed: 12','Unnamed: 16'], axis=1, inplace=True)

        # Rename columns
        df.columns = self.header_names

        # Remove dashes
        df.replace('-', np.NaN, inplace=True)

        self.df = df.copy()

        # Get number of samples

        self.n_cases = len(self.df)
        self.n_train = int((1-val_frac)*self.n_cases)
        self.n_val = self.n_cases-self.n_train

        X1 = self.df[self.input_columns_DDO].to_numpy()
        Y1 = self.df[self.output_columns_DDO].to_numpy()

        X2 = self.df[self.input_columns_RBDO].to_numpy()
        Y2 = self.df[self.output_columns_RBDO].to_numpy()

        # Train/validation split

        inds = list(range(0,self.n_cases))
        np.random.shuffle(inds)
        inds_t = inds[:self.n_train]
        inds_v = inds[self.n_train:]

        X1_t = X1[inds_t,:]
        Y1_t = Y1[inds_t,:]
        X2_t = X2[inds_t,:]
        Y2_t = Y2[inds_t,:]

        X1_v = X1[inds_v,:]
        Y1_v = Y1[inds_v,:]
        X2_v = X2[inds_v,:]
        Y2_v = Y2[inds_v,:]

        # Normalize values
        self.X1_norm = [np.nanmean(X1_t, axis = 0), np.nanstd(X1_t, axis = 0)]
        self.Y1_norm = [np.nanmean(Y1_t, axis = 0), np.nanstd(Y1_t, axis = 0)]
        self.X2_norm = [np.nanmean(X2_t, axis = 0), np.nanstd(X2_t, axis = 0)]
        self.Y2_norm = [np.nanmean(Y2_t, axis = 0), np.nanstd(Y2_t, axis = 0)]

        X1_t, X2_t, Y1_t, Y2_t = self.normalize_values(X1=X1_t,X2=X2_t,Y1=Y1_t,Y2=Y2_t)
        X1_v, X2_v, Y1_v, Y2_v = self.normalize_values(X1=X1_v,X2=X2_v,Y1=Y1_v,Y2=Y2_v)

        # Create tensors
        self.X1_t = torch.tensor(X1_t,dtype=torch.float32,device=device)
        self.Y1_t = torch.tensor(Y1_t,dtype=torch.float32,device=device)

        self.X2_t = torch.tensor(X2_t,dtype=torch.float32,device=device)
        self.Y2_t = torch.tensor(Y2_t,dtype=torch.float32,device=device)

        self.X1_v = torch.tensor(X1_v,dtype=torch.float32,device=device)
        self.Y1_v = torch.tensor(Y1_v,dtype=torch.float32,device=device)

        self.X2_v = torch.tensor(X2_v,dtype=torch.float32,device=device)
        self.Y2_v = torch.tensor(Y2_v,dtype=torch.float32,device=device)


    def normalize_values(self,X1=None,X2=None,Y1=None,Y2=None):
        
        X1 = (X1-self.X1_norm[0])/self.X1_norm[1]
        X2 = (X2-self.X2_norm[0])/self.X2_norm[1]

        Y1 = (Y1-self.Y1_norm[0])/self.Y1_norm[1]
        Y2 = (Y2-self.Y2_norm[0])/self.Y2_norm[1]

        return X1, X2, Y1, Y2

    def get_batch1(self,batch_size=10,shuffle=True,validation=False):

        if not validation:
            n = self.n_train
            X1 = self.X1_t
            Y1 = self.Y1_t
        else:
            n = self.n_val
            X1 = self.X1_v
            Y1 = self.Y1_v

        inds = list(range(n))
        if shuffle:
            np.random.shuffle(inds)

        for i in range(0, n, batch_size):
            ind = inds[i:i+batch_size]
            yield X1[ind,:], Y1[ind,:]

    def get_batch2(self,batch_size=10,shuffle=True,validation=False):

        if not validation:
            X1 = self.X1_t
            X2 = self.X2_t
            Y1 = self.Y1_t
            Y2 = self.Y2_t
        else:
            X1 = self.X1_v
            X2 = self.X2_v
            Y1 = self.Y1_v
            Y2 = self.Y2_v

        X = torch.cat((X1,X2), dim=1)
        Y = torch.cat((Y1,Y2), dim=1)

        # Only consider non-nan values
        inds = torch.nonzero(torch.isnan(Y2[:,0])==False).tolist()
        inds = [i[0] for i in inds]

        if shuffle:
            np.random.shuffle(inds)

        n = len(inds)

        for i in range(0, n, batch_size):
            ind = inds[i:i+batch_size]
            yield X[ind,:], Y[ind,:]

    def get_df(self):
        return self.df