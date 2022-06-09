import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

import numpy as np
import math
import pandas as pd

import time

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("num_gpus = " + str(num_gpus))
    #if num_gpus >= 4:
    #    device = torch.device('cuda:3')
    #else:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')

print("device = " + str(device))

epsi=1e-10
nu=0.25
torch_PI=torch.tensor([3.141592653589793],device=device)

class build_stiffness_matrix(nn.Module):
    def __init__(self, matrix_dict, X):
        super(build_stiffness_matrix, self).__init__()
        #self.E=nn.Parameter(Variable(torch.tensor(E,device=device).float(), requires_grad=True))
        self.fc1=nn.Linear(2,50,bias=True)
        self.fc2=nn.Linear(50,100,bias=True)
        self.fc3=nn.Linear(100,1,bias=True)
        self.matrix_dict=matrix_dict
        self.X=X

    @staticmethod
    def tanh_general(x, alpha):
        return (torch.exp(alpha * x) - torch.exp(-alpha * x)) / (torch.exp(alpha * x) + torch.exp(-alpha * x))

    def build_matrix(self):
        M_nei=self.matrix_dict["M_nei"]
        M_ID=self.matrix_dict["M_ID"]
        K11=self.matrix_dict["K11"]
        K12=self.matrix_dict["K12"]
        K21=self.matrix_dict["K21"]
        K22=self.matrix_dict["K22"]

        Eh = self.tanh_general(self.fc3(F.relu(self.fc2(F.relu(self.fc1(self.X))))), 1.0) * 0.53 + 2.13
        #Eh = E_gd.to(device)
        M_nei=M_nei.to(device)
        Eh = Eh.squeeze(1)
        Ehx,Ehy=torch.meshgrid(1.0/Eh,1.0/Eh,indexing='ij')
        Ehh = 2.0/(Ehx+Ehy)
        MU = torch.mul(Ehh/(2.0*(1+nu)),M_nei)
        MU=MU.to(device)

        del Ehx, Ehy, Ehh, M_nei
        torch.cuda.empty_cache()

        stiff11 = torch.mul(K11, MU)
        stiff12 = torch.mul(K12, MU)
        stiff21 = torch.mul(K21, MU)
        stiff22 = torch.mul(K22, MU)

        del MU, K11, K12, K21, K22
        torch.cuda.empty_cache()

        c11 = torch.diag(torch.sum(stiff11, 1))
        stiffness11 = -stiff11 + c11 + M_ID
        del stiff11, c11
        torch.cuda.empty_cache()

        c12 = torch.diag(torch.sum(stiff12, 1))
        stiffness12 = -stiff12 + c12
        del stiff12, c12
        torch.cuda.empty_cache()

        c21 = torch.diag(torch.sum(stiff21, 1))
        stiffness21 = -stiff21 + c21
        del stiff21, c21
        torch.cuda.empty_cache()

        c22 = torch.diag(torch.sum(stiff22, 1))
        stiffness22 = -stiff22 + c22 + M_ID
        del stiff22, c22
        torch.cuda.empty_cache()

        stiffness_matrix = torch.cat((torch.cat((stiffness11,stiffness12),1),torch.cat((stiffness21,stiffness22),1)),0)

        return stiffness_matrix, Eh


class preprocess(nn.Module):
    def __init__(self, E, meshsize, lenx, leny, X, delta_ratio):
        super(preprocess, self).__init__()
        self.meshsize=meshsize
        self.delta_ratio=delta_ratio
        delta=delta_ratio*meshsize
        self.delta=delta
        self.delfloor=math.floor(2.0*delta/meshsize)
        self.lenx=lenx
        self.leny=leny
        self.Mx=math.floor(lenx/meshsize)+1
        self.My=math.floor(leny/meshsize)+1
        self.MxC=self.Mx+self.delfloor*2-9
        self.MyC=self.My+self.delfloor*2-9
        self.M=self.MxC * self.MyC
        self.X=X

    def get_bondlist_weights(self, index):
        MxC=float(self.MxC)
        delta_ratio=self.delta_ratio
        # hard coded here, needs modification with delta_ratio
        bond_weights_list=np.row_stack((np.array([-2*MxC-2, -2*MxC-1, -2*MxC, -2*MxC+1, -2*MxC+2,
                           -1*MxC-2, -1*MxC-1, -1*MxC, -1*MxC+1, -1*MxC+2,
                           -2, -1, 0, 1, 2,
                           1*MxC-2, 1*MxC-1, 1*MxC, 1*MxC+1, 1*MxC+2,
                           2*MxC-2, 2*MxC-1, 2*MxC, 2*MxC+1, 2*MxC+2])+index,
                           np.array([9.164977923e-05,0.0001124898084,0.000166910592,0.0001124898084,9.164977923e-05,
                            0.0001124898084,8.886140696e-05,0.0001919320651,8.886140696e-05,0.0001124898084,
                            0.000166910592,0.0001919320651,-0.0002299004518,0.0001919320651,0.000166910592,
                            0.0001124898084,8.886140696e-05,0.0001919320651,8.886140696e-05,0.0001124898084,
                            9.164977923e-05,0.0001124898084,0.000166910592,0.0001124898084,9.164977923e-05]))) # be careful with the precision of the weights
        #bond_weights_list = torch.tensor(bond_weights_list, device=device)

        return bond_weights_list

    def assign_ID(self):
        M=self.M
        X=self.X
        lenx=self.lenx
        leny=self.leny
        ID=torch.zeros(M)
        for i in range(M):
            if X[i,0] > lenx+epsi:
                ID[i] = 1.0
            elif X[i,0] < -epsi:
                ID[i] = 1.0
            elif X[i,1] < -leny*0.5-epsi:
                ID[i] = 1.0
            elif X[i,1] > leny*0.5+epsi:
                ID[i] = 1.0
        self.ID=ID
        return self.ID

    def ingredients(self):
        M=self.M
        ID=self.assign_ID()
        M_ID=torch.diag(ID)
        M_r3=torch.zeros([M,M])
        M_nei=torch.zeros([M,M])
        M_wei=torch.zeros([M,M])
        M_xx=torch.zeros([M,M])
        M_xy = torch.zeros([M, M])
        M_yx = torch.zeros([M, M])
        M_yy = torch.zeros([M, M])
        for i in range(M):
            if self.ID[i] < epsi:
                bond_weights_list = self.get_bondlist_weights(i)
                bondlist = bond_weights_list[0, :]
                quadWeights = bond_weights_list[1, :]
                l_bondlist = bondlist.size
                bondlist = torch.tensor(bondlist, device=device)
                quadWeights = torch.tensor(quadWeights, device=device)
                for z in range(l_bondlist):
                    j = int(bondlist[z])
                    #Xj = X[j]
                    Xji = X[j] - X[i]
                    r = torch.norm(Xji, p=2) + epsi
                    Wij = 1.0 / r

                    M_r3[i,j]=torch.pow(Wij,3)
                    M_nei[i,j]=1.0
                    M_wei[i,j]=quadWeights[z]

                    Kijxx = Xji[0] * Xji[0]
                    Kijxy = Xji[0] * Xji[1]
                    Kijyx = Xji[1] * Xji[0]
                    Kijyy = Xji[1] * Xji[1]

                    M_xx[i,j]=Kijxx
                    M_xy[i,j]=Kijxy
                    M_yx[i,j]=Kijyx
                    M_yy[i,j]=Kijyy

        LPSm = 2.0 / 3.0 * torch_PI * self.delta ** 3
        LPSm = LPSm.squeeze(0)

        M_r3 = M_r3.to(device)
        M_xx = M_xx.to(device)
        M_xy = M_xy.to(device)
        M_yx = M_yx.to(device)
        M_yy = M_yy.to(device)
        M_wei = M_wei.to(device)
        M_ID = M_ID.to(device)

        K11 = 16.0 / LPSm * torch.mul(torch.mul(M_r3, M_xx), M_wei)
        K12 = 16.0 / LPSm * torch.mul(torch.mul(M_r3, M_xy), M_wei)
        K21 = 16.0 / LPSm * torch.mul(torch.mul(M_r3, M_yx), M_wei)
        K22 = 16.0 / LPSm * torch.mul(torch.mul(M_r3, M_yy), M_wei)

        matrix_dict={"M_nei":M_nei,"M_ID":M_ID,"K11":K11,"K12":K12,"K21":K21,"K22":K22}

        return matrix_dict


def rhs_error(stiffness_matrix, input_u, input_f):
    # M=113*113
    rhs_nl = torch.matmul(stiffness_matrix,input_u.transpose(1,0))

    rhs_gd = input_f.transpose(1,0)
    # for i in range(M):
    #     if ID[i] > epsi:
    #         rhs_gd[uoffset + i] = input_u[0:M]
    #         rhs_gd[voffset + i] = input_u[M:2*M]
    #     else:
    #         rhs_gd[uoffset + i] = input_f[0:M]
    #         rhs_gd[voffset + i] = input_f[M:2 * M]

    error = torch.mean(torch.norm(rhs_nl-rhs_gd,dim=0, p=2))

    return error

def read_data_fromfile(N, M):
    inputu = torch.zeros([N,2*(M-9)*(M-9)])
    #inputt = torch.zeros([N,(M-9)*(M-9)])
    inputf = torch.zeros([N,2*(M-9)*(M-9)])
    for i in range(N):
        #print(i)
        #pdu = pd.read_csv("/home/yiming/yiming_research/statebasedAC-master/statebasedAC-master/"+base_dir+"/hom_u_test"+str(i+1)+".csv", header=0, sep=' ')
        #pdv = pd.read_csv("/home/yiming/yiming_research/statebasedAC-master/statebasedAC-master/"+base_dir+"/hom_v_test"+str(i+1)+".csv", header=0, sep=' ')
        pdu = pd.read_csv("./"+base_dir+"/hom_u_test"+str(i+1)+".csv", header=0, sep=' ')
        pdv = pd.read_csv("./"+base_dir+"/hom_v_test"+str(i+1)+".csv", header=0, sep=' ')
        pdu = torch.tensor(pdu["content"])
        pdv = torch.tensor(pdv["content"])
        pdu = torch.reshape(pdu, (M, M))
        pdv = torch.reshape(pdv, (M, M))
        pdu = pdu[4:M-5, 4:M-5]
        pdv = pdv[4:M-5, 4:M-5]
        pdu = pdu.reshape(-1,1)
        pdv = pdv.reshape(-1,1)
        pduv= torch.row_stack((pdu, pdv))
        inputu[i,:] = pduv.transpose(1,0)

        #pdf1 = pd.read_csv("/home/yiming/yiming_research/statebasedAC-master/statebasedAC-master/"+base_dir+"/hom_f1_test"+str(i+1)+".csv", header=0, sep=' ')
        #pdf2 = pd.read_csv("/home/yiming/yiming_research/statebasedAC-master/statebasedAC-master/"+base_dir+"/hom_f2_test"+str(i+1)+".csv", header=0, sep=' ')
        pdf1 = pd.read_csv("./"+base_dir+"/hom_f1_test"+str(i+1)+".csv", header=0, sep=' ')
        pdf2 = pd.read_csv("./"+base_dir+"/hom_f2_test"+str(i+1)+".csv", header=0, sep=' ')
        pdf1 = torch.tensor(pdf1["content"])
        pdf2 = torch.tensor(pdf2["content"])
        pdf1 = torch.reshape(pdf1, (M, M))
        pdf2 = torch.reshape(pdf2, (M, M))
        pdf1 = pdf1[4:M - 5, 4:M - 5]
        pdf2 = pdf2[4:M - 5, 4:M - 5]
        pdf1 = pdf1.reshape(-1,1)
        pdf2 = pdf2.reshape(-1,1)
        pdff = torch.row_stack((pdf1, pdf2))
        inputf[i,:] = pdff.squeeze(1)

        if i%(N/10)==0:
            print('%f /100 is done.' %(i/N*100))

    return inputu, inputf

def scheduler(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
def LR_schedule(learning_rate,steps,scheduler_step,scheduler_gamma):
    #print(steps//scheduler_step)
    return learning_rate*np.power(scheduler_gamma,(steps//scheduler_step))

print('Start initializing.')
time_starti=time.time()
base_dir="smallsamples20_uvonly_baby30_0"
save_base_dir="test/"
model_filename=save_base_dir+"nnParameters.ckpt"
n_samples=900
batch_size=90
learning_rate=1e-3 #usually 1e-3, 2e-3, 5e-3
epochs=10000
step_size=100
gamma=0.75
restart_ep=10000

meshsize=0.01
lenx=0.3
leny=0.3
delta_ratio=3.0
#df=pd.read_csv("/home/yiming/yiming_research/statebasedAC-master/statebasedAC-master/"+base_dir+"/hom_u_test1.csv", header=0, sep=' ')
df=pd.read_csv("./"+base_dir+"/hom_u_test1.csv", header=0, sep=' ')
X0=torch.tensor(df["x"], device=device)
X1=torch.tensor(df["y"], device=device)
ll=int(math.sqrt(X0.shape[0]))
X0 = torch.reshape(X0, (ll, ll))
X1 = torch.reshape(X1, (ll, ll))
X0 = X0[4:ll-5, 4:ll-5].float()
X1 = X1[4:ll-5, 4:ll-5].float()
X0 = X0.reshape(-1,1)
X1 = X1.reshape(-1,1)
X=torch.column_stack((X0,X1))
#np.savetxt("X_forsave.txt", X.cpu().detach().numpy())

E_ini=np.zeros(((ll-9)*(ll-9),1))*1.6
#E_ini=np.loadtxt('/home/yiming/yiming_research/linear/best_E_step44.txt', delimiter='\t')
#E_gd=np.loadtxt('/home/yiming/yiming_research/statebasedAC-master/statebasedAC-master/smallsamples20/baby30_0', delimiter='\t')
E_gd=np.loadtxt('./smallsamples20/baby30_0', delimiter='\t')
E_gd=np.reshape(E_gd,(ll,ll))
E_gd=np.reshape(E_gd[4:ll-5,4:ll-5],(-1,1))
#E_gd=np.expand_dims(E_gd,axis=1)
E_gd=-1.06*E_gd+3.72
E_gd=torch.tensor(E_gd, device=device)
time_endi=time.time()
print("Initialization finished, time cost: %f s." %(time_endi-time_starti))

print('Start preprocess.')
time_startp=time.time()
preprocess = preprocess(E_ini, meshsize, lenx, leny, X, delta_ratio)
matrix_dict=preprocess.ingredients()
time_endp=time.time()
print("Preprocess finished, time cost: %f s." %(time_endp-time_startp))

model = build_stiffness_matrix(matrix_dict,X)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

print("Start reading data. ")
time_startrd = time.time()
inputu, inputf=read_data_fromfile(N=n_samples,M=ll)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputu, inputf), batch_size=batch_size, shuffle=True)
time_endrd = time.time()
print("data reading finished, time cost: %f s." %(time_endrd-time_startrd))

#Emat1=torch.ones(((ll-9)*(ll-9),1),requires_grad=True)*1.6
#Emat2=torch.ones(((ll-9)*(ll-9),1),requires_grad=True)*2.66

train_loss_min=1e10
#E_error_min=1e10

#model_filename_restart = './test_fc1-2-50_fc2-50-100_fc3-100-1_lr2e-3_gamma050_nsamples100_repeat2_900samples_cont2_lr5e-3_gamma075/nnParameters.ckpt'
#model_filename_restart = './test/nnParameters.ckpt'
#model.load_state_dict(torch.load(model_filename_restart))
restart_index=0

for ep in range(epochs):
    optimizer = scheduler(optimizer, LR_schedule(learning_rate, ep%restart_ep, step_size, gamma))
    print("epoch %d started: " %(ep+restart_index))
    train_loss=0.0
    time_start=time.time()
    for inputu, inputf in train_loader:
        time_start1 = time.time()
        inputu = inputu.to(device)
        inputf = inputf.to(device)

        optimizer.zero_grad()

        #print("start assigning stiffness matrix")
        time_start2 = time.time()
        stiffness_matrix, E = model.build_matrix()
        time_end2 = time.time()
        # sm_forsave = stiffness_matrix.cpu().detach().numpy()
        # np.savetxt("./sm_forsave_new1.txt", sm_forsave)
        #print("stiffness matrix assigned, start computing loss")
        time_start4 = time.time()
        loss = rhs_error(stiffness_matrix, inputu, inputf)
        time_end4 = time.time()
        #print("loss computed, computing gradient")
        time_start3 = time.time()
        loss.backward()
        time_end3 = time.time()
        #print("gradient computed, update parameters")
        time_start5 = time.time()
        optimizer.step()
        time_end5 = time.time()
        time_end1 = time.time()
        #print("parameters updated, time cost: %f s, time build matrix cost: %f s, time loss computed: %f s, time loss backward cost: %f s, time optimizer step cost: %f s." %(time_end1-time_start1, time_end2-time_start2, time_end4-time_start4,time_end3-time_start3, time_end5-time_start5))

        train_loss += loss

    train_loss /= (n_samples/batch_size)
    E_error = torch.sum(torch.pow(E_gd-E.unsqueeze(1),2))

    if train_loss < train_loss_min:
        train_loss_min = train_loss
        best_E=E
        E_forsave=best_E.data.cpu().numpy()
        np.savetxt("./"+save_base_dir+"best_E_step"+str(ep+restart_index)+".txt", E_forsave)

        torch.save(model.state_dict(), model_filename)

    # if E_error < E_error_min:
    #     E_error_min = E_error
    E_error_percentage=E_error/torch.sum(torch.pow(E_gd,2))

    time_end = time.time()
    file=open('./'+save_base_dir+'loss.txt','a')
    file.write("epoch: %d, train loss: %f, best train loss: %f, E error: %f, E error percentage: %f, time cost: %f s.\n" % (ep+restart_index, train_loss, train_loss_min, E_error, E_error_percentage, time_end-time_start))
    print("epoch: %d, train loss: %f, best train loss: %f, E error: %f, E error percentage: %f, time cost: %f s" % (ep+restart_index, train_loss, train_loss_min, E_error, E_error_percentage, time_end-time_start))
