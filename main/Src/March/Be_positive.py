
import torch
import torch.nn.functional as F 
from timeit import default_timer as timer   

from torch.nn.utils.rnn import pad_sequence

ho = torch.zeros((40,40))
print("hohoho",ho.shape)

parmdic = {'ok': [1.,1.,3], 'okk':[1.,1.,3], 'okkk':[1.,1.,3], 'not okkk':[-1.,-3], 'not okkkk':[-4.,-1.,-3]}
padded_sequences = [torch.tensor(seq) for seq in parmdic.values()]
parm = pad_sequence(padded_sequences, batch_first=True, padding_value=0)

print(parm)

start = timer()
#parm = (torch.tensor(list(parmdic.values())))
be_positive = torch.zeros(0)  # Initialize as empty tensor
for values in parm:
    neg_values = values[values < 0]
    if neg_values.numel()>0:
        be_positive = torch.cat((be_positive, torch.sum(F.relu(-neg_values)).unsqueeze(0)))
   
be_p = be_positive.sum() * 2
print("without GPU:", (timer()-start), "seconds") #/ 60
print(be_p)
start2 = timer()
#parm = (torch.tensor(list(parmdic.values())))
be_positive = torch.zeros(1) 
neg_param_values = [torch.sum(F.relu(-values[values<0])) for values in parm if torch.any(values < 0)]#parmdic.values() 
if neg_param_values:
    be_positive = torch.sum(torch.stack(neg_param_values))
be_p = be_positive * 2
print("without GPU:", (timer()-start2), " seconds") 
print(be_p)
       
