import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(r'E:/Class/AIAA 3111/Project'); DATA=ROOT/'data/telemanom/data'; OUT=ROOT/'results_v0'; OUT.mkdir(exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(42); np.random.seed(42)
labels=pd.read_csv(DATA/'../labeled_anomalies.csv'); chans=[x for x in labels[labels.spacecraft=='SMAP'].chan_id if (DATA/'train'/f'{x}.parquet').exists()]
L=64; stride=8; max_train_windows=60000
train_windows=[]; val_windows=[]; test_info=[]
for chan in chans:
 tr=pd.read_parquet(DATA/'train'/f'{chan}.parquet'); te=pd.read_parquet(DATA/'test'/f'{chan}.parquet')
 feats=[c for c in tr.columns if c!='timestep']; Xtr=tr[feats].to_numpy(np.float32); Xte=te[feats].to_numpy(np.float32)
 mu=np.nanmean(Xtr,0); sd=np.nanstd(Xtr,0); sd[sd<1e-6]=1
 Xtr=(np.nan_to_num(Xtr,nan=0.0)-mu)/sd; Xte=(np.nan_to_num(Xte,nan=0.0)-mu)/sd
 cut=int(.8*len(Xtr))
 train_windows.extend([Xtr[i:i+L] for i in range(0,max(0,cut-L+1),stride)])
 val_windows.extend([Xtr[i:i+L] for i in range(cut,max(cut, len(Xtr)-L+1),stride)])
 seqs=json.loads(labels.loc[labels.chan_id==chan,'anomaly_sequences'].iloc[0].replace("'",'"'))
 y=np.zeros(len(Xte),np.int8)
 for a,b in seqs:
  a,b=int(a),min(int(b),len(y)-1)
  if a<len(y): y[a:b+1]=1
 test_info.append((chan,Xte,y,mu,sd))
if len(train_windows)>max_train_windows:
 idx=np.random.choice(len(train_windows),max_train_windows,replace=False); train_windows=[train_windows[i] for i in idx]
Xtrain=torch.tensor(np.stack(train_windows),dtype=torch.float32); Xval=torch.tensor(np.stack(val_windows),dtype=torch.float32)
print('windows',Xtrain.shape,Xval.shape,'device',DEVICE)
class AE(nn.Module):
 def __init__(self,d=25,h=64):
  super().__init__(); self.enc=nn.LSTM(d,h,batch_first=True); self.dec=nn.LSTM(h,h,batch_first=True); self.out=nn.Linear(h,d)
 def forward(self,x):
  _,(h,_) = self.enc(x); z=h[-1].unsqueeze(1).repeat(1,x.size(1),1); y,_=self.dec(z); return self.out(y)
model=AE().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5); lossfn=nn.MSELoss(); loader=DataLoader(TensorDataset(Xtrain),batch_size=256,shuffle=True,pin_memory=True)
hist=[]; t0=time.time()
for ep in range(1,6):
 model.train(); total=0
 for (xb,) in loader:
  xb=xb.to(DEVICE,non_blocking=True); opt.zero_grad(set_to_none=True); loss=lossfn(model(xb),xb); loss.backward(); opt.step(); total+=loss.item()*len(xb)
 hist.append({'epoch':ep,'train_mse':total/len(Xtrain)}); print(hist[-1])
model.eval();
with torch.no_grad(): val_err=((model(Xval.to(DEVICE))-Xval.to(DEVICE))**2).mean(dim=(1,2)).cpu().numpy()
thr=float(np.quantile(val_err,0.995)); print('threshold',thr)
# score test end points, leave preceding values at 0
all_y=[]; all_s=[]; event_hit=event_total=0; delays=[]; per=[]
for chan,Xte,y,mu,sd in test_info:
 scores=np.zeros(len(Xte),np.float32); starts=range(0,max(0,len(Xte)-L+1),stride)
 for i in starts:
  xb=torch.tensor(Xte[i:i+L][None],dtype=torch.float32,device=DEVICE)
  with torch.no_grad(): e=float(((model(xb)-xb)**2).mean().cpu())
  scores[i+L-1]=e
 pred=scores>thr; all_y.append(y); all_s.append(scores)
 seqs=json.loads(labels.loc[labels.chan_id==chan,'anomaly_sequences'].iloc[0].replace("'",'"'))
 for a,b in seqs:
  event_total+=1; a=int(a); b=min(int(b),len(pred)-1); ix=np.where(pred[a:b+1])[0]
  if len(ix): event_hit+=1; delays.append(int(ix[0]))
 tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
 per.append({'channel':chan,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'pred_points':int(pred.sum())})
y=np.concatenate(all_y); s=np.concatenate(all_s); pred=s>thr; tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; F1=2*P*R/(P+R) if P+R else 0
summary={'dataset':'SMAP','method':'global LSTM autoencoder','device':DEVICE,'window_length':L,'stride':stride,'train_windows':len(Xtrain),'validation_windows':len(Xval),'epochs':5,'threshold':thr,'runtime_s':time.time()-t0,'pointwise':{'precision':P,'recall':R,'f1':F1,'fpr':fp/(fp+tn),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'roc_auc':float(roc_auc_score(y,s)),'pr_auc':float(average_precision_score(y,s))},'event_level':{'detected':event_hit,'total':event_total,'recall':event_hit/event_total,'mean_delay_points':float(np.mean(delays)) if delays else None},'history':hist}
(OUT/'smap_lstm_ae_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8'); pd.DataFrame(per).to_csv(OUT/'smap_lstm_ae_channel_results.csv',index=False); torch.save({'model':model.state_dict(),'threshold':thr,'window_length':L},OUT/'smap_lstm_ae.pt'); print(json.dumps(summary,indent=2))
