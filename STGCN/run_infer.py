import torch, numpy as np, os, math, argparse
import pandas as pd
from sklearn import preprocessing
from model import models
from script import dataloader, utility

DEVICE = 'cuda'
OUT = '../integration/stgcn_pred.npy'

# === 重建與訓練時完全一致的 args（對照你的 main.py 預設值）===
args = argparse.Namespace(
    dataset='metr-la',
    n_his=12,
    n_pred=3,                       # main.py 預設是 3，不是 12
    Kt=3, Ks=3,
    stblock_num=2,
    act_func='glu',
    graph_conv_type='cheb_graph_conv',
    gso_type='sym_norm_lap',
    enable_bias=True,
    droprate=0.5,
)

# 組 blocks（完全複製 main.py 的邏輯）
Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
blocks = [[1]]
for _ in range(args.stblock_num):
    blocks.append([64, 16, 64])
blocks.append([128] if Ko == 0 else [128, 128])
blocks.append([1])

# 建 GSO
adj, n_vertex = dataloader.load_adj(args.dataset)
gso = utility.calc_gso(adj, args.gso_type)
gso = utility.calc_chebynet_gso(gso)
gso = gso.toarray().astype(np.float32)
args.gso = torch.from_numpy(gso).to(DEVICE)

# 載入並正規化資料（複製 main.py data_preparate）
dataset_path = os.path.join('./data', args.dataset)
data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
rate = 0.15
len_val = int(math.floor(data_col * rate))
len_test = int(math.floor(data_col * rate))
len_train = data_col - len_val - len_test

train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
zscore = preprocessing.StandardScaler()
train = zscore.fit_transform(train)
test = zscore.transform(test)
x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, DEVICE)

# 建模型、載入權重
model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(DEVICE)
model.load_state_dict(torch.load('STGCN_metr-la.pt', map_location=DEVICE))
model.eval()

# 推論第一筆
with torch.no_grad():
    pred = model(x_test[:1]).view(1, -1)        # [1, 207]（標準化後）
pred = pred.cpu().numpy()
pred_real = zscore.inverse_transform(pred)      # 反標準化回真實速度 [1, 207]

os.makedirs('../integration', exist_ok=True)
np.save(OUT, pred_real)
print(f'STGCN prediction saved: {pred_real.shape} -> {OUT}')
print(f'Predicted speed range: {pred_real.min():.2f} ~ {pred_real.max():.2f} mph')