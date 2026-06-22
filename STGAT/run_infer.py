import torch, numpy as np, os, util
from model.stgat import STGAT

DEVICE = 'cuda'
OUT = '../integration/stgat_pred.npy'

_, _, adj_list = util.load_adj('data/METR-LA/adj_mx_dijsk.pkl', 'symnadj')
adj_mx = torch.from_numpy(np.array(adj_list))[0].to(DEVICE)

dataloader = util.load_dataset('data/METR-LA/', 1, 1, 1)
scaler = dataloader['scaler']

net = STGAT(True, 207, 2, 12, 12).to(DEVICE)
net.load_state_dict(torch.load('experiment_METR_LA/best_model.pth', map_location=DEVICE))
net.eval()

with torch.no_grad():
    for x, y in dataloader['test_loader']:
        x = x.to(DEVICE)
        out = net(adj_mx, x)                    # [1, 207, 12, 1]
        pred = out[:, :, 0, 0].cpu().numpy()    # 取下一步 [1, 207]（標準化後）
        break

pred_real = scaler.inverse_transform(pred)      # 反標準化 [1, 207]

os.makedirs('../integration', exist_ok=True)
np.save(OUT, pred_real)
print(f'STGAT prediction saved: {pred_real.shape} -> {OUT}')
print(f'Predicted speed range: {pred_real.min():.2f} ~ {pred_real.max():.2f} mph')