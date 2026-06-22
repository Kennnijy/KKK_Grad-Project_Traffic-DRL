# 在 STGAT 資料夾執行: python check_data.py
import numpy as np
d = np.load('data/METR-LA/train.npz')
x, y = d['x'], d['y']
print('x shape:', x.shape, '| y shape:', y.shape)

xs, ys = x[..., 0], y[..., 0]   # feature 0 = 速度
print('速度 mean/std:', round(xs.mean(),2), round(xs.std(),2))
print('速度 min/max:', round(xs.min(),2), round(xs.max(),2))
print('x 中 0 的比例:', round((xs==0).mean(),4))
print('y 中 0 的比例:', round((ys==0).mean(),4))

# 對齊檢查：x 最後一幀 與 y 第一幀（前 5 個 sensor）應該時間連續
print('x[0] 最後一幀:', xs[0, -1, :5])
print('y[0] 第一幀  :', ys[0, 0, :5])

# persistence baseline（遮罩掉 0 後）
pred = np.repeat(xs[:, -1:, :], ys.shape[1], axis=1)   # 用最後一幀預測全部 horizon
mask = ys != 0
print('persistence MAE (masked):', round(np.abs(pred-ys)[mask].mean(), 3))