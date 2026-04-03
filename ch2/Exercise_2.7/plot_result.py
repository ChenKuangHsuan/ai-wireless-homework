import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 設定 SNR 範圍 (與 main.py 一致)
SNR_range = [5, 10, 15, 20, 25, 30, 35, 40]

# 假設你的四個檔案分別是 (請根據你實際的檔名修改)
files = [
    'MSE_ls_4QAM.mat', 
    'MSE_mmse_4QAM.mat', 
    'MSE_dnn_4QAM.mat',
    'MSE_dnn_4QAM_CP_FREE.mat',
    'MSE_ls_4QAM_CP_FREE.mat',
    'MSE_mmse_4QAM_CP_FREE.mat'
]

plt.figure(figsize=(8, 6))

for file in files:
    try:
        data = sio.loadmat(file)
        # 取得 key (通常與檔名相同，或是查看 main.py 最後的 savemat 部分)
        # 這裡自動找尋非 __header__ 的 key
        var_name = [k for k in data.keys() if not k.startswith('__')][0]
        mse_values = data[var_name].flatten()
        
        plt.semilogy(SNR_range, mse_values, '-o', label=var_name)
    except FileNotFoundError:
        print(f"找不到檔案: {file}，跳過...")

plt.xlabel('SNR (dB)')
plt.ylabel('Mean Square Error (MSE)')
plt.title('Channel Estimation Performance Comparison')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.show()