import numpy as np

K = 64
CP = 16 # 這是大多數 OFDM 範例程式碼預設的 CP 長度
L = CP   # 關鍵：頻道長度必須等於 CP

train_samples = 100000
test_samples = 390000

# 生成 (100000, 16) 的複數矩陣
# 這樣在 raputil.py 裡面 append 48 個零之後，長度就會剛好是 64
channel_train = (np.random.randn(train_samples, L) + 1j * np.random.randn(train_samples, L)) / np.sqrt(2)
channel_test = (np.random.randn(test_samples, L) + 1j * np.random.randn(test_samples, L)) / np.sqrt(2)
# 儲存檔案
import os
if not os.path.exists('tools'):
    os.makedirs('tools')
np.save('tools/channel_train.npy', channel_train)
np.save('tools/channel_test.npy', channel_test)

print(f"已生成頻道數據，維度為: {channel_train.shape}")
print("現在執行 main.py 應該就不會報錯了。")