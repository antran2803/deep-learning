# BTL03 – Chuyển code Keras sang PyTorch


## Cấu trúc thư mục

```
BTL03_TranQuocAn_3123411009/
├── BTL03_TranQuocAn_3123411009.docx      # báo cáo
├── README.md · requirements.txt · .gitignore
├── notebooks/                            # 10 notebook PyTorch (đã chạy, có sẵn output)
│   ├── 1_MLP/                Func_apxm_MLP_pytorch.ipynb, MNIST_MLP_pytorch.ipynb
│   ├── 2_CNN_MLP/            1_MNIST_CNN_pytorch.ipynb, 2_Human_Gender_CNN_pytorch.ipynb, 3_Emotion_Reg_CNN_pytorch.ipynb
│   ├── 3_CNN_LSTM/           MNIST_CNN_LSTM_pytorch.ipynb
│   ├── 4_Training_Utilities/ 2_Human_Gender_CNN_pytorch.ipynb, 3_Emotion_Reg_CNN_pytorch.ipynb
│   └── 5_Others/             1_segnet_pytorch.ipynb, 2_obj_detection_pytorch.ipynb
├── data/                                 # KHÔNG đưa lên git – tải theo bảng bên dưới
└── outputs/                              # checkpoint, log sinh ra khi chạy (không đưa lên git)
```

Mỗi notebook có cùng bố cục với bản Keras gốc: **(1) Data → (2) Declare model → (3) Train → (4) Evaluate**, mở đầu bằng
bảng đối chiếu Keras → PyTorch và in tổng số tham số để so với bản Keras.

## Dữ liệu

Ảnh và dataset không được đưa lên git (xem `.gitignore`). Tải và đặt vào `data/` như sau:

| Notebook | Dữ liệu | Cách lấy | Đặt vào |
|---|---|---|---|
| Func_apxm_MLP | tự sinh trong code | – | – |
| MNIST_MLP, 1_MNIST_CNN, MNIST_CNN_LSTM | MNIST (`mnist.npz`) | **tự tải** khi chạy từ `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz` | `data/mnist.npz` |
| 1_segnet | ảnh độ sâu bàn tay | giải nén `Example/examples/5. Others/depth_hand.zip`, hoặc tải: https://drive.google.com/file/d/1MRiBnDW9dE7mNb5bL-jzhmtTd2CSklvL/view?usp=sharing | `data/segnet/{train,test}/{depth,labels}/` |
| 2_obj_detection | ảnh xe / không phải xe | giải nén `Example/examples/5. Others/car_data.zip`, hoặc tải: https://drive.google.com/file/d/1rqitUFrFhB84tFVtg1Ad1TVz6coCWpNZ/view?usp=sharing ; ảnh thử `Example/examples/5. Others/test.jpg` | `data/obj_detection/{vehicles,non-vehicles}/`, `data/test.jpg` |
| 2_Human_Gender_CNN (2 bản) | WIKI faces 64×64 (`.npy`) | https://drive.google.com/drive/folders/0BxINLo5jshCRYW8xODhNSlkyLTQ?resourcekey=0-RNJvvSSE67Q1bMqQi3A1RQ&usp=sharing | `data/gender/64_64_*_onehot.npy` (4 file) |
| 3_Emotion_Reg_CNN (2 bản) | FER2013 (`.csv`) | https://drive.google.com/drive/folders/0BxINLo5jshCRbXpRcnl2SjJkdlk?resourcekey=0-7qr7Ro7hgkmFknsN1yParQ&usp=sharing | `data/emotion/fer2013_training_onehot.csv`, `data/emotion/fer2013_publictest_onehot.csv` |

Giải nén nhanh hai file zip có sẵn (chạy tại thư mục này; zip đã chứa sẵn thư mục `data/`):

```bash
unzip "../Example/examples/5. Others/depth_hand.zip" -x "*.DS_Store"
unzip "../Example/examples/5. Others/car_data.zip"   -x "*.DS_Store"
cp "../Example/examples/5. Others/test.jpg" data/test.jpg
```

> Dữ liệu Gender và Emotion không có trong thư mục `Example` (phải tải từ Google Drive của thầy). Khi chưa có, hai cặp
> notebook này **tự chạy bằng dữ liệu giả** để kiểm tra code và in cảnh báo `!!! ... DỮ LIỆU GIẢ` – số liệu khi đó
> không có ý nghĩa. Đặt dữ liệu thật vào đúng chỗ rồi chạy lại là được, không cần sửa code.

## Cách chạy

```bash
python -m venv .venv && .venv\Scripts\activate     # tuỳ chọn (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook                                   # mở notebook trong notebooks/ → Run All
```

Có thể mở bằng VS Code (Jupyter extension) hoặc tải lên Google Colab (GPU). Notebook tự tìm thư mục `data/` ở thư mục
cha gần nhất và tự dùng GPU nếu có. Không cần `torchvision` hay `scikit-image`.

### Cấu hình epoch khi chạy trên CPU

Các notebook nặng được đặt hằng số ở cell đầu để chạy được trên CPU; có GPU thì đổi lại giá trị gốc:

| Notebook | Mặc định trong bài | Keras gốc | Thời gian/epoch trên CPU 12 nhân |
|---|---|---|---|
| 1_MNIST_CNN | `EPOCHS = 10` | 30 | ~2,3 phút |
| MNIST_CNN_LSTM | `N_TRAIN, N_TEST, EPOCHS = 10000, 2000, 10` | 60000, 10000, 30 | ~2,8 phút (17 phút với 60000 mẫu) |
| 1_segnet | `EPOCHS = 3` | 15 | ~9 phút |
| Các notebook còn lại | giống bản gốc | – | < 1 phút |

## Kết quả (CPU, đã chạy và lưu trong notebook)

| Notebook | Chỉ số (test) | Keras gốc | PyTorch | Ghi chú |
|---|---|---|---|---|
| Func_apxm_MLP | MSE | – | 0,0083 (A=1000) · 0,0014 (B=512) · 0,0015 (C=190) | sai số giảm khi tăng A, B, C |
| MNIST_MLP | Accuracy | 96,48% | 96,39% | giống cấu hình gốc (30 epoch) |
| 1_MNIST_CNN | Accuracy | 98,84% (30 ep) | 97,66% (10 ep) | giảm epoch vì CPU |
| MNIST_CNN_LSTM | MAE | 0,6502 (60 000 mẫu, 30 ep) | 3,0204 (10 000 mẫu, 10 ep) | giảm dữ liệu + epoch, chưa hội tụ |
| 1_segnet | Pixel accuracy | 77,32% (15 ep) | 78,17% (3 ep) | đã sửa lỗi `Reshape` của bản gốc |
| 2_obj_detection | Accuracy | 95,23% | 96,83% | giống cấu hình gốc (5 epoch) |
| 2_Human_Gender_CNN (×2) | Accuracy | 82,64% / 87,73% | – | chưa có dữ liệu, mới chạy bằng dữ liệu giả |
| 3_Emotion_Reg_CNN (×2) | Accuracy | 59,54% / 61,63% | – | chưa có dữ liệu, mới chạy bằng dữ liệu giả |

Số tham số của các mô hình khớp với Keras, trừ CNN-LSTM: bản PyTorch nhiều hơn 1 024 tham số vì `nn.LSTM` có thêm một vector bias.

Chi tiết phương pháp chuyển đổi, so sánh và các vấn đề phát hiện trong code gốc: xem báo cáo `.docx`.
