# Dự Án Phân Loại Cường Độ Ăn Của Cá Bằng Âm Thanh (Swin Transformer)

Dự án này sử dụng mô hình **Swin Transformer Tiny** để phân loại cường độ ăn của cá dựa trên dữ liệu âm thanh (Log-Mel Spectrogram).

## 1. Cấu Trúc Dự Án

```
SLP_Fish_Feeding_Intensity_Using_Audio/
├── dataset/                  # Chứa dữ liệu âm thanh gốc (.wav) chia theo thư mục class
├── processed_data/           # Chứa dữ liệu đã tiền xử lý (.pt) - Tự động tạo
├── checkpoints/              # Chứa file trọng số mô hình sau khi train (.pth)
├── models/                   # Chứa kiến trúc mô hình (BaseModel, SwinTransformer)
├── trainers/                 # Chứa logic huấn luyện (BaseTrainer, SwinTrainer)
├── evaluation/               # Chứa logic đánh giá (BaseEvaluator)
├── preprocessing/            # Mã nguồn tiền xử lý dữ liệu
├── train.py                  # File chạy huấn luyện
├── evaluate.py               # File chạy đánh giá
├── requirements.txt          # Các thư viện cần thiết
└── README.md                 # Hướng dẫn sử dụng
```

## 2. Cài Đặt Môi Trường

Đảm bảo bạn đã cài đặt Python (khuyến nghị 3.8+).

1.  **Cài đặt các thư viện phụ thuộc:**
    ```bash
    pip install -r requirements.txt
    ```

## 3. Chuẩn Bị Dữ Liệu

1.  Đặt các file âm thanh `.wav` vào thư mục `dataset/` theo cấu trúc sau:
    ```
    dataset/
    ├── none/
    │   ├── file1.wav
    │   └── ...
    ├── weak/
    ├── middle/
    └── strong/
    ```
    *(Lưu ý: Tên thư mục phải khớp với `CLASS_MAPPING` trong `preprocessing/config_preprocess.py`)*

2.  **Không cần chạy thủ công:** File `train.py` sẽ tự động kiểm tra và chạy tiền xử lý (cắt Log-Mel Spectrogram, Resize 224x224) nếu chưa có dữ liệu.

## 4. Huấn Luyện Mô Hình (Training)

Để bắt đầu huấn luyện, chạy lệnh:

```bash
python train.py
```

**Quá trình diễn ra:**
1.  Hệ thống tự động load dữ liệu và resize về `224x224`.
2.  Tải trọng số Pre-trained của Swin Transformer (ImageNet).
3.  Huấn luyện trong 30 epochs (mặc định).
4.  Mô hình có độ chính xác (Validation Accuracy) cao nhất sẽ được lưu tại:
    `checkpoints/swin_tiny/best_model.pth`

## 5. Đánh Giá Mô Hình (Evaluation)

Sau khi train xong, để kiểm tra kỹ lưỡng mô hình trên tập Test (báo cáo chi tiết Precision, Recall, F1-Score, Confusion Matrix), chạy lệnh:

```bash
python evaluate.py
```

## 6. Tùy Chỉnh (Nâng Cao)

*   **Thay đổi tham số Train:** Mở `train.py` để sửa `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`.
*   **Thay đổi cấu hình âm thanh:** Mở `preprocessing/config_preprocess.py`.
*   **Thêm mô hình mới:**
    1.  Tạo file model mới trong `models/` (kế thừa `BaseModel`).
    2.  Sửa `train.py` để import model mới.

---
**Lưu ý:** Nếu gặp lỗi "Out of Memory" (OOM) khi chạy trên GPU, hãy giảm `BATCH_SIZE` trong `train.py` xuống (ví dụ: 16 hoặc 8).
