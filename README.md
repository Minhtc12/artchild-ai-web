# Niên luận cơ sở KHMT – Hệ thống đánh giá tranh trẻ em vẽ

## Giới thiệu

AI Artwork Reviewer là một ứng dụng web viết bằng **Flask** cho phép:

- Nhận diện các đối tượng trong tranh trẻ em vẽ bằng **YOLOv8**.
- Phân loại thể loại tranh bằng **ResNet18**.
- Tự động chấm điểm bằng hàm **Constraint Programming** tích hợp trong `app.py`.
- Sinh nhận xét chi tiết bằng **Local LLM** chạy qua **LM Studio**.

## Cấu trúc thư mục

    ```text
    ARTCHILD_AI_WEB/
    │
    ├── runs/ # Kết quả train YOLO (nếu còn giữ)
    ├── static/ # File tĩnh (CSS, JS, ảnh demo)
    ├── templates/ # HTML giao diện Flask
    │ └── index.html
    ├── uploads/ # Ảnh upload (tạo khi chạy app)
    ├── venv/ # Môi trường ảo Python
    ├── .gitignore
    ├── app.py # File Flask chính
    ├── best_01.pt # Mô hình YOLOv8 đã huấn luyện (link Drive)
    ├── resnet18_4class_drawing.pth # Mô hình ResNet18 đã huấn luyện (link Drive)
    └── requirements.txt # Danh sách thư viện Python
    ```

## Tải mô hình

Do dung lượng lớn, 2 mô hình không được lưu trực tiếp trên GitHub.  
Vui lòng tải từ Google Drive và đặt vào thư mục gốc của project:

- **YOLOv8 model (`best_01.pt`)**: [Link tải](https://drive.google.com/file/d/1WyHo5bVpCA-PtXx1sngcBAwWxre7embq/view?usp=drive_link)
- **ResNet18 model (`resnet18_4class_drawing.pth`)**: [Link tải](https://drive.google.com/file/d/1TiW6VfKeX55WEEVhigQWRN0uwdfs5qk5/view?usp=drive_link)

## Yêu cầu cài đặt

- Python 3.8+
- pip
- LM Studio (để sinh nhận xét AI)

Tải LM Studio tại: [https://lmstudio.ai/](https://lmstudio.ai/)  
Sau khi cài, bạn cần:

1. Tải một mô hình LLM (gợi ý: **Vistral-7B-Chat(Q4_k_M)** hoặc tương tự).
2. Chạy mô hình trên LM Studio và bật **Local Inference API Server**.
3. Vào tab **Local Inference Server** → bật server và ghi nhớ địa chỉ **base_url**
4. Mở file `app.py` và thêm/chỉnh dòng kết nối sau:

```python
client = OpenAI(
    base_url="http://localhost:1234/v1",  # thay bằng địa chỉ server LM Studio của bạn
    api_key="vistral-7b-chat"             # Thay bằng API key của bạn
)
```

## Cài đặt & chạy ứng dụng

### Clone repo

```bash
git clone https://github.com/Minhtc12/artchild-ai-web.git
cd artchild-ai-web

### Tạo môi trường ảo và cài thư viện
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Đặt file mô hình vào thư mục gốc

    ```text
    ARTCHILD_AI_WEB/
        best_01.pt
        resnet18_4class_drawing.pth
    ```

### Chạy Flask app

```bash
python app.py
```

Mở trình duyệt và truy cập: http://127.0.0.1:5000
