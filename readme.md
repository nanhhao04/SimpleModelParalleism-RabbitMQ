
Dự án minh họa cách **chia nhỏ mô hình AI (Model Parallelism)** giữa hai thiết bị (hoặc hai tiến trình) —  
mỗi thiết bị giữ một phần của mạng neural, giao tiếp qua **RabbitMQ message broker**.




###  Model Parallelism
<img src="pics/model_parallelism.png" width="300">

- Mô hình được chia tại điểm `split_point` thành 2 phần:
  - **Device A (device1)**: xử lý input → tạo activation.
  - **Device B (device2)**: nhận activation → tính output, loss, gradient.
- Sau khi backward, `Device2` gửi gradient ngược lại cho `Device1`.

###  RabbitMQ Communication
<img src="pics/rabbitmq-flow.png" width="300">

- **forward_queue**: Device1 ➜ Device2 (activation + label)  
- **backward_queue**: Device2 ➜ Device1 (gradient)
---

## Các bước thức hiện

### 1️. Cài Python packages
```bash
pip install -r requirements.txt
```

### 2. Chạy RabbitMQ bằng Docker Compose
```bash
docker compose up -d
```
---

## Chạy dự án

Mở **2 terminal** riêng biệt:

### Terminal 1 — Device 2 (nhận forward)
```bash
python device2.py
```

### Terminal 2 — Device 1 (gửi forward)
```bash
python device1.py
```
