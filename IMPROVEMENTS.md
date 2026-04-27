# Danh sách các điểm cần cải thiện cho App ASL

### 1. Chuẩn hóa toạ độ Landmark (Rất quan trọng) - ĐÃ XỬ LÝ VÀ CHUẨN HOÁ
* Đã áp dụng `utils.py` để module hoá việc xử lý landmark.
* Logic chuẩn hoá: Tính mốc từ gốc tọa độ `(0, 0, 0)` tại cổ tay, scale các toạ độ chia cho khoảng cách tổng thể lớn nhất để khử sự khác biệt về khoảng cách tay - camera và góc nhìn trước khi đưa vào mô hình hoặc lưu file csv.

### 2. Xung đột ghi đè tập dữ liệu (Lỗi Data Pipeline) - ĐÃ SỬA LỖI
* Dữ liệu từ Kaggle giờ được bóc tách riêng ra `data_kaggle.csv`.
* Dữ liệu tự quay tay được nối tiếp vào `data_custom.csv`.
* Trong `train_model.py`, đã được gộp (concat) dữ liệu thông minh từ 2 nguồn nếu có.

### 3. Tối ưu hoá việc nhận diện chữ chuyển động (J và Z) - ĐÃ XỬ LÝ
* Trích xuất các ngưỡng số cứng (`J_DY_THRESHOLD`, `J_DX_THRESHOLD`, `Z_DX_THRESHOLD`, `Z_DIR_CHANGES_THRESHOLD`) thành hằng số để dễ dàng config trên đầu file app.

### 4. Thiếu Try-Catch và phòng ngừa lỗi (Application robustness) - ĐÃ XỬ LÝ
* Thêm kiểm tra `cap.isOpened()` và hiển thị Error Exception nếu Camera bị lỗi.
* Bọc logic load model `with open(...)` bằng `try - except FileNotFoundError` kèm theo `exit()` để tránh lỗi văng App khó hiểu.

### 5. Lặp lại code (Nguyên tắc DRY) - ĐÃ XỬ LÝ CÙNG MỤC 1
* Đã tạo `utils.py` chứa hàm trích xuất và chuẩn hóa tọa độ chung, tái sử dụng tại tất cả các file liên quan (app, extract, collect).

### 6. Cải tiến model huấn luyện - ĐÃ XỬ LÝ
* Áp dụng `GridSearchCV` (`sklearn.model_selection`) cho `RandomForestClassifier` trong `train_model.py`. Quá trình này sẽ giúp máy đo đạc từ thư viện các bộ tham số `n_estimators` và `max_depth` để chọn ra combo mang lại accuracy đỉnh cao nhất.
