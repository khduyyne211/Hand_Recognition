import cv2
import mediapipe as mp
import pickle
import numpy as np

# 1. Mở "hộp" lấy model đã train ra
print("Đang tải model...")
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
print("Tải model thành công!")

# 2. Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# 3. Mở Camera
cap = cv2.VideoCapture(0)

print("Bắt đầu nhận dạng. Nhấn 'Q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Lật camera cho tự nhiên giống soi gương
    frame = cv2.flip(frame, 1)
    
    # Chuyển BGR sang RGB cho MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    # 4. Xử lý Logic Dự Đoán (Inference)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Vẽ khung xương tay lên màn hình
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Làm phẳng 21 điểm thành 63 con số
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Chuyển list thành mảng 2 chiều (1 hàng, 63 cột) để model hiểu
            input_data = np.array([landmarks])
            
            # Đưa cho model dự đoán
            prediction = model.predict(input_data)
            predicted_char = prediction[0]
            
            # Hiển thị kết quả lên góc trên bên trái
            cv2.putText(frame, f"Chu: {predicted_char}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        # Nếu không thấy tay -> Đợi và xuất thông báo
        cv2.putText(frame, "Dang cho ban tay...", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị video
    cv2.imshow("Nhan Dang Cu Chi Tay ASL", frame)
    
    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()