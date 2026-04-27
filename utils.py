import numpy as np

def process_landmarks(hand_landmarks):
    """
    Trích xuất và chuẩn hóa landmark:
    - Đưa gốc tọa độ về cổ tay (landmark 0).
    - Chuẩn hóa tỷ lệ (scale) theo khoảng cách xa nhất để không bị ảnh hưởng bởi cự ly/kích thước tay.
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    
    landmarks = np.array(landmarks)
    
    # Trừ đi tọa độ điểm gốc (cổ tay - index 0)
    base = landmarks[0]
    landmarks = landmarks - base
    
    # Scale theo khoảng cách lớn nhất từ cổ tay tới các ngón tay
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
        
    # Flatten thành mảng 1 chiều (63 phần tử) gồm (x0, y0, z0, x1, y1, ...)
    return landmarks.flatten().tolist()
