import cv2
import numpy as np
from ultralytics import YOLO
from Crypto.Cipher import DES 
from Crypto.Util.Padding import pad
import time
import psutil
import os
import torch

# ==========================================
# TEST KONFİGÜRASYONU
# ==========================================
TEST_CORES = 4
TARGET_FPS = 10
VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aes_fhd.mp4')
YOLO_MODEL = 'yolov8n.pt' 
DES_MODE = 'ECB' 
CRYPT_KEY = b'87654321' # Tam 8 byte olmalı
DES_IV = b'12345678'    # CBC modu için gerekli 8 byte IV
# ==========================================

p = psutil.Process(os.getpid())
p.cpu_affinity(list(range(TEST_CORES)))
torch.set_num_threads(TEST_CORES)

def encrypt_image_region_des(region):
    raw_bytes = region.tobytes()
    
    # DES objesi oluşturma (Mod desteği eklendi)[cite: 1]
    if DES_MODE == 'ECB':
        cipher = DES.new(CRYPT_KEY, DES.MODE_ECB)
    elif DES_MODE == 'CBC':
        cipher = DES.new(CRYPT_KEY, DES.MODE_CBC, DES_IV)
    else:
        return region # Bilinmeyen mod durumunda orijinali döndür
    
    # Veriyi 8 byte bloklara tamamla
    padded_bytes = pad(raw_bytes, DES.block_size)
    encrypted_padded = cipher.encrypt(padded_bytes)
    
    # Orijinal boyuta sadık kalmak için fazlalığı at[cite: 3]
    encrypted_bytes = encrypted_padded[:len(raw_bytes)]

    return np.frombuffer(encrypted_bytes, dtype=np.uint8).reshape(region.shape)

model = YOLO(YOLO_MODEL) 
import os
print(f"Şu anki çalışma dizini: {os.getcwd()}")
print(f"Klasördeki dosyalar: {os.listdir()}")
cap = cv2.VideoCapture(VIDEO_PATH)

# Video dosyası kontrolü döngüden önce yapılmalıdır
if not cap.isOpened():
    print(f"Hata: '{VIDEO_PATH}' dosyası açılamadı! Dosya adını veya yolunu kontrol edin.")
    exit()

cv2.namedWindow('Sol: Orijinal | Sag: DES Sifreli', cv2.WINDOW_NORMAL)

original_fps = cap.get(cv2.CAP_PROP_FPS)
if original_fps == 0: original_fps = 30
frame_skip_interval = max(1, int(original_fps / TARGET_FPS))

frame_counter = 0
processed_frame_count = 0
total_processing_time = 0
total_encryption_time = 0

print(f"--- DES TEST BAŞLADI ---")
print(f"Şifreleme Modu: DES-{DES_MODE}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % frame_skip_interval != 0:
        continue

    encrypted_frame = frame.copy()
    t_start_frame = time.perf_counter()

    results = model.predict(frame, classes=[0], verbose=False)
    frame_encryption_time = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Görüntü sınırlarını kontrol et[cite: 4]
            human_roi = encrypted_frame[y1:y2, x1:x2]
            
            if human_roi.size > 0:
                t_start_enc = time.perf_counter()
                encrypted_frame[y1:y2, x1:x2] = encrypt_image_region_des(human_roi)
                frame_encryption_time += (time.perf_counter() - t_start_enc)

    t_end_frame = time.perf_counter()
    frame_processing_time = t_end_frame - t_start_frame
    
    processed_frame_count += 1
    total_processing_time += frame_processing_time
    total_encryption_time += frame_encryption_time

    fps_achieved = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
    
    cv2.putText(encrypted_frame, f"Mode: DES-{DES_MODE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(encrypted_frame, f"FPS: {fps_achieved:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    combined_frame = np.hstack((frame, encrypted_frame))
    cv2.imshow('Sol: Orijinal | Sag: DES Sifreli', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if processed_frame_count > 0:
    print(f"\n--- TEST SONUÇLARI ---")
    print(f"Ortalama İşlem: {(total_processing_time / processed_frame_count) * 1000:.2f} ms")