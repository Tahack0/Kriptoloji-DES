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
TEST_CORES = 8
TARGET_FPS = 10
VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'des_vga_480p.mp4')
YOLO_MODEL = 'yolov8n.pt' 
DES_MODE = 'CBC' 
CRYPT_KEY = b'87654321' 
DES_IV = b'12345678' 
# ==========================================

p = psutil.Process(os.getpid())
p.cpu_affinity(list(range(TEST_CORES)))
torch.set_num_threads(TEST_CORES)

# Sayaçları döngü öncesinde başlatıyoruz (Hata almamak için şart)[cite: 1, 3]
frame_counter = 0
processed_frame_count = 0
total_processing_time = 0
total_encryption_time = 0

def encrypt_image_region_des(region):
    raw_bytes = region.tobytes()
    if DES_MODE == 'CBC':
        cipher = DES.new(CRYPT_KEY, DES.MODE_CBC, DES_IV)
    else:
        cipher = DES.new(CRYPT_KEY, DES.MODE_ECB)
    
    padded_bytes = pad(raw_bytes, DES.block_size)
    encrypted_padded = cipher.encrypt(padded_bytes)
    encrypted_bytes = encrypted_padded[:len(raw_bytes)]
    return np.frombuffer(encrypted_bytes, dtype=np.uint8).reshape(region.shape)

model = YOLO(YOLO_MODEL) 
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Hata: '{VIDEO_PATH}' dosyası açılamadı!")
    exit()

cv2.namedWindow('DES Figur Sifreleme', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    encrypted_frame = frame.copy()
    t_start_frame = time.perf_counter()

    results = model.predict(frame, classes=[0], verbose=False)
    frame_encryption_time = 0

    for r in results:
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            for i, mask in enumerate(masks):
                x1, y1, x2, y2 = boxes[i]
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_roi = mask_resized[y1:y2, x1:x2]
                human_roi = encrypted_frame[y1:y2, x1:x2]
                bool_mask = mask_roi > 0.5
                if np.any(bool_mask):
                    t_start_enc = time.perf_counter()
                    human_pixels = human_roi[bool_mask]
                    encrypted_frame[y1:y2, x1:x2][bool_mask] = encrypt_image_region_des(human_pixels)
                    frame_encryption_time += (time.perf_counter() - t_start_enc)
        else:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                human_roi = encrypted_frame[y1:y2, x1:x2]
                if human_roi.size > 0:
                    t_start_enc = time.perf_counter()
                    encrypted_frame[y1:y2, x1:x2] = encrypt_image_region_des(human_roi)
                    frame_encryption_time += (time.perf_counter() - t_start_enc)

    # İstatistiklerin Hesaplanması[cite: 1, 3]
    t_end_frame = time.perf_counter()
    frame_processing_time = t_end_frame - t_start_frame
    
    processed_frame_count += 1
    total_processing_time += frame_processing_time
    total_encryption_time += frame_encryption_time

    fps_achieved = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
    enc_ms = frame_encryption_time * 1000

    # Bilgi Metinlerinin Ekrana Yazdırılması[cite: 1, 3]
    cv2.putText(encrypted_frame, f"Mod: DES-{DES_MODE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(encrypted_frame, f"Process FPS: {fps_achieved:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(encrypted_frame, f"DES Time: {enc_ms:.2f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
    
    combined_frame = np.hstack((frame, encrypted_frame))
    cv2.imshow('DES Figur Sifreleme', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# Final İstatistikleri[cite: 1, 3]
if processed_frame_count > 0:
    avg_proc_time = (total_processing_time / processed_frame_count) * 1000
    avg_enc_time = (total_encryption_time / processed_frame_count) * 1000
    print("\n--- TEST SONUÇLARI ---")
    print(f"Toplam İşlenen Kare: {processed_frame_count}")
    print(f"Kare Başına Ortalama Toplam İşlem: {avg_proc_time:.2f} ms")
    print(f"Kare Başına Ortalama DES Şifreleme: {avg_enc_time:.2f} ms")