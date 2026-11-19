import cv2
import numpy as np
from ultralytics import YOLO
from Crypto.Cipher import DES 
from Crypto.Util.Padding import pad
import time
import psutil
import os
import torch


# DES BATCH TEST KONFİGÜRASYONU

YOLO_MODEL = 'yolov8n-seg.pt'  
CRYPT_KEY = b'87654321' # DES için tam 8 byte[cite: 1]
DES_IV = b'12345678'    # CBC için 8 byte IV[cite: 1]
OUTPUT_FILE = 'des_test_sonuclari.txt'


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATHS_LIST = [
    os.path.join(BASE_PATH, 'des_vga_480p.mp4'), 
    os.path.join(BASE_PATH, 'des_hd.mp4'), 
    os.path.join(BASE_PATH, 'des_fhd.mp4')
]
TEST_CORES_LIST = [2, 4, 8]
TARGET_FPS_LIST = [10, 20]
DES_MODES_LIST = ['ECB', 'CBC'] 


with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("OTOMATİK DES BATCH TEST BAŞLADI\n\n")

model = YOLO(YOLO_MODEL) 
test_no = 1

for video_path in VIDEO_PATHS_LIST:
    for fps in TARGET_FPS_LIST:
        for cores in TEST_CORES_LIST:
            for mode in DES_MODES_LIST:
                
                TEST_CORES = cores
                TARGET_FPS = fps
                DES_MODE = mode
                CURRENT_VIDEO = video_path
                
                # İşlemci çekirdek sınırlaması
                p = psutil.Process(os.getpid())
                try:
                    p.cpu_affinity(list(range(TEST_CORES)))
                except AttributeError:
                    pass
                torch.set_num_threads(TEST_CORES)

                def encrypt_image_region_des(region):
                    raw_bytes = region.tobytes()
                    # Mod seçimine göre DES objesi oluşturma
                    if DES_MODE == 'CBC':
                        cipher = DES.new(CRYPT_KEY, DES.MODE_CBC, DES_IV)
                    else:
                        cipher = DES.new(CRYPT_KEY, DES.MODE_ECB)
                    
                    padded_bytes = pad(raw_bytes, DES.block_size)
                    encrypted_padded = cipher.encrypt(padded_bytes)
                    encrypted_bytes = encrypted_padded[:len(raw_bytes)]
                    return np.frombuffer(encrypted_bytes, dtype=np.uint8).reshape(region.shape)

                cap = cv2.VideoCapture(CURRENT_VIDEO)
                if not cap.isOpened():
                    print(f"Atlanıyor: {CURRENT_VIDEO} bulunamadı.")
                    continue

                original_fps = cap.get(cv2.CAP_PROP_FPS)
                if original_fps == 0: original_fps = 30
                frame_skip_interval = max(1, int(original_fps / TARGET_FPS))
                
                vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                frame_counter = 0
                processed_frame_count = 0
                total_processing_time = 0
                total_encryption_time = 0
                total_encrypted_bytes = 0

                print(f"[{test_no}] Başlıyor: Video={CURRENT_VIDEO}, Cores={TEST_CORES}, Mode={DES_MODE}")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_counter += 1
                    if frame_counter % frame_skip_interval != 0:
                        continue

                    encrypted_frame = frame.copy()
                    t_start_frame = time.perf_counter()

                    # YOLO Segmentasyon tahmini
                    results = model.predict(frame, classes=[0], verbose=False)
                    frame_encryption_time = 0

                    for r in results:
                        if r.masks is None:
                            continue
                            
                        masks = r.masks.data.cpu().numpy()
                        boxes = r.boxes.xyxy.cpu().numpy().astype(int)

                        for i, mask in enumerate(masks):
                            x1, y1, x2, y2 = boxes[i]
                            
                            # Kenar boşluğu (Padding) ve sınır kontrolü
                            p_val = 30
                            x1, y1 = max(0, x1 - p_val), max(0, y1 - p_val)
                            x2, y2 = min(frame.shape[1], x2 + p_val), min(frame.shape[0], y2 + p_val)
                            
                            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                            mask_roi = mask_resized[y1:y2, x1:x2]
                            human_roi = encrypted_frame[y1:y2, x1:x2]
                            
                            # Maskeyi genişletme (Dilation)
                            bool_mask = mask_roi > 0.2
                            kernel = np.ones((25, 25), np.uint8) 
                            dilated_mask = cv2.dilate(bool_mask.astype(np.uint8), kernel, iterations=1)
                            bool_mask = dilated_mask > 0
                            
                            if np.any(bool_mask):
                                human_pixels = human_roi[bool_mask]
                                t_start_enc = time.perf_counter()
                                
                                total_encrypted_bytes += human_pixels.nbytes
                                encrypted_pixels = encrypt_image_region_des(human_pixels)
                                
                                t_end_enc = time.perf_counter()
                                frame_encryption_time += (t_end_enc - t_start_enc)
                                human_roi[bool_mask] = encrypted_pixels

                    t_end_frame = time.perf_counter()
                    processed_frame_count += 1
                    total_processing_time += (t_end_frame - t_start_frame)
                    total_encryption_time += frame_encryption_time

                    # Görselleştirme 
                    combined_frame = np.hstack((frame, encrypted_frame))
                    cv2.putText(combined_frame, f"TEST: {test_no} | DES-{DES_MODE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                   
                    display_width = 1280
                    display_height = 720
                    resized_display = cv2.resize(combined_frame, (display_width, display_height))

                    cv2.imshow('Otomatik DES Testi', resized_display)



                    if cv2.waitKey(1) & 0xFF == 27: 
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                cap.release()
                
                # İstatistikleri hesapla ve dosyaya yaz
                if processed_frame_count > 0:
                    avg_proc_time = (total_processing_time / processed_frame_count) * 1000
                    avg_enc_time = (total_encryption_time / processed_frame_count) * 1000
                    total_mb = total_encrypted_bytes / (1024 * 1024)
                    throughput_mb_s = total_mb / total_encryption_time if total_encryption_time > 0 else 0
                    
                    sonuc_metni = (
                        f"TEST SONUCU - {test_no}\n"
                        f"Video: {CURRENT_VIDEO} ({vid_width}x{vid_height})\n"
                        f"Parametreler: {TARGET_FPS} FPS, DES-{DES_MODE}, Çekirdek: {TEST_CORES}\n"
                        f"İşlenen Kare: {processed_frame_count}\n"
                        f"Toplam Veri: {total_mb:.2f} MB\n"
                        f"Ort. İşlem: {avg_proc_time:.2f} ms/kare\n"
                        f"Ort. DES Süresi: {avg_enc_time:.2f} ms/kare\n"
                        f"Hız: {throughput_mb_s:.2f} MB/s\n"
                        f"---------------------------------------------\n"
                    )
                    
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                        f.write(sonuc_metni)
                
                test_no += 1 

cv2.destroyAllWindows()
print(f"\nTüm testler tamamlandı. Sonuçlar '{OUTPUT_FILE}' dosyasında.")