import cv2
import os
import numpy as np
import time
import sys

# Добавляем текущую папку в путь поиска модулей, чтобы импорт сработал
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Импортируем движок распознавания
from gesture_engine import GestureRecognizer

# Поиск картинок
project_root = os.path.dirname(current_dir)

def find_asset(filename):
    "Поиск папки assets"
    path = os.path.join(project_root, "assets", filename)
    return path if os.path.exists(path) else filename

# Загрузка картинок
memes = {}
for class_id in ['1', '2', '3', '4']:
    img_path = find_asset(f"meme_{class_id}.png")
    
    if os.path.exists(img_path):
        # Загружаем картинку, флаг IMREAD_UNCHANGED нужен, чтобы добавить прозрачность
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is not None:
            # Если у картинки 4 канала
            if len(img.shape) == 3 and img.shape[2] == 4:
                # Меняем прозрачный фон на полностью черный
                # Выделяем 4-й канал и переводим его в диапазон от 0.0 до 1.0
                alpha_channel = img[:, :, 3] / 255.0
                
                # Создаем абсолютно черный холст такого же размера
                black_bg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                
                # Проходим по каждому цвету (Синий-0, Зеленый-1, Красный-2)
                for c in range(3):
                    # Умножаем цвет на маску, там, где прозрачно, пиксель станет черным
                    # Там, где не прозрачно (1.0), оставляем оригинальный цвет.
                    black_bg[:, :, c] = (alpha_channel * img[:, :, c]).astype(np.uint8)
                
                img = black_bg # Заменяем оригинальную картинку на обработанную
            
            # Сохраняем готовую картинку в словарь
            memes[class_id] = img
    else:
        print(f"Внимание: Картинка {img_path} не найдена!")

# Инициализация движка
print("Загрузка нейросетей...")
recognizer = GestureRecognizer()

# Переменные для таймера
current_stable_gesture = None
gesture_start_time = time.time() # Время, когда мы впервые заметили жест
showing_meme_until = 0 # Временная метка: до какой секунды показывать мем
active_meme_class = None # Какой именно мем сейчас на весь экран

HOLD_TIME = 0.3 # Сколько секунд держать жест для срабатывания
MEME_DISPLAY_TIME = 0.3  # Сколько секунд висит мем на весь экран

# Запуск камеры
cap = cv2.VideoCapture(1) 
print("\nMemeCam запущена! Покажи жест в камеру. Для выхода нажми 'q'")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Отзкркаливание камеры 
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Показ картинки на весь экран
    # Если текущее время меньше, чем время, до которого мы должны показывать мем
    if time.time() < showing_meme_until:
        if active_meme_class in memes:
            # Растягиваем мем на весь экран
            frame = cv2.resize(memes[active_meme_class], (w, h))
        cv2.imshow("MemeCam (AI Gestures)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue # Пропускаем поиск руки, пока висит мем

    # Обращаемся к движку, получаем информацию о текущем жесте 
    stable_gesture, landmarks_px = recognizer.process_frame(frame)

    if landmarks_px:
        # Просим движок соединить суставы прямой
        recognizer.draw_landmarks(frame, landmarks_px)
        
        if stable_gesture:
            # Если жест тот же самый, что и в прошлый момент
            if stable_gesture == current_stable_gesture:
                # Считаем, сколько секунд мы его уже держим
                held_time = time.time() - gesture_start_time
                # Вычисляем прогресс от 0 до 100%
                progress = min(100, int((held_time / HOLD_TIME) * 100))
                
                # Выводим загрузку над запястьем (индекс 0)
                wrist_px = landmarks_px[0]
                text_pos = (wrist_px[0], wrist_px[1] - 50)
                cv2.putText(frame, f"Loading: {progress}%", text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                if held_time >= HOLD_TIME:
                    # Устанавливаем таймер на текущее время + время показа
                    showing_meme_until = time.time() + MEME_DISPLAY_TIME
                    active_meme_class = stable_gesture

                    # Очищаем память движка, чтобы не было ложных срабатываний после скрытия картинки
                    recognizer.gesture_buffer.clear()
                    current_stable_gesture = None
            else:
                # Если жест сменился, сбрасываем таймер, запоминаем новый жест
                current_stable_gesture = stable_gesture
                gesture_start_time = time.time()
        else:
            # Если вернулся None непонятный жест обнуляем
            current_stable_gesture = None
    else:
        # Если руки нет в кадре обнуляем
        current_stable_gesture = None

    # Показываем финальный кадр со всеми отрисовками
    cv2.imshow("MemeCam (AI Gestures)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрываем программу, освобождая камеру
cap.release()
cv2.destroyAllWindows()