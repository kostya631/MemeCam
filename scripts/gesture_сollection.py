import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

# Модель распознающая "скелет" руки
MODEL_PATH = 'hand_landmarker.task'

# Создаем базовый объект модели
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# Параметры детектора (донастройки модели)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1, # одна рука
    # пороги уверенности
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# Итоговый объект детектора
detector = vision.HandLandmarker.create_from_options(options)

# Граф соединений точек "скелета"
mp_hands_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm connections
]

# Файл для сохранения координат и вида жеста
CSV_FILE = "gesture_dataset.csv"

# Если файл отсутствует, создаем его
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Заголовок таблицы: вид жеста, координаты 21 точки "скелета"
        header = ['class_id'] + [f'pt{i}_{axis}' for i in range(21) for axis in ['x', 'y']]
        writer.writerow(header)
        
print("Камера запущена! Для сохранения жеста нажать:")
print(" '1' - Жест для приветствия")
print(" '2' - Жест для лайка")
print(" '3' - Жест для не знаю")
print(" '4' - Жест для кулака")
print(" 'q' - Выход")

# Запуск камеры
cap = cv2.VideoCapture(1)

while True:
    # Считывание кадра
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    # Конвертация цветовой политры кадра в RGB 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Формат необходимый для работы MediaPipe 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Получаем распознавание
    detection_result = detector.detect(mp_image)

    # Рисуем интерфейс
    cv2.putText(frame, "Press 0, 1, 2, 3 or 4 to save", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Если в кадре найдена рука
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            
            # Ручная отрисовка "скелета"
            h, w, _ = frame.shape
            
            # Соединяем точки прямыми 
            for connection in mp_hands_connections:
                pt1 = hand_landmarks[connection[0]]
                pt2 = hand_landmarks[connection[1]]
                # Переводим нормализованные координаты в пиксели кадра
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # Рисуем "суставы"
            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            
            # Нормализация положения точек для сохранения и обучения
            # Координаты "запястья"
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y
            
            # Список для хранения нормализованных данных
            relative_coords = []

            # Перебираем точки "суставов" и находим координаты относительно запясьтья
            for landmark in hand_landmarks:
                relative_coords.append(landmark.x - wrist_x)
                relative_coords.append(landmark.y - wrist_y)

            # Ждем нажатия клавиши
            key = cv2.waitKey(1) & 0xFF
            
            # Если нажали от 0 до 4 - сохраняем вектор в CSV файл
            if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                class_id = chr(key) 
                
                # Добавляем в файл новую запись
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    # ID класса + координаты "суставов"
                    row = [class_id] + relative_coords
                    writer.writerow(row)
                    
                print(f"Сохранен вектор для мема {class_id}!")
                
                # Делаем экран белым на долю секунды
                frame[:] = 255 

    cv2.imshow("Gesture Data Collector", frame)

    # Выход по клавише q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершение работы
# Освобождаем камеру и закрываем все окна OpenCV
cap.release()
cv2.destroyAllWindows()