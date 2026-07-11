import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import warnings
from collections import Counter
import os

# В дальнейшем передаем только строки с фичами без заголовков, игнорируем сообщения об ошибках
warnings.filterwarnings("ignore", message="X does not have valid feature names")

class GestureRecognizer:
    
    # Класс-движок
    # Скрывает в себе всю математику MediaPipe, загрузку модели и определение объекта

    def __init__(self, model_file=None, task_file=None):
        # Пути к моделям
        if model_file is None:
            model_file = "models/gesture_model.pkl"
        if task_file is None:
            task_file = "models/hand_landmarker.task"

        # Проверки наличия файлов
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Модель машинного обучения {model_file} не найдена!")
        if not os.path.exists(task_file):
            raise FileNotFoundError(f"Файл нейросети Google {task_file} не найден!")
            
        # Загрузка классификатора
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
            
        # Инициализация детектора рук
        base_options = python.BaseOptions(model_asset_path=task_file)
        
        # Настройка детектора рук
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1, # Одна рука
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Граф "скелета" для отрисовки
        self.mp_hands_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
            (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Буфер из 15 кадров с уверенностью в жесте на 75%, как только заполняется показывается картинка
        self.confidence_threshold = 0.75
        self.gesture_buffer = []
        self.buffer_size = 15 

    def process_frame(self, frame):

        # Принимает кадр с камеры. Возвращает номер стабильного жеста и координаты точек.
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Меняем цветовую палитру на RGB 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) 
        
        detection_result = self.detector.detect(mp_image)
        
        landmarks_px = []
        stable_gesture = None
        
        if detection_result.hand_landmarks: # Если рука была задетектированна
            for hand_landmarks in detection_result.hand_landmarks:
                # Переводим координаты из относительных в пиксели экрана
                for landmark in hand_landmarks:
                    landmarks_px.append((int(landmark.x * w), int(landmark.y * h)))
                
                # Подготовка данных для Случайного дерева, собраем координаты
                wrist_x = hand_landmarks[0].x
                wrist_y = hand_landmarks[0].y
                relative_coords = []
                for landmark in hand_landmarks:
                    relative_coords.append(landmark.x - wrist_x)
                    relative_coords.append(landmark.y - wrist_y)

                # Предсказание с проверкой на уверенность predict_proba
                # до этого использовался метод predict, небыло "пустого" жеста
                probabilities = self.model.predict_proba([relative_coords])[0]
                max_prob = np.max(probabilities)
                
                # Если текущий жест определился с вероятностью >= установленное значение
                if max_prob >= self.confidence_threshold: 
                    predicted_index = np.argmax(probabilities)
                    # Запоминаем класс жеста
                    prediction = str(self.model.classes_[predicted_index])
                else:
                    prediction = None 
                
                # Защита от тряски (буферизация последних 15 кадров)
                self.gesture_buffer.append(prediction)
                if len(self.gesture_buffer) > self.buffer_size:
                    self.gesture_buffer.pop(0)

                most_common_gesture, count = Counter(self.gesture_buffer).most_common(1)[0]
                
                # Если жест повторяется верно на 
                if count >= self.buffer_size * 0.7 and most_common_gesture is not None:
                    stable_gesture = most_common_gesture
        else:
            # Если рука пропала, очищаем память
            self.gesture_buffer.clear()
            
        return stable_gesture, landmarks_px

    def draw_landmarks(self, frame, landmarks_px):
        # Вспомогательная функция для отрисовки зеленого скелета кисти на кадре
        if not landmarks_px:
            return
        
        for connection in self.mp_hands_connections:
            pt1 = landmarks_px[connection[0]]
            pt2 = landmarks_px[connection[1]]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
        for pt in landmarks_px:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)