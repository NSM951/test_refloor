from ultralytics import YOLO
import os
import cv2

def save_image_with_boxes(Results, image_path, output_dir="output_images"):
    """
    Сохраняет изображение с нарисованными bounding boxes
    
    Args:
        Results: результаты детекции от YOLO
        image_path: путь к исходному изображению
        output_dir: директория для сохранения результатов
    """
    # Создаем директорию для сохранения картинок
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем исходное изображение
    img = cv2.imread(image_path)
    
    # Получаем информацию о боксах
    boxes = Results.boxes.xyxy.cpu().numpy()
    
    # Рисуем bounding boxes на изображении
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # Рисуем прямоугольник
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Формируем имя выходного файла
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{name_without_ext}_with_boxes.jpg")
    
    # Сохраняем изображение
    cv2.imwrite(output_path, img)

def writing(Results, output_name, image_name):
    """
    Записывает результаты в формате JSON в указанный файл
    У объекта walls запись идет в виде двух точек:
    [[x1,y1],[x2,y2]]
    x1, y1 - координаты верхней левой вершины бокса (в пикселях)
    x2, y2 - координаты нижней правой вершины бокса (в пикселях)
    
    Args:
        Results: результаты детекции от YOLO
        output_name: название выходного файла
        image_name: название картинки
    """

    # открываем файл на запись
    with open( f"{output_name}", 'a') as file:
        
        # Шапка для результатов
        file.write(f'{{\n"meta":{{"source": "{image_name}"}},\n"walls:[\n')
        cnt = 1
        
        # Проходим по боксам из результатов
        for i in Results.boxes.xyxy:
            x1 = float(i[0])
            y1 = float(i[1])
            x2 = float(i[2])
            y2 = float(i[3])
            
            # случай когда дошли до последнего бокса
            if cnt == len(Results):
                file.write(f'{{"id":"w{cnt}","points":[[{x1},{y1}],[{x2},{y2}]]}}\n]\n')
            else:
                file.write(f'{{"id":"w{cnt}","points":[[{x1},{y1}],[{x2},{y2}]]}},\n')
            cnt += 1
            
        #закрываем скобки
        file.write(f'}}\n')
    

model = YOLO('baseline_model.pt')

print('Введите путь до файла или папку из которой надо обработать картинки')
path = input()

print('Введите название выходного файла для сохранения в формате JSON:')
out_name = input()

# Если на входе 1 изображение
if path[-4::] == '.jpg' or path[-4::] == '.png':
    img_name =  os.path.basename(path)
    results = model.predict(
        source = path,
        conf = 0.5
    )
    save_image_with_boxes(results[0], image_path = path)
    writing(Results = results[0], output_name = out_name, image_name = img_name)
# Если на входе дается папка с несколькими изображениями
else:
    for filename in os.listdir(path):
        path_to_img = path + '\\' + filename
        if  '.jpg' in filename or '.png' in  filename:
            results = model.predict(
                source = path_to_img,
                conf = 0.5
            )
            save_image_with_boxes(Results = results[0], image_path = path_to_img)
            writing(Results = results[0], output_name = out_name, image_name = filename)
