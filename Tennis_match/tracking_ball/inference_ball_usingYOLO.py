from ultralytics import YOLO
import pandas as pd


model = YOLO('models/best_yolov5.pt')


result = model.predict('/luidhy_docker/DELVEIMAGENS/other_things/tenis/input_videos/Shanghai2024Highlights.mp4', conf=0.25,save=True)

#list to store ball positions (frame number, x_min, y_min, x_max, y_max, confidence)
ball_positions = []

#interate over detected boxes for each frame
for i, res in enumerate(result):
    for box in res.boxes:
        #get bounding box coordinates and confidence
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
        confidence = box.conf.tolist()[0]
        
        
        ball_positions.append([i, x_min, y_min, x_max, y_max, confidence])

#save the positons as dataframe and save it as a csv file
df = pd.DataFrame(ball_positions, columns=['frame', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence'])
df.to_csv('ball_positions_1.csv', index=False)

print("Ball positions saved to ball_positions.csv")
