import os 
import random
import torch
import numpy as np
import PIL
import cv2
import time
import pyautogui
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import pygame
import dxcam


def getFrame(width=640,height=640):
    
    frame_array = camera.get_latest_frame()
    
    frame = PIL.Image.fromarray(frame_array)

    resized = frame.resize((width,height))
    resized = np.array(resized).astype(np.float32) # Converting to the expected float 32 input
    resized = np.expand_dims(resized.transpose(2,0,1),0) # Setting dimensions to (1,3,640,640)
    resized = resized/255 # Normalizing values 
    
    return frame_array,resized

original_res = np.array(pyautogui.screenshot()).shape

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model("model.onnx")
ort_sess = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

camera = dxcam.create(device_idx=0, output_idx=0)

image_width = 640
image_height = 640

window_width = 1280
window_height = 720

classes_dict = {0: 'GoldenCartridge',
           1: 'GoldenCookie',
           2: 'RedCartridge',
           3: 'RedCookie'}

classes_color_dict = {0: (100,255,0),
           1: (100,255,0),
           2: (255,0,220),
           3: (255,0,220)}

font = cv2.FONT_HERSHEY_SIMPLEX

camera.start(target_fps=40)

has_detected = False

while True:
    
    pygame.init()
    surface = pygame.display.set_mode((window_width,window_height))
    pygame.display.set_caption("Image")

    original,resized = getFrame(image_width, image_height)

    outputs = ort_sess.run(None, {'images': resized})
    
    img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (window_width,window_height))
        
    if len(outputs[0]) > 0:
        x1_all = []
        y1_all = []
        x2_all = []
        y2_all = []
        classes_all = []
        confidences = []

        for i in range(len(outputs[0])):
            x1_all.append(outputs[0][i][1] * window_width / image_width)
            y1_all.append(outputs[0][i][2] * window_height / image_height)
            x2_all.append(outputs[0][i][3] * window_width / image_width)
            y2_all.append(outputs[0][i][4] * window_height / image_height)
            classes_all.append(int(outputs[0][i][5]))
            confidences.append(round(outputs[0][i][6],2))



        for x1,y1,x2,y2,classes,confidence in zip(x1_all,y1_all,x2_all,y2_all,classes_all,confidences):
            cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), classes_color_dict[classes])
            cv2.putText(img,classes_dict[classes] + ": " + str(confidence),(int(x1),int(y1)-8), 
                        font, 0.6,classes_color_dict[classes],2,cv2.LINE_AA)
            
        
        x1 = original_res[1] * x1_all[0] / window_width
        x2 = original_res[1] * x2_all[0] / window_width
        y1 = original_res[0] * y1_all[0] / window_height
        y2 = original_res[0] * y2_all[0] / window_height

        pyautogui.moveTo((x1+x2)/2, (y1+y2)/2)
        has_detected = True
        
    elif has_detected:
        
        pyautogui.moveTo(385, 570) # Back to the main cookie
        has_detected = False

    displayImage = pygame.image.frombuffer(img.tobytes(), img.shape[1::-1],"BGR")
    surface.blit(displayImage,(0,0))
    pygame.display.update()

    

