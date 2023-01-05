import torch
import numpy as np
from PIL import Image
import cv2
import time
import onnx
import onnxruntime as ort
import pygame
import dxcam
import threading
import sys


from pynput.mouse import Button, Controller
from pynput.keyboard import Listener, KeyCode, Key

from execution_arguments import get_arguments

print("\n")
class bcolors(object): # Class to print in colors on the console
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = '\033[33m'
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BLACK = "\033[90m"
    CYAN = "\033[96m"

cookie_logo = open("art/cookie_art.ans", "r")
cookie_banner = open("art/cookie_banner.txt", "r")
print(cookie_logo.read() + "\n")
print(bcolors.YELLOW + cookie_banner.read() + bcolors.ENDC)
print(bcolors.MAGENTA + "Version 0.0.2" + bcolors.ENDC + "\n")

# Get arguments variables
my_args = get_arguments()

window_width = my_args.real_time_viewer_width
window_height = my_args.real_time_viewer_height


activate_rtv = not my_args.real_time_viewer # Show real time viewer
clicking = not my_args.auto_clicker
activate_auto_aim = not my_args.auto_aim


# Select the toggle key that activates/deactivates autoclicking
selected_toggle_key = my_args.toggle_key.lower()

function_keys = [("f1", Key.f1), ("f2", Key.f2),
                 ("f3", Key.f3), ("f4", Key.f4),
                 ("f5", Key.f5), ("f6", Key.f6),
                 ("f7", Key.f7), ("f8", Key.f8),
                 ("f9", Key.f9), ("f10", Key.f10),
                 ("f11", Key.f11), ("f12", Key.f12)]


TOGGLE_KEY = KeyCode(char=selected_toggle_key)

for i in function_keys:
    if selected_toggle_key == i[0]:
        TOGGLE_KEY = i[1]

print(bcolors.CYAN + bcolors.BOLD + bcolors.UNDERLINE + "CONFIG" + bcolors.ENDC)
print(bcolors.CYAN + f"Show pygame window: {activate_rtv}" + bcolors.ENDC)
print(bcolors.CYAN + f"Pygame window width: {window_width}" + bcolors.ENDC)
print(bcolors.CYAN + f"Pygame window height: {window_height}" + bcolors.ENDC)
print(bcolors.CYAN + f"Start with autoclicker active: {clicking}" + bcolors.ENDC)
print(bcolors.CYAN + f"Autoclicker toggle key: {selected_toggle_key}" + bcolors.ENDC)
print(bcolors.CYAN + f"Auto aim: {activate_auto_aim}" + bcolors.ENDC)
print("\n")



def getFrame(width=640, height=640):

    frame_array = camera.get_latest_frame()
    frame = Image.fromarray(frame_array)

    resized = frame.resize((width, height))
    resized = np.array(resized).astype(np.float32) # Converting to the expected float 32 input
    resized = np.expand_dims(resized.transpose(2, 0, 1), 0) # Setting dimensions to (1,3,640,640)
    resized /= 255 # Normalizing values
    
    return frame_array,resized

def clicker():
    while True:
        if clicking and centered:
            mouse.click(Button.left, 1)
        time.sleep(0.001)


def toggle_event(key):
    if key == TOGGLE_KEY:
        global clicking

        if not clicking:
            mouse.position = (385, 570)  # Back to the main cookie

        clicking = not clicking




# Loading ONNX model

print(bcolors.CYAN + "Loading ONNX model..." + bcolors.ENDC)

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model("model.onnx")
ort_sess = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

print(bcolors.CYAN + "Finished reading model" + bcolors.ENDC)


print(bcolors.CYAN + "Starting virtual camera..." + bcolors.ENDC)
camera = dxcam.create(device_idx=0, output_idx=0)

image_width = 640
image_height = 640


classes_dict = {0: 'GoldenCartridge',
           1: 'GoldenCookie',
           2: 'RedCartridge',
           3: 'RedCookie'}

classes_color_dict = {0: (100,255,0),
           1: (100,255,0),
           2: (255,0,220),
           3: (255,0,220)}

font = cv2.FONT_HERSHEY_SIMPLEX

camera.start() # You can set target_fps=x
time.sleep(0.5) # Giving time to start the camera

original_res = np.array(camera.get_latest_frame()).shape

print(bcolors.CYAN + "Virtual camera ready" + bcolors.ENDC)

has_detected = True

# Creating clicker thread

centered = False # Variable that tells if the click is centered on the big cookie or not, being not centered means not to spam clicks

mouse = Controller()


click_thread = threading.Thread(target=clicker, daemon=True)  # A Daemon thread kills it when the main thread ends
click_thread.start()

if activate_rtv:
    print(bcolors.CYAN + "Launching pygame window" + bcolors.ENDC)
    pygame.init()
    surface = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Cookie Clicker Bot")
    pygame_font = pygame.font.Font('freesansbold.ttf', 32)


# Start the loop
FPS = 0.0
with Listener(on_press=toggle_event) as listener: # Starting the listener thread
    while True:

        if activate_rtv:
            pygame.display.update()

        loop_time = time.time()

        original, resized = getFrame(image_width, image_height)

        outputs = ort_sess.run(None, {'images': resized})

        img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (window_width, window_height))

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
                confidences.append(round(outputs[0][i][6], 2))


            for x1,y1,x2,y2,classes,confidence in zip(x1_all,y1_all,x2_all,y2_all,classes_all,confidences):
                cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), classes_color_dict[classes])
                cv2.putText(img,classes_dict[classes] + ": " + str(confidence),(int(x1),int(y1)-8),
                            font, 0.6,classes_color_dict[classes],2,cv2.LINE_AA)


            x1 = original_res[1] * x1_all[0] / window_width
            x2 = original_res[1] * x2_all[0] / window_width
            y1 = original_res[0] * y1_all[0] / window_height
            y2 = original_res[0] * y2_all[0] / window_height

            if activate_auto_aim and clicking:
                centered = False
                time.sleep(0.01)
                mouse.position = ((x1+x2)/2, (y1+y2)/2)
                if clicking:
                    mouse.click(Button.left, 1)

            has_detected = True

        if has_detected:
            if activate_auto_aim & clicking:
                mouse.position = (385, 570) # Back to the main cookie #TODO get main cookie coords
                time.sleep(0.01)
                centered = True
            has_detected = False

        if activate_rtv:
            displayImage = pygame.image.frombuffer(img.tobytes(), img.shape[1::-1],"BGR")
            surface.blit(displayImage, (0, 0))
            # create a text surface object, on which text is drawn on it.
            text = pygame_font.render("FPS: " + str(FPS), True, (0, 170, 0), (0, 0, 0))
            # create a rectangular object for the text surface object
            textRect = text.get_rect()
            # set the center of the rectangular object.
            textRect.center = (window_width // 18, window_height // 1.02)
            surface.blit(text, textRect)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print(bcolors.CYAN + "Closing app..." + bcolors.ENDC)
                    sys.exit(0) # Ends the code

        loop_time = time.time() - loop_time
        FPS = round(1/loop_time, 0) # Calculates the frequency of each iteration





    

