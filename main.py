import torch  # Imported as it adds CUDA to the PATH
import numpy as np
import cv2
import time
import onnxruntime as ort
import pygame
import dxcam
import threading
import sys
import os

from pynput.mouse import Button, Controller
from pynput.keyboard import Listener, KeyCode, Key

from functions import template_matching, get_model_providers

from execution_arguments import get_arguments

print("\n")
print("\n")
print("\n")

############################################################################################################


def getFrame(width=640, height=640):

    frame_array = camera.get_latest_frame()

    resized = cv2.resize(frame_array, (width, height), interpolation=cv2.INTER_AREA)
    resized = np.array(resized).astype(np.float32)  # Converting to the expected float 32 input
    resized = np.expand_dims(resized.transpose(2, 0, 1), 0)  # Setting dimensions to (1,3,640,640)
    resized /= 255  # Normalizing values

    return frame_array, resized


def clicker():
    while True:
        if clicking and centered and not isGardening:
            mouse.click(Button.left, 1)
        time.sleep(0.001)


def toggle_event(key):
    global big_cookie_coords
    global TOGGLE_KEY
    global first_toggle
    global clicking
    global centered
    global instaGarden

    if key == TOGGLE_KEY:

        if first_toggle:
            big_cookie_coords = template_matching(camera.get_latest_frame(), template="Big_cookie",
                                                  resolution=original_res[0], threshold=-1, RGB=True, verbose=False)
            first_toggle = False

        if not clicking:
            mouse.position = big_cookie_coords  # Back to the main cookie
            centered = True

        clicking = not clicking

        if clicking:
            print(bcolors.RED + "Auto-clicker enabled" + bcolors.ENDC + "\n")
        else:
            print(bcolors.RED + "Auto-clicker disabled" + bcolors.ENDC + "\n")

    if key == GARDEN_TOGGLE_KEY:
        instaGarden = True


def garden_process():

    global open_farm_coords
    global open_garden_coords
    global close_garden_coords
    global crop_remover_coords
    global target_seed_coords
    global speed_compost_coords
    global slow_compost_coords
    global holes_coords

    global isGardening
    global has_detected
    global gardening_crono
    global auto_garden_extra_time

    total_holes = len(os.listdir("Template_Matching_Imgs/" + str(
                      original_res[0]) + "p/Garden_holes"))  # Getting the number of holes by counting the number of holes images

    isGardening = True

    # Step 1: Silence all parcels
    silence_coords = ["first_iteration"]
    while len(silence_coords) > 0:
        silence_coords = template_matching(camera.get_latest_frame(), template="Silent_parcel",
                                           resolution=original_res[0], threshold=0.85, RGB=True, verbose=False)

        for coords in silence_coords:
            mouse.position = coords
            mouse.click(Button.left, 1)
            time.sleep(0.01)
        time.sleep(0.1)

    # Step 2: Open farm and garden

    if len(open_farm_coords) == 0:
        open_farm_coords = template_matching(camera.get_latest_frame(), template="Farm_icon",
                                             resolution=original_res[0], threshold=-1, RGB=True, verbose=False)

    mouse.position = open_farm_coords
    mouse.click(Button.left, 1)
    time.sleep(0.2)

    if len(close_garden_coords) == 0:
        close_garden_coords = template_matching(camera.get_latest_frame(), template="Close_garden",
                                                resolution=original_res[0], threshold=-1, RGB=True, verbose=False)

    mouse.position = close_garden_coords
    mouse.click(Button.left, 1)
    time.sleep(0.2)

    if len(open_garden_coords) == 0:
        open_garden_coords = template_matching(camera.get_latest_frame(), template="Open_garden",
                                               resolution=original_res[0], threshold=-1, RGB=True, verbose=False)

    mouse.position = open_garden_coords
    mouse.click(Button.left, 1)
    time.sleep(0.2)

    # Step 3: Check the garden availability

    growth_state_1 = template_matching(camera.get_latest_frame(), template="Target_plant1",
                                       resolution=original_res[0], threshold=0.9, RGB=True, verbose=False)

    growth_state_2 = template_matching(camera.get_latest_frame(), template="Target_plant2",
                                       resolution=original_res[0], threshold=0.9, RGB=True, verbose=False)

    growth_state_3 = template_matching(camera.get_latest_frame(), template="Target_plant3",
                                       resolution=original_res[0], threshold=0.9, RGB=True, verbose=False)

    growth_state_4 = template_matching(camera.get_latest_frame(), template="Target_plant4",
                                       resolution=original_res[0], threshold=0.9, RGB=True, verbose=False)

    planted_space = len(growth_state_1) + len(growth_state_2) + len(growth_state_3) + len(growth_state_4)

    print(f"Empty holes: {total_holes-planted_space}")
    print(f"Plants in growth state 1: {len(growth_state_1)}")
    print(f"Plants in growth state 2: {len(growth_state_2)}")
    print(f"Plants in growth state 3: {len(growth_state_3)}")
    print(f"Plants in growth state 4: {len(growth_state_4)}")

    free_space = (total_holes-planted_space)/total_holes  # Percentage of free space
    print(f"Free space: {round(free_space*100,2)}%")

    if free_space >= 0.4: # Threshold to start new replanting

        print(bcolors.GREEN + f"Free space greater than 40%, replanting started..." + bcolors.ENDC)

        if len(crop_remover_coords) == 0:
            time.sleep(0.2) # Ensuring it takes a good picture
            crop_remover_coords = template_matching(camera.get_latest_frame(), template="Crop_remover",
                                                    resolution=original_res[0], threshold=-1, RGB=True, verbose=False)
        if len(target_seed_coords) == 0:
            target_seed_coords = template_matching(camera.get_latest_frame(), template="Target_seed",
                                                   resolution=original_res[0], threshold=-1, RGB=True, verbose=False)
        if len(speed_compost_coords) == 0:
            speed_compost_coords = template_matching(camera.get_latest_frame(), template="Speed_compost",
                                                     resolution=original_res[0], threshold=-1, RGB=True, verbose=False)

        mouse.position = crop_remover_coords
        mouse.click(Button.left, 1)
        time.sleep(0.2)

        if len(holes_coords) == 0:
            time.sleep(0.2) # Ensuring it takes a good picture
            holes_coords = template_matching(camera.get_latest_frame(), template="Holes",
                                             resolution=original_res[0], threshold=-1, RGB=True, verbose=False)

        # Planting the seeds
        for hole_coord in holes_coords:

            mouse.position = target_seed_coords
            mouse.click(Button.left, 1)
            time.sleep(0.02)

            mouse.position = hole_coord
            mouse.click(Button.left, 1)
            time.sleep(0.02)

        mouse.position = speed_compost_coords
        mouse.click(Button.left, 1)
        time.sleep(0.1)

    else:

        percentage_of_growth = len(growth_state_4)/total_holes
        print(f"Growth plants: {round(percentage_of_growth*100,2)}%")

        if percentage_of_growth >= 0.4:

            print(bcolors.GREEN + f"Mature plants greater than 40%, changing compost to slow..." + bcolors.ENDC)

            if len(slow_compost_coords) == 0:
                slow_compost_coords = template_matching(camera.get_latest_frame(), template="Slow_compost",
                                                        resolution=original_res[0], threshold=-1, RGB=True, verbose=False)

            mouse.position = slow_compost_coords
            mouse.click(Button.left, 1)
            time.sleep(0.001)

            gardening_crono += auto_garden_extra_time # Adding extra time when while using the slow compost

        else:
            print(bcolors.GREEN + f"No actions required for garden" + bcolors.ENDC)

    # Closing garden
    silence_coords = template_matching(camera.get_latest_frame(), template="Silent_parcel",
                                       resolution=original_res[0], threshold=-1, RGB=True, verbose=False)

    mouse.position = silence_coords
    mouse.click(Button.left, 1)
    time.sleep(0.001)

    isGardening = False  # Process ended
    has_detected = True

    print(bcolors.GREEN + "Finished checking garden!" + bcolors.ENDC + "\n")
    return


def check_stuck_state():

    cross_coords = template_matching(camera.get_latest_frame(), template="Close_menu",
                                            resolution=original_res[0], threshold=0.9, RGB=True, verbose=False)

    if len(cross_coords) > 0:

        print(bcolors.RED + "STUCK STATE DETECTED, PROCEEDING TO CLOSE MENU WINDOW" + bcolors.ENDC + "\n")

        global clicking
        global has_detected

        clicking = False
        time.sleep(0.01)
        mouse.position = cross_coords[0]
        mouse.click(Button.left, 1)
        time.sleep(0.1)
        clicking = True
        has_detected = True

    found_parcels = False
    silence_coords = ["first_iteration"]
    while len(silence_coords) > 0:
        silence_coords = template_matching(camera.get_latest_frame(), template="Silent_parcel",
                                           resolution=original_res[0], threshold=0.85, RGB=True, verbose=False)

        for coords in silence_coords:
            clicking = False
            mouse.position = coords
            mouse.click(Button.left, 1)
            found_parcels = True
            time.sleep(0.01)
        time.sleep(0.1)

        if found_parcels: # In the case it has silenced some parcels, the click will be centered
            clicking = True
            has_detected = True

    return


############################################################################################################

# Color declaration and printing starting logo and banner
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
print(bcolors.MAGENTA + "Version 0.1.2" + bcolors.ENDC + "\n")

time.sleep(1) # Just to appreciate the logo and the banner xd


# Get arguments variables
my_args = get_arguments()

window_width = my_args.real_time_window_width
window_height = my_args.real_time_window_height

big_cookie_coords = ()

run_type = my_args.run_type.lower()
use_tiny_model = my_args.use_tiny_model
activate_rtw = not my_args.real_time_window  # Show real time window
activate_auto_aim = not my_args.auto_aim
activate_auto_garden = my_args.auto_garden
auto_garden_check = my_args.auto_garden_check  # How much time we need to check the garden when using the fast compost
auto_garden_extra_time = my_args.auto_garden_boost  # How much to add in seconds when using the slow compost

stuck_check_time = my_args.check_if_stuck_timer  # How much seconds need to pass in order to perform a stuck state check

print_every_minutes = my_args.check_run_time_every

# Select the toggle key that activates/deactivates autoclicking
selected_toggle_key = my_args.toggle_key.lower()
selected_garden_toggle_key= my_args.instagarden_toggle_key.lower()

# Mapping the strings to the proper function keys
function_keys = [("f1", Key.f1), ("f2", Key.f2),
                 ("f3", Key.f3), ("f4", Key.f4),
                 ("f5", Key.f5), ("f6", Key.f6),
                 ("f7", Key.f7), ("f8", Key.f8),
                 ("f9", Key.f9), ("f10", Key.f10),
                 ("f11", Key.f11), ("f12", Key.f12)]


TOGGLE_KEY = KeyCode(char=selected_toggle_key)
GARDEN_TOGGLE_KEY = KeyCode(char=selected_garden_toggle_key)

for i in function_keys:
    if selected_toggle_key == i[0]:
        TOGGLE_KEY = i[1]
    if selected_garden_toggle_key == i[0]:
        GARDEN_TOGGLE_KEY = i[1]

print(bcolors.CYAN + bcolors.BOLD + bcolors.UNDERLINE + "CONFIG" + bcolors.ENDC)
print(bcolors.CYAN + f"NN running in: {run_type}" + bcolors.ENDC)
print(bcolors.CYAN + f"Show pygame window: {activate_rtw}" + bcolors.ENDC)

if activate_rtw:
    print(bcolors.CYAN + f"Pygame window width: {window_width}" + bcolors.ENDC)
    print(bcolors.CYAN + f"Pygame window height: {window_height}" + bcolors.ENDC)

print(bcolors.CYAN + f"Autoclicker toggle key: {selected_toggle_key}" + bcolors.ENDC)
print(bcolors.CYAN + f"Auto aim: {activate_auto_aim}" + bcolors.ENDC)
print(bcolors.CYAN + f"Autogarden: {activate_auto_garden}" + bcolors.ENDC)
print(bcolors.CYAN + f"Start autogarden process key: {selected_garden_toggle_key}" + bcolors.ENDC)

if activate_auto_garden:
    print(bcolors.CYAN + f"Autogarden check every: {auto_garden_check}s" + bcolors.ENDC)
    print(bcolors.CYAN + f"Autogarden slow compost extra time: {auto_garden_extra_time}s" + bcolors.ENDC)

print("\n")


# Loading ONNX model

model_name, providers = get_model_providers(use_tiny_model, run_type)

ort.set_default_logger_severity(3)  # Avoiding the TensorRT warnings
ort_sess = ort.InferenceSession(model_name, providers=providers)

print(bcolors.CYAN + "Finished reading model" + bcolors.ENDC + "\n" + "\n")

# Variables to print bounding boxes

image_width = 640
image_height = 640

classes_dict = {0: 'GoldenCartridge',
                1: 'GoldenCookie',
                2: 'RedCartridge',
                3: 'RedCookie'}

classes_color_dict = {0: (100, 255, 0),  # Green for GoldenCartridge
                      1: (100, 255, 0),  # Green for GoldenCookie
                      2: (255, 0, 220),  # Magenta for RedCartridge
                      3: (255, 0, 220)}  # Magenta for RedCookie

font = cv2.FONT_HERSHEY_SIMPLEX # Font used on the bounding boxes


# Starting virtual camera

print(bcolors.CYAN + "Starting virtual camera..." + bcolors.ENDC)
camera = dxcam.create(device_idx=0, output_idx=0)

camera.start() # You can set target_fps=x
time.sleep(0.5) # Giving time to start the camera

original_res = np.array(camera.get_latest_frame()).shape

print(bcolors.CYAN + f"Detected resolution: {original_res[1]}x{original_res[0]}p" + bcolors.ENDC)

print(bcolors.CYAN + "Virtual camera ready" + bcolors.ENDC + "\n" + "\n")

# Toggle variables

has_detected = True
first_toggle = True # Get the big cookie coordinates on the first toggle
centered = False # Variable that tells if the click is centered on the big cookie or not, being not centered means not to spam clicks
clicking = False # Start the loop without auto clicking

# Creating clicker thread

mouse = Controller()  # Creating the mouse controller object
click_thread = threading.Thread(target=clicker, daemon=True)  # A Daemon thread kills it when the main thread ends
click_thread.start()


# Creating the pygame window

if activate_rtw:
    print(bcolors.CYAN + "Launching pygame window" + bcolors.ENDC + "\n" + "\n")
    pygame.init()
    surface = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Cookie Clicker Bot")
    pygame_font = pygame.font.Font('freesansbold.ttf', 32)


# Initialize garden variables

instaGarden = False  # If true, activates the autogarden function immediately
isGardening = False  # Stops auto clicker when gardening
gardening_crono = 0  # Starting the timer to perform the gardening (checks the garden on the first toggle)

# The first time the garden process occurs, the coords of the static objects are gathered with the template matching,
# the following times, these are already stored and there is no need to recalculate them.
open_farm_coords = []
open_garden_coords = []
close_garden_coords = []
crop_remover_coords = []
target_seed_coords = []
speed_compost_coords = []
slow_compost_coords = []
holes_coords = []

# Start the loop
FPS = []
FPS_window = 10  # Size of the window to calculate the average of FPS
total_run_time = time.time()
minute_check = 0
anti_stuck_timer = 0  # Starting timer to check whether you are stuck on the menu or options or not

print(bcolors.YELLOW + bcolors.BOLD + bcolors.UNDERLINE + "COOKIE BOT READY!" + bcolors.ENDC + "\n")
print(bcolors.YELLOW + f"Press {selected_toggle_key} to activate the autoclicking functions" + bcolors.ENDC)
if activate_auto_garden:
    print(bcolors.YELLOW + f"Press {selected_garden_toggle_key} to manually activate the autogarden" + bcolors.ENDC + "\n")

print(bcolors.RED + bcolors.BOLD + "WARNING: " + bcolors.ENDC +
      bcolors.RED + "Ensure you have the cookie clicker app running on your main screen when " + bcolors.ENDC)

print(bcolors.RED + "toggling the autoclick functions, otherwise undesired behaviours could happen" + bcolors.ENDC)

if activate_auto_garden:
    print("\n" + bcolors.RED + bcolors.BOLD + bcolors.UNDERLINE + "OPEN THE GARDEN BEFORE STARTING" + bcolors.ENDC)

with Listener(on_press=toggle_event) as listener:  # Starting the listener thread
    while True:
        if activate_rtw:
            pygame.display.update()

        if (activate_auto_garden and (time.time()-gardening_crono > auto_garden_check) and clicking) or instaGarden:
            instaGarden = False
            print("\n" + bcolors.GREEN + "Checking garden..." + bcolors.ENDC)
            check_stuck_state()  # Check if the game is stuck before checking the garden
            garden_thread = threading.Thread(target=garden_process, daemon=True)
            garden_thread.start()
            gardening_crono = time.time()

        if (time.time() - anti_stuck_timer > stuck_check_time) and clicking and not isGardening:
            check_stuck_thread = threading.Thread(target=check_stuck_state, daemon=True)
            check_stuck_thread.start()
            anti_stuck_timer = time.time()

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

            for x1, y1, x2, y2, classes, confidence in zip(x1_all, y1_all, x2_all, y2_all, classes_all, confidences):
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), classes_color_dict[classes])
                cv2.putText(img, classes_dict[classes] + ": " + str(confidence), (int(x1), int(y1)-8),
                            font, 0.6, classes_color_dict[classes], 2, cv2.LINE_AA)

                if len(outputs[0]) > 7:  # Cookie storm
                    x1 = original_res[1] * x1 / window_width
                    x2 = original_res[1] * x2 / window_width
                    y1 = original_res[0] * y1 / window_height
                    y2 = original_res[0] * y2 / window_height

                    if activate_auto_aim and clicking and not isGardening:
                        if centered:
                            centered = False
                            time.sleep(0.1)  # Just for security add extra time to properly deactivate clicks
                        time.sleep(0.01)
                        mouse.position = ((x1 + x2) / 2, (y1 + y2) / 2)
                        if clicking:
                            mouse.click(Button.left, 1)

            if len(outputs[0]) <= 7:  # Single cookie
                x1 = original_res[1] * x1_all[0] / window_width
                x2 = original_res[1] * x2_all[0] / window_width
                y1 = original_res[0] * y1_all[0] / window_height
                y2 = original_res[0] * y2_all[0] / window_height

                if activate_auto_aim and clicking and not isGardening:
                    centered = False
                    time.sleep(0.1)
                    mouse.position = ((x1+x2)/2, (y1+y2)/2)
                    if clicking:
                        mouse.click(Button.left, 1)

            has_detected = True

        if has_detected:
            if activate_auto_aim and clicking and not isGardening:
                mouse.position = big_cookie_coords
                time.sleep(0.1)  # Just for security add extra time to properly center the mouse
                centered = True
            has_detected = False

        if activate_rtw:
            # Display the screen
            displayImage = pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], "BGR")
            surface.blit(displayImage, (0, 0))
            # Creating FPS counter
            text = pygame_font.render("FPS: " + str(0), True, (0, 170, 0), (0, 0, 0))
            if len(FPS) >= FPS_window:
                text = pygame_font.render("FPS: " + str(int(np.mean(FPS))), True, (0, 170, 0), (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (window_width // 18, window_height // 1.02)
            surface.blit(text, textRect)
            # Creating text to show if autocliker is enabled or not
            if clicking:
                text = pygame_font.render("Auto-clicker: Active", True, (0, 170, 0), (0, 0, 0))
            else:
                text = pygame_font.render("Auto-clicker: Disabled", True, (0, 170, 0), (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (window_width // 2, window_height // 1.02)
            surface.blit(text, textRect)
            # Creating text to show the autogarden timer
            if activate_auto_garden:
                remaining_time_garden = auto_garden_check - (time.time() - gardening_crono)

                if not isGardening and remaining_time_garden > 0:

                    minutes = str(int(remaining_time_garden // 60))
                    seconds = str(int(remaining_time_garden % 60))

                    if int(seconds) < 10:
                        seconds = "0" + seconds
                    if int(minutes) < 10:
                        minutes = "0" + minutes

                    text = pygame_font.render(f"Garden check in: {minutes}:{seconds}", True, (0, 170, 0), (0, 0, 0))

                elif not clicking:
                    text = pygame_font.render(f"Waiting for gardening", True, (0, 170, 0), (0, 0, 0))
                else:
                    text = pygame_font.render(f"Gardening in course...", True, (0, 170, 0), (0, 0, 0))

                textRect = text.get_rect()
                textRect.center = (window_width // 1.2, window_height // 1.02)
                surface.blit(text, textRect)

            # Getting all the pygame events and shutting down script when the window is closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print(bcolors.CYAN + "Closing..." + bcolors.ENDC)
                    sys.exit(0)  # Ends the code

        # Total run tim calculations
        run_time = time.time() - total_run_time
        run_time_hours = int(run_time / 3600)
        run_time_minutes = int(run_time / 3600 % 1 * 60)
        run_time_seconds = int(run_time / 3600 % 1 * 60 % 1 * 60)

        if (run_time_minutes + run_time_hours * 60) % print_every_minutes == 0 and run_time_minutes != minute_check:

            minute_check = run_time_minutes

            if run_time_seconds < 10:
                run_time_seconds = "0" + str(run_time_seconds)
            if run_time_minutes < 10:
                run_time_minutes = "0" + str(run_time_minutes)
            if run_time_hours < 10:
                run_time_hours = "0" + str(run_time_hours)

            print(bcolors.RED + f"Current run time: {run_time_hours}h:{run_time_minutes}m:{run_time_seconds}s" + bcolors.ENDC + "\n")

        loop_time = time.time() - loop_time
        FPS.append(round(1/loop_time, 0))  # Calculates the frequency of each iteration
        if len(FPS) > FPS_window:
            FPS.pop(0) # Removing the oldest item from the list











