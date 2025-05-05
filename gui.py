# set this to load camera faster
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# load camera:
import cv2
import numpy as np

vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    running, frame = vc.read()
else:
    running = False

# exit if camera not working:
import sys
if not running:
    sys.exit()

print("camera loaded")

# load haar cascade for detecting face
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
haar_finds = []
i = 0

print("haar cascade loaded")

# load LBHF for recognizing face
recognizer = cv2.face.LBPHFaceRecognizer_create()
data_path = 'faces'

global faces, labels, names_dict, model_trained
model_trained = False

def train_model():
    global faces, labels, names_dict, model_trained

    faces = []
    labels = []

    name_index = 0
    names_dict = {}

    if os.path.exists("faces"):

        for person_name in os.listdir(data_path):
            person_dir = os.path.join(data_path, person_name)
            for filename in os.listdir(person_dir):
                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(name_index)
            names_dict[name_index] = person_name
            name_index+=1

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        #recognizer.save('trained_model.xml')
        model_trained = True

train_model()

print(len(faces))

print("LBHF loaded")


face_img_diameter = 100

global frame_toggle, last_frame
last_frame = None
frame_toggle = True

def get_camera(width, height):

    global frame_toggle, last_frame

    if frame_toggle:
        rval, frame = vc.read()
        frame_toggle = False
        last_frame = frame
    else:
        rval = True
        frame = last_frame
        frame_toggle = True

    pg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pg_frame = np.rot90(pg_frame)
    pg_frame = np.flip(pg_frame, axis=0)
    cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(pg_frame), (width,height))

    return rval, frame, cam_surf


global last_img

def recognize_faces(surface, frame, frame_size, skip_render = False):
    global last_img, model_trained

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_detection = facedetect.detectMultiScale(gray, 1.3, 3)

    x_offset, y_offset, width, height = frame_size

    xr = width/640
    yr = height/480

    detections = []

    for (x, y, w, h) in haar_detection:
        crop_img = frame[y:y+h, x:x+w, :]
        crop_gray = gray[y:y+h, x:x+w]

        resized_img = cv2.resize(crop_img, (face_img_diameter, face_img_diameter))

        last_img = resized_img

        if not skip_render:
            pygame.draw.rect(surface, (255,0,0), (x_offset+x*xr,y_offset+y*yr,w*xr,h*yr), 1, border_radius=1)


        label_txt = "???"#str(confidence)
        if model_trained:
            label, confidence = recognizer.predict(crop_gray)
            if confidence < 115:
                label_txt = names_dict[label]
                detections.append(label_txt)

        if not skip_render:
            text_surface = fonts["small"].render(label_txt, True, (255, 0, 0))
            surface.blit(text_surface, (x_offset+x*xr,y_offset+(y+h)*yr))

    return detections


# rendering:
import pygame

pygame.init()
pygame.font.init()

from globals import *
from gui_elements import *

def get_display_size():
    sizes = pygame.display.get_desktop_sizes()
    index = 0
    if len(sizes) > 1:
        print(sizes)
        index = int(input("Enter display index: "))
    return sizes[index]

WIDTH, HEIGHT = get_display_size()
display = pygame.display.set_mode((WIDTH, HEIGHT))

import pickle

def write_profile(profile):
    with open("user_profile.pickle", "wb") as file:
        pickle.dump(profile, file, pickle.HIGHEST_PROTOCOL)

def read_profile():
    if not os.path.exists("user_profile.pickle"):
        print("no profile, setting defaults")

        user_profile = {
            "pin": "123456",
            "font-size": 2,
            "alarm-volume": 0.5,
            "high-contrast": False,
            "policy-agreed": False,
        }

        write_profile(user_profile)

    with open("user_profile.pickle", 'rb') as f:
        return pickle.load(f)

user_profile = read_profile()
select_font(user_profile["font-size"])
print(user_profile) # TODO: remove

from enum import Enum


scroll_grid = ScrollGrid((50,500,400,300),3,(10,10))

slider = Slider((50,500,200,30), ["a","b","c","d"], snap=True)

HEADER_HEIGHT = HEIGHT * 0.15
SIDE_BAR_OFFSET = WIDTH * 0.7

header_padding = 20
back_button = Button((header_padding,header_padding,HEADER_HEIGHT-header_padding*2,HEADER_HEIGHT-header_padding*2), symbols["back"])
home_button = Button((WIDTH - HEADER_HEIGHT + header_padding,header_padding,HEADER_HEIGHT-header_padding*2,HEADER_HEIGHT-header_padding*2), symbols["home"])

global system_armed, system_locked

system_locked = True
system_armed = True # TODO: save this?

class Screen(Enum):
    LOGIN = 1
    HOME = 2
    SETTINGS = 3
    LIVE_CAMERA = 4
    SYSTEM_HISTORY = 5
    CHANGE_PIN = 6
    ADD_FACE = 7
    MANAGE_USERS = 8
    PRIVACY_POLICY = 9
    CAMERA_UNLOCK = 10



def system_activity():

    global system_armed

    centre_x = SIDE_BAR_OFFSET + (WIDTH - SIDE_BAR_OFFSET)/2

    pygame.draw.line(display, colours["foreground"], (SIDE_BAR_OFFSET, HEADER_HEIGHT), (SIDE_BAR_OFFSET, HEIGHT))
    text1 = fonts["medium"].render("The system", True, colours["text"])
    text2 = fonts["medium"].render("is currently:", True, colours["text"])

    text_y = HEIGHT*0.3

    display.blit(text1, (centre_x - text1.get_rect().width/2, text_y))
    display.blit(text2, (centre_x - text2.get_rect().width/2, text_y + text1.get_rect().height + TEXT_GAP_MEDIUM))

    armed_string = "Armed " + symbols["lock-closed"]
    if not system_armed:
        armed_string = "Disarmed " + symbols["lock-open"]

    text_y = HEIGHT*0.5
    armed_text = fonts["big"].render(armed_string, True, colours["text"])

    display.blit(armed_text, (centre_x - armed_text.get_rect().width/2, text_y))


pin_inp = TextField((0, 0), max_length=6, numbers_only=True, always_active=True, min_width = 200)
global invalid_pin, current_screen
invalid_pin = False

face_scan_button = Button((0,0,SIDE_BAR_OFFSET*0.3, SIDE_BAR_OFFSET*0.3 * (3.0/4.0)), symbols["camera"], font_size = 2)

def login_screen(events):

    global invalid_pin, current_screen, system_armed

    submitted_pin = ""
    for event in events:
        submit = pin_inp.process(event)
        if submit != "":
            submitted_pin = submit

    if submitted_pin != "":
        if submitted_pin == user_profile["pin"]:
            current_screen = Screen.HOME
            system_armed = False
            system_locked = False
            submitted_pin = ""
            invalid_pin = False
        else:
            invalid_pin = True

    system_activity()
    
    centre_x = SIDE_BAR_OFFSET / 2
    y = HEADER_HEIGHT + 40

    # authenticate to access
    authenticate_text = fonts["medium"].render("Authenticate to Access Control", True, colours["text"])
    display.blit(authenticate_text, (centre_x - authenticate_text.get_rect().width/2, y))

    y += authenticate_text.get_rect().height + 20

    # enter pin
    enter_pin_text = fonts["medium"].render(symbols["dot"] + " Enter PIN Code:  ", True, colours["text"])
    display.blit(enter_pin_text, (centre_x - enter_pin_text.get_rect().width/2, y))

    y += enter_pin_text.get_rect().height + 10

    # pin input
    pin_inp.set_pos((centre_x, y))
    pin_inp.render(display)

    y += pin_inp.height

    invalid_pin_text = fonts["medium"].render(symbols["error"]+" Incorrect pin", True, colours["error-text"])

    if invalid_pin:
        display.blit(invalid_pin_text, (centre_x - invalid_pin_text.get_rect().width/2, y))

    y += invalid_pin_text.get_rect().height + 20

    # or
    or_text = fonts["medium"].render("or", True, colours["text"])
    display.blit(or_text, (centre_x - or_text.get_rect().width/2, y))

    y += or_text.get_rect().height + 40

    # use face scan
    use_face_scan_text = fonts["medium"].render(symbols["dot"] + " Use Face Scan:  ", True, colours["text"])
    display.blit(use_face_scan_text, (centre_x - use_face_scan_text.get_rect().width/2, y))

    y += use_face_scan_text.get_rect().height + 20

    face_scan_button.set_pos((centre_x - face_scan_button.rect.width/2, y))
    clicked = face_scan_button.render(display)

    if clicked:
        current_screen = Screen.CAMERA_UNLOCK
        submitted_pin = ""
        invalid_pin = False

    y += face_scan_button.rect.height / 2


arm_button = Button((0,0,0,0), symbols["lock-closed"])
disarm_button = Button((0,0,0,0), symbols["lock-open"])
settings_button = Button((0,0,0,0), "System Settings " + symbols["cog"])

def home_screen(events):

    global system_armed, current_screen
    
    system_activity()

    centre_x = SIDE_BAR_OFFSET / 2
    inner_height = HEIGHT - HEADER_HEIGHT

    arm_box_width = SIDE_BAR_OFFSET * 0.7
    arm_box_height = inner_height * 0.35

    y = HEADER_HEIGHT + 60

    pygame.draw.rect(display, colours["foreground"], (centre_x - arm_box_width / 2, y, arm_box_width, arm_box_height), 2)

    box_left_centre_x = centre_x - arm_box_width / 4
    box_right_centre_x = centre_x + arm_box_width / 4
    padding = 30

    arm_text = fonts["medium"].render("Arm System", True, colours["text"])
    disarm_text = fonts["medium"].render("Disarm System", True, colours["text"])

    text_height = max(arm_text.get_rect().height, disarm_text.get_rect().height)

    left_rect = (box_left_centre_x - arm_box_width / 4 + padding, y + padding + text_height, arm_box_width/2 - 1.5*padding, arm_box_height - 2*padding - text_height)
    right_rect = (box_right_centre_x - arm_box_width / 4 + padding/2, y + padding + text_height, arm_box_width/2 - 1.5*padding, arm_box_height - 2*padding - text_height)

    display.blit(arm_text, (box_left_centre_x - arm_text.get_rect().width/2, y + padding/2))
    display.blit(disarm_text, (box_right_centre_x - disarm_text.get_rect().width/2, y + padding/2))

    arm_button.set_rect(left_rect)
    disarm_button.set_rect(right_rect)

    if arm_button.render(display):
        system_armed = True
        current_screen = Screen.LOGIN
    if disarm_button.render(display):
        system_armed = False

    y += arm_box_height + 60

    settings_width = arm_box_width / 2
    settings_height = arm_box_height / 2

    settings_button.set_rect((centre_x - settings_width/2, y, settings_width, settings_height))
    if settings_button.render(display):
        current_screen = Screen.SETTINGS

    maintainence_text = fonts["medium"].render(" Next annual maintainence check: 04/05/2026 ", True, colours["text"])

    width = maintainence_text.get_rect().width
    height = maintainence_text.get_rect().height
    x = centre_x - width / 2
    y = HEIGHT - height - 60

    display.blit(maintainence_text, (x, y))
    pygame.draw.rect(display, colours["foreground"], (x,y,width,height), 2)


live_camera_button = Button((0,0,0,0), "Live Camera Feed")
system_history_button = Button((0,0,0,0), "System History")
change_pin_button = Button((0,0,0,0), "Change PIN Code")
font_slider = Slider((0,0,WIDTH/4 - 90,0),["Small", "Medium", "Large"],True, offset=user_profile["font-size"])
alarm_slider = Slider((0,0,WIDTH/4 - 90,0),["Quiet", "Loud"], offset=user_profile["alarm-volume"])
high_contrast = CheckBox((0,0), checked=user_profile["high-contrast"])
add_face_button = Button((0,0,0,0), "Add New Face")
manage_users_button = Button((0,0,0,0), "Manage Stored Users")
privacy_button = Button((0,0,0,0), "Privacy Policy")
data_agree_button = Button((0,0,0,0))

global data_agree
data_agree = True

def settings_screen(events):

    global data_agree, current_screen

    pygame.draw.line(display, colours["foreground"], (WIDTH/2, HEADER_HEIGHT), (WIDTH/2, HEIGHT))

    half_height = (HEIGHT-HEADER_HEIGHT) / 2

    pygame.draw.line(display, colours["foreground"], (0,  HEADER_HEIGHT + half_height), (WIDTH/2, HEADER_HEIGHT + half_height))

    # top left section

    centre_x = WIDTH / 4
    y = HEADER_HEIGHT + 40

    title_text = fonts["medium"].render("Control Box Management", True, colours["text"])
    display.blit(title_text, (centre_x - title_text.get_rect().width/2, y))

    y += title_text.get_rect().height + 30

    button_height = title_text.get_rect().height + 10
    button_width = WIDTH*0.3

    live_camera_button.set_rect((centre_x - button_width/2, y, button_width, button_height))
    live_camera_button.render(display)

    y += button_height + 10

    system_history_button.set_rect((centre_x - button_width/2, y, button_width, button_height))
    system_history_button.render(display)

    y += button_height + 10

    change_pin_button.set_rect((centre_x - button_width/2, y, button_width, button_height))
    change_pin_button.render(display)



    # bottom left section

    y = HEADER_HEIGHT + half_height + 40

    title_text = fonts["medium"].render("Accessibility Settings", True, colours["text"])
    display.blit(title_text, (centre_x - title_text.get_rect().width/2, y))

    y += title_text.get_rect().height + 20

    font_size_text = fonts["medium"].render("Font Size:", True, colours["text"])
    display.blit(font_size_text, (WIDTH/8 - font_size_text.get_rect().width / 2, y))

    slider_height = 34
    slider_padding = 30
    font_slider.set_rect((centre_x + slider_padding, y + (font_size_text.get_rect().height - slider_height)/2, WIDTH/4 - 3*slider_padding, slider_height))
    
    slider_val = font_slider.render(display)
    if slider_val is not None:
        font_val = int(slider_val)
        user_profile["font-size"] = font_val
        select_font(font_val)
        write_profile(user_profile)

    y += font_size_text.get_rect().height + 40

    alarm_volume_text = fonts["medium"].render("Alarm Volume:", True, colours["text"])
    display.blit(alarm_volume_text, (WIDTH/8 - alarm_volume_text.get_rect().width / 2, y))

    alarm_slider.set_rect((centre_x + slider_padding, y + (alarm_volume_text.get_rect().height - slider_height)/2, WIDTH/4 - 3*slider_padding, slider_height))
    slider_val = alarm_slider.render(display)
    if slider_val is not None:
        user_profile["alarm-volume"] = slider_val
        write_profile(user_profile)

    y += alarm_volume_text.get_rect().height + 40

    high_contrast_text = fonts["medium"].render("High Contrast Text:", True, colours["text"])
    display.blit(high_contrast_text, (WIDTH/8 - high_contrast_text.get_rect().width / 2, y))

    high_contrast.set_pos((centre_x + slider_padding, y + high_contrast_text.get_rect().height / 2))
    # (render at the end)


    # right section

    y = HEADER_HEIGHT + 40
    centre_x = WIDTH * (3.0/4.0)

    title_text = fonts["medium"].render("Facial Data Management", True, colours["text"])
    display.blit(title_text, (centre_x - title_text.get_rect().width/2, y))

    y += title_text.get_rect().height + 50

    add_face_button.set_rect((centre_x-button_width/2,y,button_width,button_height))
    add_face_button.render(display)

    y += button_height + 40

    manage_users_button.set_rect((centre_x-button_width/2,y,button_width,button_height))
    manage_users_button.render(display)

    y += button_height + 40

    privacy_button.set_rect((centre_x-button_width/2,y,button_width,button_height))
    privacy_button.render(display)

    centre_y = HEADER_HEIGHT + half_height * 1.5

    agree1_text = fonts["medium"].render("I agree to the storage of", True, colours["text"])
    agree2_text = fonts["medium"].render("face data for authentication", True, colours["text"])

    text_height = agree1_text.get_rect().height + agree2_text.get_rect().height
    y = centre_y - text_height/2

    checkbox_padding = 80

    text_width = max(agree1_text.get_rect().width, agree2_text.get_rect().width)
    agree_width = text_width + 3*checkbox_padding

    x = centre_x - agree_width/2

    string = symbols["empty-box"]
    if user_profile["policy-agreed"]:
        string = symbols["checked-box"]

    check_pos = (x + checkbox_padding, centre_y)

    centre_x = x + checkbox_padding*2 + text_width/2

    top_y = y

    display.blit(agree1_text, (centre_x - agree1_text.get_rect().width/2, y))
    y += agree1_text.get_rect().height
    display.blit(agree2_text, (centre_x - agree2_text.get_rect().width/2, y))

    v_padding = 10
    data_agree_button.set_rect((x,top_y - v_padding,agree_width,text_height + 2*v_padding))
    data_agree_button.render(display)

    check_text = fonts["big"].render(string, True, colours["text"])
    display.blit(check_text, (check_pos[0] - check_text.get_rect().width/2, check_pos[1] - check_text.get_rect().height/2))


    # processing:

    for event in events:
        checked = high_contrast.process(event)
        if checked is not None:
            user_profile["high-contrast"] = checked
            write_profile(user_profile)

        if data_agree_button.process(event):
            user_profile["policy-agreed"] = not user_profile["policy-agreed"]
            write_profile(user_profile)

        if live_camera_button.process(event):
            current_screen = Screen.LIVE_CAMERA

        if system_history_button.process(event):
            current_screen = Screen.SYSTEM_HISTORY

        if change_pin_button.process(event):
            current_screen = Screen.CHANGE_PIN
            old_pin_field.text = ""
            new_pin_field.text = ""
            new_pin2_field.text = ""

        if add_face_button.process(event):
            current_screen = Screen.ADD_FACE

        if manage_users_button.process(event):
            current_screen = Screen.MANAGE_USERS

        if privacy_button.process(event):
            current_screen = Screen.PRIVACY_POLICY

    high_contrast.render(display)


def multi_line_text(words, rect, font, h_pad = 10, comma=False):
    lines = [[]]
    width = 0
    num_words = len(words)

    start_x = rect[0]
    end_x = rect[0] + rect[2]

    y = rect[1]

    x = start_x + h_pad
    line_index = 0
    for i,word in enumerate(words):

        add_string = " "
        if comma:
            add_string = ", "

        if i+1 >= num_words:
            add_string = ""

        new_line = False
        if word[0] == '\n':
            new_line = True
            word = word[1:]
        text = font.render(word+add_string,True,colours["text"])
        x += text.get_rect().width

        if x > end_x or new_line: 
            line_index += 1
            lines.append([])
            x = start_x + h_pad + text.get_rect().width

        lines[line_index].append(text)

    height = 0

    for line in lines:
        x = start_x + h_pad
        for text in line:
            rect = text.get_rect()
            height = max(height, rect.height)
            display.blit(text, (x,y))
            x += rect.width
        y += height
        height = 0

def live_camera_screen(events):

    global current_screen, system_locked

    PADDING = 40
    camera_area_height = HEIGHT-(HEADER_HEIGHT+PADDING)-PADDING
    camera_area_width = camera_area_height * (4.0/3.0)

    camera_area_x = WIDTH-camera_area_width-PADDING/2
    camera_area_y = HEADER_HEIGHT+PADDING

    camera_area_rect = (camera_area_x, camera_area_y, camera_area_width, camera_area_height)

    rval, frame, cam_surf = get_camera(camera_area_width, camera_area_height)

    display.blit(cam_surf, (camera_area_x, camera_area_y))
    pygame.draw.rect(display, colours["foreground"], camera_area_rect, 2)

    detections = recognize_faces(display, frame, camera_area_rect)

    centre_x = camera_area_x / 2

    title_text = fonts["big"].render("Live Camera Feed", True, colours["text"])

    y = camera_area_y + 20
    display.blit(title_text, (centre_x - title_text.get_rect().width/2, y))

    y += title_text.get_rect().height + HEIGHT*0.15

    status_string = "Active" # TODO: what is this?
    status_text = fonts["medium"].render("Status: " + status_string, True, colours["text"])
    display.blit(status_text, (centre_x - status_text.get_rect().width/2, y))

    y += status_text.get_rect().height + HEIGHT*0.15

    detected_text = fonts["medium"].render("Detected Users:", True, colours["text"])
    display.blit(detected_text, (centre_x - detected_text.get_rect().width/2, y))

    y += detected_text.get_rect().height + 20

    start_x = centre_x - WIDTH*0.3 / 2
    pygame.draw.rect(display, colours["foreground"], (start_x, y, WIDTH*0.3, HEIGHT*0.25), 2)
    
    multi_line_text(detections, (start_x, y, WIDTH*0.3, HEIGHT*0.25), fonts["small"], comma=True)


def system_history_screen(events):

    y = HEADER_HEIGHT + 30
    centre_x = WIDTH / 2

    title_text = fonts["big"].render("System History", True, colours["text"])
    display.blit(title_text, (centre_x - title_text.get_rect().width/2, y))

    # TODO: scrolling table of history


old_pin_field = TextField((0,0), max_length=6, min_width=WIDTH*0.2, numbers_only=True, confirmable=False)
new_pin_field = TextField((0,0), max_length=6, min_width=WIDTH*0.2, numbers_only=True, confirmable=False)
new_pin2_field = TextField((0,0), max_length=6, min_width=WIDTH*0.2, numbers_only=True, confirmable=False)
confirm_pin_button = Button((0,0,WIDTH*0.2,HEIGHT*0.1), "Confirm PIN")

global invalid_pin2, non_matching_pins, invalid_pin_length
# 0 - not submitted
# < 0 - error
# > 0 - success
invalid_pin2 = 0
non_matching_pins = 0
invalid_pin_length = 0

def change_pin_screen(events):

    global invalid_pin2, non_matching_pins, invalid_pin_length

    y = HEADER_HEIGHT + 40
    centre_x = WIDTH / 2

    title_text = fonts["big"].render("Change PIN Code", True, colours["text"])
    display.blit(title_text, (centre_x - title_text.get_rect().width/2, y))

    y += title_text.get_rect().height + 40

    old_pin_text = fonts["medium"].render("Old Pin", True, colours["text"])
    display.blit(old_pin_text, (centre_x - old_pin_field.min_width/2 - 30 - old_pin_text.get_rect().width,y))

    if invalid_pin2 < 0:
        invalid_text = fonts["medium"].render(symbols["error"]+" Incorrect pin", True, colours["error-text"])
        display.blit(invalid_text, (centre_x + old_pin_field.min_width/2 + 30, y))
    if invalid_pin2 > 0:
        valid_text = fonts["medium"].render(symbols["check"], True, colours["success-text"])
        display.blit(valid_text, (centre_x + old_pin_field.min_width/2 + 30, y))

    old_pin_field.set_pos((centre_x, y))
    y += old_pin_field.height + 40


    new_pin_text = fonts["medium"].render("New Pin", True, colours["text"])
    display.blit(new_pin_text, (centre_x - new_pin_field.min_width/2 - 30 - new_pin_text.get_rect().width,y))

    if invalid_pin_length < 0:
        invalid_text = fonts["medium"].render(symbols["error"]+" Invalid length", True, colours["error-text"])
        display.blit(invalid_text, (centre_x + new_pin_field.min_width/2 + 30, y))
    if invalid_pin_length > 0:
        valid_text = fonts["medium"].render(symbols["check"], True, colours["success-text"])
        display.blit(valid_text, (centre_x + new_pin_field.min_width/2 + 30, y))

    new_pin_field.set_pos((centre_x, y))
    y += new_pin_field.height + 40


    new_pin2_text = fonts["medium"].render("Confirm New Pin", True, colours["text"])
    display.blit(new_pin2_text, (centre_x - new_pin2_field.min_width/2 - 30 - new_pin2_text.get_rect().width,y))

    if invalid_pin_length > 0:
        if non_matching_pins < 0:
            invalid_text = fonts["medium"].render(symbols["error"]+" PINs don't match", True, colours["error-text"])
            display.blit(invalid_text, (centre_x + new_pin2_field.min_width/2 + 30, y))
        if non_matching_pins > 0:
            valid_text = fonts["medium"].render(symbols["check"], True, colours["success-text"])
            display.blit(valid_text, (centre_x + new_pin2_field.min_width/2 + 30, y))



    new_pin2_field.set_pos((centre_x, y))
    y += new_pin2_field.height + 40

    confirm_pin_button.set_pos((centre_x - WIDTH*0.1, y))
    y += HEIGHT*0.1 + 60

    info_text = fonts["medium"].render("PIN must be 6 digits (numbers), e.g. 123456", True, colours["text"])
    display.blit(info_text, (centre_x-info_text.get_rect().width/2, y))


    for event in events:
        old_pin_field.process(event)
        new_pin_field.process(event)
        new_pin2_field.process(event)

        if confirm_pin_button.process(event):
            old = old_pin_field.text
            new = new_pin_field.text
            new2 = new_pin2_field.text

            invalid_pin2 = 1
            invalid_pin_length = 1
            non_matching_pins = 1

            invalid = False

            if old != user_profile["pin"]:
                invalid_pin2 = -1
                invalid = True
            if len(new) != 6:
                invalid_pin_length = -1
                invalid = True
            if new != new2:
                non_matching_pins = -1
                invalid = True

            if not invalid:
                user_profile["pin"] = new
                write_profile(user_profile)

    old_pin_field.render(display)
    new_pin_field.render(display)
    new_pin2_field.render(display)

    confirm_pin_button.render(display)

name_field = TextField((0,0), centred=(False,False), always_active=True, confirmable = False)
begin_button = Button((0,0,WIDTH*0.2,HEIGHT*0.1), "Begin")
next_button = Button((0,0,WIDTH*0.2,HEIGHT*0.1), "Next")
confirm_button = Button((0,0,WIDTH*0.15,HEIGHT*0.1), "Confirm")
deny_button = Button((0,0,WIDTH*0.15,HEIGHT*0.1), "Deny")

global invalid_name, add_face_phase, add_images, add_images_raw
invalid_name = False
add_face_phase = 0
add_images = []
add_images_raw = []

def add_face_screen(events):

    global current_screen, invalid_name, add_face_phase, last_img, add_images, add_images_raw

    PADDING = 40
    camera_area_height = HEIGHT-(HEADER_HEIGHT+PADDING)-PADDING
    camera_area_width = camera_area_height * (4.0/3.0)

    camera_area_x = WIDTH-camera_area_width-PADDING/2
    camera_area_y = HEADER_HEIGHT+PADDING

    camera_area_rect = (camera_area_x, camera_area_y, camera_area_width, camera_area_height)

    rval, frame, cam_surf = get_camera(camera_area_width, camera_area_height)

    display.blit(cam_surf, (camera_area_x, camera_area_y))
    pygame.draw.rect(display, colours["foreground"], camera_area_rect, 2)

    detections = recognize_faces(display, frame, camera_area_rect)

    centre_x = camera_area_x / 2

    title_text = fonts["big"].render("Add New User", True, colours["text"])

    y = camera_area_y + 20
    display.blit(title_text, (centre_x - title_text.get_rect().width/2, y))

    y += title_text.get_rect().height + HEIGHT*0.1

    name_text = fonts["medium"].render("Name", True, colours["text"])
    display.blit(name_text, (centre_x - name_text.get_rect().width - 20, y))

    name_field.set_pos((centre_x + 20, y))

    y += max(name_field.height, name_text.get_rect().height) + 50

    instruct1_text = fonts["medium"].render("Instructions: keep a neutral", True, colours["text"])
    instruct2_text = fonts["medium"].render("expression & remove any glasses.", True, colours["text"])
    instruct3_text = fonts["medium"].render("Ensure only one person is in frame", True, colours["text"])

    display.blit(instruct1_text, (centre_x - instruct1_text.get_rect().width/2, y))
    y += instruct1_text.get_rect().height

    display.blit(instruct2_text, (centre_x - instruct2_text.get_rect().width/2, y))
    y += instruct2_text.get_rect().height

    display.blit(instruct3_text, (centre_x - instruct3_text.get_rect().width/2, y))
    y += instruct3_text.get_rect().height + 60

    look_string = " "
    if add_face_phase == 1:
        look_string = "Look left"
    elif add_face_phase == 2:
        look_string = "Look forward"
    elif add_face_phase == 3:
        look_string = "Look right"

    look1_text = fonts["medium"].render(look_string, True, colours["text"])
    display.blit(look1_text, (centre_x - look1_text.get_rect().width/2, y))

    if 0 < add_face_phase < 4:
        look2_text = fonts["medium"].render(" "+look_string+" ", True, colours["text"], colours["background"])
        display.blit(look2_text, (camera_area_x + (camera_area_width - look2_text.get_rect().width)/2, camera_area_y + 20))

    y += look1_text.get_rect().height + 20

    begin_button.set_pos((centre_x - WIDTH*0.1, y))
    next_button.set_pos((centre_x - WIDTH*0.1, y))

    y += HEIGHT*0.1 + 20

    if invalid_name:
        invalid_text = fonts["medium"].render(symbols["error"]+" Enter a name first", True, colours["error-text"])
        display.blit(invalid_text, (centre_x - invalid_text.get_rect().width/2, y))

    inner_height = HEIGHT - HEADER_HEIGHT
    popup_size = (WIDTH*0.4, HEIGHT*0.4)
    popup_pos = (WIDTH/2 - popup_size[0]/2, HEADER_HEIGHT+inner_height/2-popup_size[1]/2)

    centre_x = popup_pos[0] + popup_size[0]/2

    confirm_button.set_pos((centre_x - WIDTH*0.15 - 10, popup_pos[1]+popup_size[1] - popup_size[1]*0.3, popup_size[0]/2, popup_size[1]/2))
    deny_button.set_pos((centre_x + 10, popup_pos[1]+popup_size[1] - popup_size[1]*0.3, popup_size[0]/2, popup_size[1]/2))

    for event in events:
        name_field.process(event)

        if add_face_phase == 0:

            if begin_button.process(event):
                if len(name_field.text) < 1:
                    invalid_name = True
                else:
                    invalid_name = False
                    add_face_phase = 1
                    add_images = []
                    add_images_raw = []

        elif add_face_phase < 4:
            if begin_button.process(event):

                add_images_raw.append(last_img)

                pg_last_img = cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB)
                pg_last_img = np.rot90(pg_last_img)
                pg_last_img = np.flip(pg_last_img, axis=0)

                add_images.append(pygame.surfarray.make_surface(pg_last_img))

                add_face_phase += 1

        elif add_face_phase == 4:
            if confirm_button.process(event):
                add_face_phase = 0

                os.makedirs(os.path.join("faces", name_field.text),exist_ok=True)

                count = len(os.listdir("faces/"+name_field.text+"/"))
                index = (count + 2) // 3

                cv2.imwrite("faces/"+name_field.text+"/"+str(index)+"_left_img.png",add_images_raw[0])
                cv2.imwrite("faces/"+name_field.text+"/"+str(index)+"_forward_img.png",add_images_raw[1])
                cv2.imwrite("faces/"+name_field.text+"/"+str(index)+"_right_img.png",add_images_raw[2])

                name_field.text = ""

                train_model()

            if deny_button.process(event):
                add_face_phase = 0


    name_field.render(display)
    if add_face_phase == 0:
        begin_button.render(display)

    elif add_face_phase < 4:
        next_button.render(display)

    elif add_face_phase == 4:

        pygame.draw.rect(display, colours["background"],(popup_pos[0],popup_pos[1],popup_size[0],popup_size[1]))
        pygame.draw.rect(display, colours["foreground"],(popup_pos[0],popup_pos[1],popup_size[0],popup_size[1]), 2)

        confirm_button.render(display)
        deny_button.render(display)

        img_diameter = popup_size[1]/2
        width = img_diameter * 3 + 20
        x = centre_x - width/2
        y = popup_pos[1] + 40

        for i in range(3):
            image = add_images[i]
            display.blit(pygame.transform.scale(image, (img_diameter, img_diameter)), (x,y))

            x += img_diameter + 10


def manage_users_screen(events):
    pass

privacy_policy = """This system uses facial recognition to enhance your property's security. In accordance with the UK General Data Protection Regulation (UK GDPR) and the Data Protection Act 2018, we collect and process facial images of trusted users to authenticate access. This data is processed with your explicit consent and stored securely within the system. 
 
Your facial data is encrypted and never shared with third parties. It is retained for a maximum of 30 days unless linked to a security incident, in which case it may be held slightly longer for investigative purposes. You can withdraw consent or delete your facial data at any time using the control box interface. 
 
By proceeding, you agree to the collection and use of your facial recognition data solely for security purposes. You have the right to access, correct, or delete your data, and to withdraw consent at any time."""


def privacy_policy_screen(events):

    centre_x = WIDTH/2
    y = HEADER_HEIGHT + 30

    title_text = fonts["big"].render("Privacy Policy", True, colours["text"])
    display.blit(title_text, (centre_x - title_text.get_rect().width/2, y))

    y += title_text.get_rect().height + 20
    rect = (centre_x - WIDTH*0.4, y, WIDTH*0.8, HEIGHT-y - 40)
    pygame.draw.rect(display, colours["foreground"], rect, 2)

    multi_line_text(privacy_policy.split(' '), rect, fonts["small"])


def camera_unlock(events):

    global current_screen, system_locked

    PADDING = 40
    camera_area_height = HEIGHT-(HEADER_HEIGHT+PADDING)-PADDING
    camera_area_width = camera_area_height * (4.0/3.0)

    camera_area_x = WIDTH-camera_area_width-PADDING/2
    camera_area_y = HEADER_HEIGHT+PADDING

    camera_area_rect = (camera_area_x, camera_area_y, camera_area_width, camera_area_height)

    rval, frame, cam_surf = get_camera(camera_area_width, camera_area_height)

    display.blit(cam_surf, (camera_area_x, camera_area_y))
    pygame.draw.rect(display, colours["foreground"], camera_area_rect, 2)

    detections = recognize_faces(display, frame, camera_area_rect)

    if len(detections) > 0:
        system_locked = False
        current_screen = Screen.HOME
        display.fill(colours["background"])
        return

    centre_x = camera_area_x / 2

    title_text = fonts["big"].render("Facial Authentication", True, colours["text"])
    display.blit(title_text, (centre_x - title_text.get_rect().width/2, camera_area_y + 20))


    instruct1_text = fonts["medium"].render("Please stand still", True, colours["text"])
    instruct2_text = fonts["medium"].render("and wait infront", True, colours["text"])
    instruct3_text = fonts["medium"].render("of the camera", True, colours["text"])

    instruct_height = instruct1_text.get_rect().height + instruct2_text.get_rect().height + instruct3_text.get_rect().height

    y = HEADER_HEIGHT + (HEIGHT-HEADER_HEIGHT)/2 - instruct_height / 2
    
    display.blit(instruct1_text, (centre_x - instruct1_text.get_rect().width/2, y))

    y += instruct1_text.get_rect().height

    display.blit(instruct2_text, (centre_x - instruct2_text.get_rect().width/2, y))

    y += instruct2_text.get_rect().height

    display.blit(instruct3_text, (centre_x - instruct3_text.get_rect().width/2, y))


screen_functions = {
    Screen.LOGIN: login_screen,
    Screen.HOME: home_screen,
    Screen.SETTINGS: settings_screen,
    Screen.LIVE_CAMERA: live_camera_screen,
    Screen.SYSTEM_HISTORY: system_history_screen,
    Screen.CHANGE_PIN: change_pin_screen,
    Screen.ADD_FACE: add_face_screen,
    Screen.MANAGE_USERS: manage_users_screen,
    Screen.PRIVACY_POLICY: privacy_policy_screen,
    Screen.CAMERA_UNLOCK: camera_unlock
}


current_screen = Screen.PRIVACY_POLICY

def back_button_dest():
    if current_screen.value < 4:
        return None
    elif current_screen.value < 10:
        return Screen.SETTINGS
    else:
        return Screen.LOGIN

def home_button_dest():
    s = current_screen.value
    if s < 3 or s == 10:
        return None
    else:
        return Screen.HOME


running = True
while running:
    events = pygame.event.get()

    home_dest = home_button_dest()
    back_dest = back_button_dest()


    header_title = fonts["big"].render("Security System Control", True, colours["text"])
    header_subtitle = fonts["medium"].render("Manage the features of your security system", True, colours["text"])

    for event in events:
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

        if back_dest is not None:
            if back_button.process(event):
                current_screen = back_dest

        if home_dest is not None:
            if home_button.process(event):
                current_screen = home_dest


    display.fill(colours["background"])

    # header
    display.blit(header_title, ((WIDTH-header_title.get_rect().width)/2, 10))
    display.blit(header_subtitle, ((WIDTH-header_subtitle.get_rect().width)/2, 10 + header_title.get_rect().height))

    if back_dest is not None:
        back_button.render(display)
    if home_dest is not None:
        home_button.render(display)

    pygame.draw.line(display, colours["foreground"], (0, HEADER_HEIGHT), (WIDTH, HEADER_HEIGHT))

    screen_functions[current_screen](events)

    pygame.display.update()
