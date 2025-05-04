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

faces = []
labels = []

name_index = 0

names_dict = {}

for person_name in os.listdir(data_path):
    person_dir = os.path.join(data_path, person_name)
    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(name_index)
    names_dict[name_index] = person_name
    name_index+=1

recognizer.train(faces, np.array(labels))
recognizer.save('trained_model.xml')


print(len(faces))

print("LBHF loaded")

# load window:
import pygame
pygame.init()

WIDTH = 640
HEIGHT = 480
ADD_WIDTH = HEIGHT // 3

display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.font.init()
font_20 = pygame.font.SysFont('Comic Sans MS', 20)
font_30 = pygame.font.SysFont('Comic Sans MS', 30)

reading_mode_text = font_30.render("Reading", True, (0, 0, 0), (255,255,255))
writing_mode_text = font_30.render("Writing", True, (0, 0, 0), (255,255,255))
no_faces_text = font_20.render("(no faces in database)", True, (0, 0, 0), (255,255,255))
enter_name_text = font_20.render("Enter name:", True, (0, 0, 0), (255,255,255))
confirm_prompt_text = font_30.render("Is this okay?", True, (0, 0, 0), (255,255,255))
yes_text = font_20.render("Yes", True, (0, 0, 0), (255,255,255))
no_text = font_20.render("No", True, (0, 0, 0), (255,255,255))

reading_mode = True
writing_index = 0

if (len(faces) == 0):
    reading_mode = False

show_grayscale = False

last_img = None
left_img = None
forward_img = None
right_img = None
pg_last_img = None
pg_left_img = None
pg_forward_img = None
pg_right_img = None

name_inp = ""

hovering_yes = False
hovering_no = False

face_img_diameter = 80

# main loop:
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONUP:
            if writing_index == 5:
                if hovering_yes:
                    writing_index = 0
                    os.makedirs(os.path.join("faces", name_inp),exist_ok=True)
                    cv2.imwrite("faces/"+name_inp+"/left_img1.png",left_img)
                    cv2.imwrite("faces/"+name_inp+"/forward_img1.png",forward_img)
                    cv2.imwrite("faces/"+name_inp+"/right_img1.png",right_img)
                    print("faces/"+name_inp)
                    print(left_img)
                    print(forward_img)
                    print(right_img)
                    
                elif hovering_no:
                    writing_index = 0
        
        if event.type == pygame.KEYDOWN:

            if writing_index == 4:
                if event.key == pygame.K_BACKSPACE:
                    if len(name_inp) == 1:
                        name_inp = ""
                    else:
                        name_inp = name_inp[:-1]
                    
                else:
                    if event.unicode > 'A' and event.unicode < 'z': # TODO: maybe implement accents for names
                        name_inp += event.unicode
                                    
            #if event.key == pygame.K_BACKSPACE:
            #    show_grayscale = not show_grayscale
            if event.key == pygame.K_RETURN:

                if writing_index == 1:
                    left_img = last_img
                    pg_left_img = pg_last_img
                elif writing_index == 2:
                    forward_img = last_img
                    pg_forward_img = pg_last_img
                elif writing_index == 3:
                    right_img = last_img
                    pg_right_img = pg_last_img

                elif writing_index == 0:
                    left_img = None
                    forward_img = None
                    right_img = None
                    name_inp = ""

                if writing_index != 4 or len(name_inp)>0:
                    print("'"+name_inp+"'", len(name_inp))
                    writing_index += 1
                
                if writing_index > 5:
                    writing_index = 0
                    os.makedirs(os.path.dirname("faces/"+name_inp),exist_ok=True)
                    cv2.imwrite("faces/"+name_inp+"/left_img1.png",left_img)
                    cv2.imwrite("faces/"+name_inp+"/forward_img1.png",forward_img)
                    cv2.imwrite("faces/"+name_inp+"/right_img1.png",right_img)
                    print("faces/"+name_inp)
                    
            if event.key == pygame.K_TAB:
                if len(faces) == 0:
                    reading_mode = False
                else:
                    reading_mode = not reading_mode

    rval, frame = vc.read()

    # exit early if camera stopped
    if not rval:
        break

    # convert to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_detection = facedetect.detectMultiScale(gray, 1.3, 3)

    # show camera feed
    if show_grayscale:
        pg_frame = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    else:
        pg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pg_frame = np.rot90(pg_frame)
    pg_frame = np.flip(pg_frame, axis=0)
    surf = pygame.surfarray.make_surface(pg_frame)
    
    display.blit(surf, (0, 0))

    # render face images
    """
    if left_img is not None:
        display.blit(pygame.surfarray.make_surface(pg_left_img), (WIDTH, 0))
    if forward_img is not None:
        display.blit(pygame.surfarray.make_surface(pg_forward_img), (WIDTH, ADD_WIDTH))
    if right_img is not None:
        display.blit(pygame.surfarray.make_surface(pg_right_img), (WIDTH, 2*ADD_WIDTH))
    """


    input_text = font_30.render(" "+name_inp+" ", True, (0, 0, 0), (255,255,255))


    # draw reading/writing mode
    if reading_mode:
        display.blit(reading_mode_text, (5,0))
    else:
        display.blit(writing_mode_text, (5,0))

        writing_face_text = "..."
        if writing_index == 1:
            writing_face_text = "Look left!"
        elif writing_index == 2:
            writing_face_text = "Look forwards!"
        elif writing_index == 3:
            writing_face_text = "Look right!"

        writing_face_surf = font_20.render(writing_face_text, True, (255,0,0))
        w = writing_face_surf.get_width()
        
        display.blit(writing_face_surf, ((WIDTH-w)/2, 5))

        if writing_index == 4:
            display.blit(enter_name_text, ((WIDTH-enter_name_text.get_width())//2,
                                           (HEIGHT-input_text.get_height())//2 - enter_name_text.get_height()))
        
        if (len(faces) == 0):
            display.blit(no_faces_text, (5,writing_mode_text.get_height()))

    # draw box on face
    for (x, y, w, h) in haar_detection:
        crop_img = frame[y:y+h, x:x+w, :]
        crop_gray = gray[y:y+h, x:x+w]
        
        resized_img = cv2.resize(crop_img, (face_img_diameter, face_img_diameter))
        last_img = resized_img
        pg_last_img = cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB)
        pg_last_img = np.rot90(pg_last_img)
        pg_last_img = np.flip(pg_last_img, axis=0)

        
        pygame.draw.rect(display, (255,0,0), (x,y,w,h), 1, border_radius=1)


        label, confidence = recognizer.predict(crop_gray)
        label_txt = str(confidence)
        if confidence < 115:
            label_txt = names_dict[label]
        text_surface = font_20.render(label_txt, True, (255, 0, 0))
        display.blit(text_surface, (x,y+h))


    # text input:
    if writing_index == 4:
        temp_name_inp = name_inp
        if temp_name_inp == "":
            temp_name_inp = "..."
        display.blit(input_text, ((WIDTH-input_text.get_width())//2,(HEIGHT-input_text.get_height())//2))

    # confirm / deny face
    hovering_yes = False
    hovering_no = False
    if writing_index == 5:
        cbox_padding = 12
        cbox_spacing = 40

        cbox_width = 3*face_img_diameter
        cbox_height = face_img_diameter+3*cbox_padding+confirm_prompt_text.get_height()+yes_text.get_height()
        cbox_x = (WIDTH-cbox_width)//2
        cbox_y = (HEIGHT-cbox_height)//2
        
        pygame.draw.rect(display, (255,255,255), (cbox_x,cbox_y,cbox_width,cbox_height))

        display.blit(confirm_prompt_text, (cbox_x+(cbox_width-confirm_prompt_text.get_width())//2,
                                           cbox_y + face_img_diameter + cbox_padding))


        yes_x = cbox_x+(cbox_width - yes_text.get_width())//2 - cbox_spacing
        yes_y = cbox_y + face_img_diameter + 2*cbox_padding + confirm_prompt_text.get_height()

        no_x = yes_x + 2*cbox_spacing
        no_y = yes_y

        mx,my = pygame.mouse.get_pos()
        button_outline_x = 5
        button_outline_y = 2

        if (yes_text.get_rect().collidepoint((mx-yes_x,my-yes_y))):
            hovering_yes = True
            pygame.draw.rect(display, (0,0,0), (yes_x-button_outline_x, yes_y-button_outline_y,
                                                yes_text.get_width()+2*button_outline_x, yes_text.get_height()+2*button_outline_y), 2, 1)
        if (no_text.get_rect().collidepoint((mx-no_x,my-no_y))):
            hovering_no = True
            pygame.draw.rect(display, (0,0,0), (no_x-button_outline_x, no_y-button_outline_y,
                                                no_text.get_width()+2*button_outline_x, no_text.get_height()+2*button_outline_y), 2, 1)
        
        display.blit(yes_text, (yes_x,yes_y))
        display.blit(no_text, (no_x,no_y))

        if left_img is not None:
            display.blit(pygame.surfarray.make_surface(pg_left_img), (cbox_x, cbox_y))
        if forward_img is not None:
            display.blit(pygame.surfarray.make_surface(pg_forward_img), (cbox_x +face_img_diameter, cbox_y))
        if right_img is not None:
            display.blit(pygame.surfarray.make_surface(pg_right_img), (cbox_x +2*face_img_diameter, cbox_y))
    
    pygame.display.update()

    
pygame.quit()
vc.release()
