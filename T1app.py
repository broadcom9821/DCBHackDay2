from tkinter.ttk import Separator

import face_recognition
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from operator import itemgetter
import tkinter.font as font
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import glob
from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# data = np.random.random(size=(10, 10, 10))
# z, x, y = data.nonzero()
# ax.scatter(x, y, z, c=z, alpha=1)
# plt.show()
window = Tk()
window.title("DCB")
window.geometry('800x180')
i = IntVar()
images = []
names = []
imagesencodings = []
vidpath = ''
first_appear = []

def choisebutvid():
    global vidpath
    vidpath = filedialog.askopenfilename()
def choisebut():
    print(i)
    if i.get() == True:
        dirpath = filedialog.askdirectory()
        if dirpath:
            filenames = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
            path = glob.glob(dirpath)
            print(path)
            for filename in filenames:
                print(filename)
                file_name = os.path.basename(filename)
                images.append(face_recognition.load_image_file(path[0]+'/'+filename))
                index_of_dot = file_name.index('.')
                names.append(file_name[:index_of_dot])
                imagesencodings.append(face_recognition.face_encodings(images[len(images) - 1])[0])
        else:
            pass
    else:
        filepathes = filedialog.askopenfilenames()
        if filepathes:
            print(filepathes)
            for filepath in filepathes:
                print(filepath)
                file_name = os.path.basename(filepath)
                images.append(face_recognition.load_image_file(filepath))
                index_of_dot = file_name.index('.')
                names.append(file_name[:index_of_dot])
                imagesencodings.append(face_recognition.face_encodings(images[len(images)-1])[0])
        else:
            pass
def submit():
    global first_appear
    global names
    global imagesencodings
    print(vidpath)
    cap = cv2.VideoCapture(vidpath)
    outpath = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=(("MP4 file", "*.mp4"),("All Files", "*.*") ))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    fps = 20
    out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'XVID'), 20, frame_size)
    print(outpath)
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    for i in range(len(names)):
        first_appear.append(-1)
    first_appear.append(-1)

    while cap.isOpened():
        # Grab a single frame of video
        ret, frame = cap.read()
        if ret == True:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            frame_count += 1
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(imagesencodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(imagesencodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = names[best_match_index]
                    if name != "Unknown":
                        if first_appear[names.index(name)] == -1:
                            first_appear[names.index(name)] = int(frame_count / fps)
                    elif name == "Unknown":
                        if first_appear[len(first_appear)-1] == -1:
                            first_appear[len(first_appear)-1] = int(frame_count / fps)
                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            time = int(frame_count / fps)
            cv2.putText(frame, str(time)+' S', (65, 65), font, 1.0, (255, 255, 255), 1)
            # Display the resulting image
            out.write(frame)
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    f = open('Recognition_Results.txt','w')
    for i in range(len(first_appear)-1):
        if first_appear[i] ==-1:
            f.write(names[i] + " -  didn't appear"+'\n')
        else:
            f.write(names[i]+' - '+str(first_appear[i])+' s'+'\n')
    if first_appear[len(first_appear)-1] ==-1:
        f.write("Unknown  -  hasn't appeared"+'\n')
    else:
        f.write('Unknown  - ' + str(first_appear[len(first_appear)-1]) + ' s'+'\n')
    f.close()
    # Release handle to the webcam
    cap.release()
    out.release()
    cv2.destroyAllWindows()
def clicked():
    garz = int(txtHi.get())
    garx = int(txtWi.get())
    gary = int(txtLe.get())
    filepath = filedialog.askopenfilename()
    f1 = open(filepath, 'r')
    arr = []
    while True:
        # считываем строку
        line = f1.readline()
        # прерываем цикл, если строка пустая
        if not line:
            break
        # выводим строку
        arr.append(list(map(int, line.split(sep=', '))))
    # закрываем файл
    # print(arr)
    f1.close()
    sidesp = (garx - 320) / 2
    backsp = (gary - 560)
    rowside = (sidesp - 60) / 2
    sideobj = []
    backobj = []
    s = 0
    b = 0
    for i in range(0, len(arr)):
        if arr[i][0] <= sidesp and arr[i][2] <= 150:
            temp = arr[i].copy()
            temp.append(i + 1)
            sideobj.append(temp)
        # 0,1,2
        if arr[i][0] <= sidesp and arr[i][1] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[0], temp[2], temp[1]
            temp.append(i + 1)
            sideobj.append(temp)
        # 0,2,1
        if arr[i][1] <= sidesp and arr[i][2] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[1], temp[0], temp[2]
            temp.append(i + 1)
            sideobj.append(temp)
        # 1,0,2
        if arr[i][1] <= sidesp and arr[i][0] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[1], temp[2], temp[0]
            temp.append(i + 1)
            sideobj.append(temp)
        # 1,2,0
        if arr[i][2] <= sidesp and arr[i][1] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[2], temp[0], temp[1]
            temp.append(i + 1)
            sideobj.append(temp)
        # 2,0,1
        if arr[i][2] <= sidesp and arr[i][0] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[2], temp[1], temp[0]
            temp.append(i + 1)
            sideobj.append(temp)
        # 2,1,0

        if arr[i][1] <= backsp and arr[i][2] <= 150:
            temp = arr[i].copy()
            temp.append(i + 1)
            backobj.append(temp)
            b += 1
        # 0,1,2
        if arr[i][2] <= backsp and arr[i][1] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[0], temp[2], temp[1]
            temp.append(i + 1)
            backobj.append(temp)
            b += 1
        # 0,2,1
        if arr[i][0] <= backsp and arr[i][2] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[1], temp[0], temp[2]
            temp.append(i + 1)
            backobj.append(temp)
            b += 1
        # 1,0,2
        if arr[i][2] <= backsp and arr[i][0] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[1], temp[2], temp[0]
            temp.append(i + 1)
            backobj.append(temp)
            b += 1
        # 1,2,0
        if arr[i][0] <= backsp and arr[i][1] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[2], temp[0], temp[1]
            temp.append(i + 1)
            backobj.append(temp)
            b += 1
        # 2,0,1
        if arr[i][1] <= backsp and arr[i][0] <= 150:
            temp = arr[i].copy()
            temp[0], temp[1], temp[2] = temp[2], temp[1], temp[0]
            temp.append(i + 1)
            backobj.append(temp)
            b += 1
        # 2,1,0

    # print(sideobj)
    # print(backobj)
    def MyFn(s):
        return s[0] * s[1]

    sideobj = sorted(sideobj, key=MyFn, reverse=True)
    backobj = sorted(backobj, key=MyFn, reverse=True)
    # print(sideobj)
    # print(backobj)

    y = 250
    y1 = y
    y2 = y
    y3 = y
    y4 = y
    x = garx - sidesp
    sideminus = 0
    indexlist = []
    indexlist2 = []
    control = 0
    both_sides = True
    plus_2_minus_first = True
    plus_2_minus_second = True
    for i in range(len(arr)):
        indexlist.append(i + 1)

    list_of_coordinates = []

    previous1 = 0
    previous2 = 0
    temp = 0

    def check(l1, l2, w1, w2, h1, h2):
        if l1 > l2 and w1 > w2 and h1 + h2 <= 150:
            return True
        else:
            return False

    for i in range(len(sideobj)):
        # if i == 0:
        #     list_of_coordinates.append(list(x, y - sideobj[i] + sideobj[0][0], y))
        if sideobj[i][3] in indexlist:
            if both_sides == True:
                if plus_2_minus_first == True:
                    temp = y1
                    y1 = y1 - sideobj[i][1]
                    if y1 > 0:
                        previous1 += sideobj[i][1]
                        list_of_coordinates.append(
                            [x, y1, sideobj[i][2], x + sideobj[i][0], y1 + sideobj[i][1], sideobj[i][2], sideobj[i][3]])
                        prov = 0
                        indexlist.remove(sideobj[i][3])
                        for j in range(len(sideobj) - 1):
                            if sideobj[j + 1][3] in indexlist:
                                if check(sideobj[i][0], sideobj[j + 1][0], sideobj[i][1], sideobj[j + 1][1],
                                         sideobj[i][2] + prov, sideobj[j + 1][2]) == True:
                                    list_of_coordinates.append(
                                        [x, y1, sideobj[j + 1][2] + sideobj[i][2] + prov, x + sideobj[i][0],
                                         y1 + sideobj[i][1], sideobj[j + 1][2] + sideobj[i][2] + prov,
                                         sideobj[j + 1][3]])
                                    prov += sideobj[j + 1][2] + sideobj[i][2]
                                    indexlist.remove(sideobj[j + 1][3])
                        plus_2_minus_first = False
                    else:
                        y1 = temp
                else:
                    y2 = y1 + previous1
                    previous1 += sideobj[i][1]
                    if y2 + sideobj[i][1] < 500:
                        list_of_coordinates.append(
                            [x, y2, sideobj[i][2], x + sideobj[i][0], y2 + sideobj[i][1], sideobj[i][2], sideobj[i][3]])
                        prov = 0
                        indexlist.remove(sideobj[i][3])
                        for j in range(len(sideobj) - 1):
                            if sideobj[j + 1][3] in indexlist:
                                if check(sideobj[i][0], sideobj[j + 1][0], sideobj[i][1], sideobj[j + 1][1],
                                         sideobj[i][2] + prov, sideobj[j + 1][2]) == True:
                                    list_of_coordinates.append(
                                        [x, y2, sideobj[j + 1][2] + sideobj[i][2] + prov, x + sideobj[i][0],
                                         y2 + sideobj[i][1], sideobj[j + 1][2] + sideobj[i][2] + prov,
                                         sideobj[j + 1][3]])
                                    prov += sideobj[j + 1][2] + sideobj[i][2]
                                    indexlist.remove(sideobj[j + 1][3])
                        plus_2_minus_first = True
                both_sides = False
            else:
                if plus_2_minus_second == True:
                    temp = y3
                    y3 = y3 - sideobj[i][1]
                    if y3 > 0:
                        previous2 += sideobj[i][1]
                        list_of_coordinates.append(
                            [x - 320 - sideobj[i][0], y3, sideobj[i][2], x - 320, y3 + sideobj[i][1], sideobj[i][2],
                             sideobj[i][3]])
                        prov = 0
                        indexlist.remove(sideobj[i][3])
                        for j in range(len(sideobj) - 1):
                            if sideobj[j + 1][3] in indexlist:
                                if check(sideobj[i][0], sideobj[j + 1][0], sideobj[i][1], sideobj[j + 1][1],
                                         sideobj[i][2] + prov, sideobj[j + 1][2]) == True:
                                    list_of_coordinates.append(
                                        [x - 320 - sideobj[i][0], y3, sideobj[j + 1][2] + sideobj[i][2] + prov, x - 320,
                                         y3 + sideobj[i][1], sideobj[j + 1][2] + sideobj[i][2] + prov,
                                         sideobj[j + 1][3]])
                                    prov += sideobj[j + 1][2] + sideobj[i][2]
                                    indexlist.remove(sideobj[j + 1][3])
                        plus_2_minus_second = False
                    else:
                        y3 = temp
                else:
                    y4 = y3 + previous2
                    previous2 += sideobj[i][1]
                    if y4 + sideobj[i][1] < 500:
                        list_of_coordinates.append(
                            [x - 320 - sideobj[i][0], y4, sideobj[i][2], x - 320, y4 + sideobj[i][1], sideobj[i][2],
                             sideobj[i][3]])
                        prov = 0
                        indexlist.remove(sideobj[i][3])
                        for j in range(len(sideobj) - 1):
                            if sideobj[j + 1][3] in indexlist:
                                if check(sideobj[i][0], sideobj[j + 1][0], sideobj[i][1], sideobj[j + 1][1],
                                         sideobj[i][2] + prov, sideobj[j + 1][2]) == True:
                                    list_of_coordinates.append(
                                        [x - 320 - sideobj[i][0], y4, sideobj[j + 1][2] + sideobj[i][2] + prov, x - 320,
                                         y4 + sideobj[i][1], sideobj[j + 1][2] + sideobj[i][2] + prov,
                                         sideobj[j + 1][3]])
                                    prov += sideobj[j + 1][2] + sideobj[i][2]
                                    indexlist.remove(sideobj[j + 1][3])
                        plus_2_minus_second = True
                both_sides = True

    x_zad = (garx - 200)/2
    y_zad = 560


    if y_zad < gary - 60:
        for i in range(len(backobj)):
            if backobj[i][3] in indexlist:
                if x_zad + backobj[i][0] < garx - ((garx - 200)/2) and y_zad + backobj[i][1] <= gary:
                    x_zad += backobj[i][0]
                    list_of_coordinates.append(
                        [x_zad, y_zad, backobj[i][2], x_zad + backobj[i][0], y_zad + backobj[i][1], backobj[i][2],
                         backobj[i][3]])
                    prov = 0
                    indexlist.remove(backobj[i][3])
                    for j in range(len(backobj) - 1):
                        if backobj[j + 1][3] in indexlist:
                            if check(backobj[i][0], backobj[j + 1][0], backobj[i][1], backobj[j + 1][1],
                                     backobj[i][2] + prov, backobj[j + 1][2]) == True:
                                list_of_coordinates.append(
                                    [x_zad, y_zad, backobj[j + 1][2] + backobj[i][2], x_zad + backobj[i][0],
                                     y_zad + backobj[i][1], backobj[j + 1][2] + backobj[i][2], backobj[j + 1][3]])
                                prov += backobj[j + 1][2] + backobj[i][2]
                                indexlist.remove(backobj[j + 1][3])

    outr = list_of_coordinates
    count = 0
    txtpath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=(("text file", "*.txt"),("All Files", "*.*") ))
    out = open(txtpath, "w")
    outr = sorted(outr, key=itemgetter(6))

    for i in range(len(outr)):
        for j in range(len(outr[i])):
            outr[i][j]=int(outr[i][j])
    print(outr)

    for i in range(len(outr)):
        str_a = ', '.join(map(str, outr[i]))
        out.write(str_a + '\n')
    out.close()




separator = Separator(window, orient='vertical')
separator.place(x=210.5, y=0, width=1, height=1000)
lbl = Label(window, text="Units calculator")
lbl['font'] = font.Font(size=30)
lbl0 = Label(window, text="Width")
lbl0['font'] = font.Font(size=20)
lbl1 = Label(window, text="Length")
lbl1['font'] = font.Font(size=20)
lbl2 = Label(window, text="Height")
lbl2['font'] = font.Font(size=20)
lbl.grid(row=0, columnspan=2)
lbl0.grid(column=0, row=1)
lbl1.grid(column=0, row=2)
lbl2.grid(column=0, row=3)

txtWi = Entry(window,width=10)
txtLe = Entry(window,width=10)
txtHi = Entry(window,width=10)
txtWi.grid(column=1, row=1)
txtLe.grid(column=1, row=2)
txtHi.grid(column=1, row=3)

lbl101 = Label(window, text="Face recognition")
lbl101['font'] = font.Font(size=30)
lbl101.grid(row=0, columnspan=2,column = 2)
lbl10 = Label(window, text="Выберите изображение/я")
lbl10['font'] = font.Font(size=20)
lbl11 = Label(window, text="Выберите видео для обработки")
lbl11['font'] = font.Font(size=20)
lbl10.grid(column=2, row=1)
lbl11.grid(column=2, row=2)
btn11 = Button(window, text="Выбрать", command=choisebut)
btn11['font'] = font.Font(size=20)
btn11.grid(column = 3,row=1)
btn12 = Button(window, text="Выбрать", command=choisebutvid)
btn12['font'] = font.Font(size=20)
btn12.grid(column = 3,row=2)
btn111 = Button(window, text="Submit", command=submit)
btn111['font'] = font.Font(size=20)
btn111.grid(column=3, row=3)
c = Checkbutton(window, text = "Пакетная загрузка", variable = i)
c['font'] = font.Font(size=20)
c.grid(column=4, row=1)

btns = Button(window, text="Submit", command=clicked)
btns['font'] = font.Font(size=20)
btns.grid(column=1, row=4)

window.mainloop()