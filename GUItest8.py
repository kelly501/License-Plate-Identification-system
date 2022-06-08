#from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
import numpy as np
import time
import math
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'D:\Tesseract\\tesseract.exe'
import difflib as diff

window = tk.Tk()
window.title('Identification')
window.geometry("800x480")
window.configure(bg='#8B7765')
# window.resizable(width=False, height=False)
align_mode = 'nswe'
pad = 5   #上下左右各添加多少 ( pixel )

window.update()
win_size = min( window.winfo_width(), window.winfo_height())  #獲取窗口寬度、高度（單位：像素）
# print(win_size)

# div_size = win_size
# img_size = div_size * 3
# div1 = tk.Frame(window, width=800, height=300, bg='#E0E0E0')
# div2 = tk.Frame(window, width=800, height=180, bg='#C4E1FF')

# div1.pack() #sticky:對齊方式
# div2.pack()

# window.update()
# w1,h1=div1.winfo_width(), div1.winfo_height()
# print(w1,h1)


video_canvas = tk.Canvas(window, width=440, height=280, bg = '#CDB38B')
video_canvas.place(x=25, y=95)

image_canvas = tk.Canvas(window, width=260, height=110, bg='#EED8AE',relief='groove')
image_canvas.place(x=500, y=35)

text_Iden = tk.Text(window, width=14, height=2, bg='#EECBAD', fg='#642100')
text_Iden.place(x=500, y=180)
text_Iden.configure(font=("Courier", 23, "italic"))

circle_canvas = tk.Canvas(window, width=280, height=160, bg='#EEE8CD')
circle_canvas.place(x=500, y=280)

cap = cv2.VideoCapture('test0511.h264')
door = 0    # 1開 0關
door_t = time.ctime() #開門時間點


f = open('plat.txt', 'w')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def read_data():
    global  data
    r = open("data.txt", 'r')
    data = r.read()
    data = data.split('\n')
    for s in range(len(data)):
        data[s] = data[s].split(" ")
read_data()


def rota_img(img):
    ret2, binary_dst_inv = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(binary_dst_inv > 0))
    angle = cv2.minAreaRect(coords)[2]
    d_h, d_w = img.shape
    if angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
    center = (d_w // 2, d_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (d_w, d_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return  rotated

def check(plt):
    global data,door_t,door

    ratio_plat = []

    for i in data:
        b = diff.SequenceMatcher(None, plt, i[1]).ratio()
        ratio_plat.append(b)
    if max(ratio_plat) > 0.7:
        # 0.7-> 5/7
        co_plat = ratio_plat.index(max(ratio_plat))
        print('合法車牌'+' '+data[co_plat][1])
        door_t = time.process_time()
        door = 1
        setTextInput(data[co_plat][1]+ '\n' + '合法車牌')
    else:
        print('錯誤車牌')
        door = 0
        setTextInput('\n'+ '非合法車牌')
def carnum2(img_o):
    global text,time_now
    img_o = img_o[600:1100,800:1100]
    # img_o = img_o[100:400,100:400]
    new_img = img_o.copy()
    # print(new_img.shape)
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    pi = []
    for c in contours:
        x = []
        y = []
        for point in c:
            y.append(point[0][0])
            x.append(point[0][1])
        r = [min(y), min(x), max(y), max(x), (max(x) + min(x)) // 2, (max(y) + min(y)) // 2]

        if cv2.contourArea(c) < 200 or cv2.contourArea(c) > 4*300 or float((r[3]-r[1])/(r[2]-r[0])) < 0.5 or float((r[3]-r[1])/(r[2]-r[0])) > 7.5:
            contours = np.delete(contours,count,0)
            count = count - 1
        else:
            pi.append(r)
        count = count + 1

    # 畫刪除後的輪廓圖
    # contours_cut_img = new_img.copy()
    contours_cut_img = cv2.drawContours(new_img.copy(), contours, -1, (0, 255, 0), 2)

    # 區塊分類
    count_p = [[0]]
    for i in range(1, len(contours)):
        for j in range(len(count_p)):
            i_h = pi[i][3] - pi[i][1]  # 首塊高度
            j_h = pi[count_p[j][0]][3] - pi[count_p[j][0]][1]  # 檢視的高度
            # 首塊中心 pi[i][4],pi[i][5]
            ij_l = math.sqrt((pi[i][4] - pi[count_p[j][0]][4]) ** 2 + (pi[i][5] - pi[count_p[j][0]][5]) ** 2)
            if pi[i][5] - pi[count_p[j][0]][5] == 0:
                ij_m = 999
            else:
                ij_m = (pi[i][4] - pi[count_p[j][0]][4]) / (pi[i][5] - pi[count_p[j][0]][5])

            if i_h / j_h > 0.8 and i_h / j_h < 1.2 and ij_l < 4*i_h and ij_m < 0.3 and ij_m > -0.3:
                count_p[j].append(i)
                break
            if j == (len(count_p) - 1):
                count_p.append([i])
    # print(count_p)

    for i in range(len(count_p)):
        # 取出區塊中6-7
        count_img_plate = 0

        if len(count_p[i]) < 8 and len(count_p[i]) > 5:
            # 依照y大小牌順序
            for ii in range(len(count_p[i])):
                for jj in range(len(count_p[i])):
                    y_ii = []
                    for point in contours[count_p[i][ii]]:
                        y_ii.append(point[0][0])
                    y_ii = min(y_ii)
                    y_jj = []
                    for point in contours[count_p[i][jj]]:
                        y_jj.append(point[0][0])
                    y_jj = min(y_jj)
                    if y_jj > y_ii:
                        t = count_p[i][jj]
                        count_p[i][jj] = count_p[i][ii]
                        count_p[i][ii] = t
            x_all = []
            y_all = []

            for j in range(len(count_p[i])):

                x = []
                y = []
                for point in contours[count_p[i][j]]:
                    y.append(point[0][0])
                    x.append(point[0][1])
                r = [min(y), min(x), max(y), max(x)]
                x_all.append(r[1])
                x_all.append(r[3])
                y_all.append(r[0])
                y_all.append(r[2])
            # print("count_p",count_p[i])

            r_all = [min(y_all), min(x_all), max(y_all), max(x_all)]

            out_img = new_img.copy()

            mask = np.zeros(out_img.shape[:2], np.uint8)
            for k in range(len(count_p[i])):
                cv2.drawContours(mask, contours, count_p[i][k], 255, -1)
            dst = cv2.bitwise_and(out_img, out_img, mask=mask) + 255
            # print('dst',type(dst))
            if r_all[1] > 10 and r_all[0] > 10 and r_all[3]+10 < binary_img.shape[0] and r_all[2]+10 < binary_img.shape[1]:
                dst = dst[r_all[1]-10:r_all[3]+10,r_all[0]-10:r_all[2]+10]
            else:
                dst = dst[r_all[1]:r_all[3],r_all[0]:r_all[2]]
            gray_dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

            ret2, binary_dst = cv2.threshold(gray_dst, 150, 255, cv2.THRESH_BINARY)
            binary_dst = rota_img(binary_dst)
            image(binary_dst)
            count_img_plate += 1
            text = pytesseract.image_to_string(binary_dst)
            re_text = ''
            for p in text:
                if p == 'O':
                    re_text += '0'
                elif p == '|':
                    re_text += '1'
                elif p == 'I':
                    re_text += '1'
                elif p == ' ':
                    re_text += '-'
                else:
                    re_text += p
            check(re_text[0:-2])



            cv2.rectangle(new_img, (min(y_all)-3, min(x_all)-3), (max(y_all)+3, max(x_all)+0), (0, 0, 255), 4)
            cv2.rectangle(contours_cut_img, (min(y_all), min(x_all)), (max(y_all), max(x_all)), (0, 0, 255), 3)
    # time_now = time.ctime()

    # cv2.putText(contours_cut_img, time_now, (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # cv2.waitKey(0)
    return new_img,contours_cut_img


def video_imd():
    global door,door_t
    global time_now
    success, img = cap.read()
    time_now = time.ctime()
    img_show = img.copy()
    w, h, _ = img_show.shape
    rw, rh = w // 4, h // 4
    # time.sleep(0)
    if door == 1 and time.process_time() - door_t < 5:
        # time.sleep(0.02)
        img_show = cv2.resize(img, (400, 240), interpolation=cv2.INTER_CUBIC)
        cv2.putText(img_show, time.ctime(), (0, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.circle(img_show, (15, 15), 5, (0, 255, 0), 3)
        circle_canvas.create_oval(15, 25, 135, 145, outline="green", width=12)
        circle_canvas.create_oval(150, 25, 270, 145, outline="#3C3C3C", width=12)
        circle_canvas.create_text(78, 85, font=("Courier", 20), text="Open", fill="green")
        circle_canvas.create_text(213, 85, font=("Courier", 20), text="Close", fill="#3C3C3C")
        # cv2.imshow("imgwith plat", n)
        # cv2.imshow("video", img_show)
    else:
        door = 0
        n, c = carnum2(img)
        img_show[600:1100, 800:1100] = n
        img_show = cv2.resize(img_show, (400, 240), interpolation=cv2.INTER_CUBIC)
        cv2.putText(img_show, time.ctime(), (0, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.circle(img_show, (15, 15), 5, (0, 0, 255), 3)
        circle_canvas.create_oval(15, 25, 135, 145,  outline="#3C3C3C", width=12)
        circle_canvas.create_oval(150, 25, 270, 145, outline="red", width=12)
        circle_canvas.create_text(78, 85, font=("Courier", 20), text="Open", fill="#3C3C3C")
        circle_canvas.create_text(213, 85,  font=("Courier", 20), text="Close",fill="red")
        # cv2.imshow("imgwith plat", n)
        # cv2.imshow("video", img_show)


    video_frame = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)  # 把图像从BGR转到RGB
    img_frame = ImageTk.PhotoImage(Image.fromarray(video_frame))
    video_canvas.create_image(222, 142, image=img_frame)
    video_canvas.img = img_frame
    video_canvas.after(10, video_imd)  # 每30毫秒重置副程式
video_imd()

def image(img_to_canvas):
    img = cv2.cvtColor(img_to_canvas, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (250, 100), interpolation=cv2.INTER_CUBIC)
    img = ImageTk.PhotoImage(Image.fromarray(img))
    image_canvas.create_image(132,58, image = img)
    image_canvas.img = img

def setTextInput(t):
    text_Iden.delete(1.0,"end")
    text_Iden.insert(1.0, t)



window.mainloop()