import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy 
#load the trained model to classify sign
from keras.models import load_model
model = load_model('traffic_classifier.h5')

#dictionary to label all traffic signs class.
classes = { 0 : '터널',
1 : '+형교차로',
2 : 'T형교차로',
3 : 'Y형교차로',
4 : 'ㅏ형교차로',
5 : 'ㅓ형교차로',
6 : '좌합류도로',
7 : '우합류도로',
8 : '우선도로',
9 : '좌우로이중굽은도로',
10 : '우좌로이중굽은도로',
11 : '우로굽은도로',
12 : '좌로굽은도로',
13 : '2방향통행',
14 : '우측차로없어짐',
15 : '좌측차로없어짐',
16 : '도로폭이좁아짐',
17 : '회전형교차로',
18 : '철길건널목',
19 : '양측방통행',
20 : '우측방통행',
21 : '중앙분리대시작',
22 : '중앙분리대끝남',
23 : '낙석도로',
24 : '신호기',
25 : '미끄러운도로',
26 : '강변도로',
27 : '과속방지턱',
28 : '노면고르지못함',
29 : '내리막경사',
30 : '오르막경사',
31 : '횡풍',
32 : '횡단보도',
33 : '고인물튐',
34 : '비행기',
35 : '도로공사중',
36 : '자전거',
37 : '어린이보호',
38 : '위험',
39 : '야생동물보호',
40 : '위험물적재차량 통행금지',
41 : '통행금지표지판',
42 : '승용자동차통행금지',
43 : '화물자동차통행금지',
44 : '승합자동차통행금지',
45: '2륜자동차및 원동기 통행금지',
46: '우마차통행금지',
47 : '손수레통행금지',
48: '자전거통행금지',
49 : '경음기사용금지',
50 : '앞지르기금지',
51 : '직진금지',
52 : '횡단금지',
53 : '유턴금지',
54 : '우회전금지',
55 : '좌회전금지',
56 : '차폭제한',
57 : '최고속도제한',
58 : '차중량제한',
59 : '차높이제한',
60 : '최저속도제한',
61 : '차간거리확보',
62 : '주차금지',
63 : '진입금지',
64 : '주 정차금지',
65 : '서행',
66 : '양보',
67 : '일시정지',
68 : '보행자횡단금지',
69 : '보행자보행금지',
70 : '자동차전용도로',
71 : '자전거전용도로',
72 : '직진',
73 : '좌회전',
74 : '우회전',
75 : '직진 및 우회전',
76 : '좌우회전',
77 : '양측방통행',
78 : '좌측면통행',
79 : '경음기사용',
80 : '회전',
81 : '주차장',
82 :  '버스전용차로',
83 : '진행방향통행구분',
84 : '안전지대',
85 : '일방통행',
86 : '비보호좌회전',
87 : '아동보호',
88 : '횡단보도',
89 : '자전거횡단도',
90 : '보행자전용도로',
91 : '우회로' }
                 
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict(image)
    pred = numpy.argmax(pred, axis=1)
    sign = classes[pred[0]]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
