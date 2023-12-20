import streamlit as st
from pytube import YouTube
import hydralit_components as hc
from streamlit.components.v1 import html
#--------------------------#
import cv2
import numpy as np
import os
import time
import mediapipe as mp
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import pygame
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
#--------------------------#

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

pygame.init()

st.set_page_config(
    page_title='bisindo',
    layout='wide',
    initial_sidebar_state='expanded',
)

#----------------Interpreter-----------------------------#
# MP Holistic:
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def text_to_speech(text):
    tts = gTTS(text=text, lang='id')
    tts.save('output.mp3')
    pygame.mixer.music.load('output.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    os.remove('output.mp3')

def mediapipe_detection(image, model):
    image = frame.to_ndarray(format="bgr24")
    results = model.process(image)                
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

# Extract Keypoint values
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# Load model:
model = tf.keras.models.load_model('./pretrained_models/model_thirdrun.h5')

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Visualize prediction:
def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()

    pred_dict = dict(zip(actions, res))
    # sorting for prediction and get top 5
    prediction = sorted(pred_dict.items(), key=lambda x: x[1])[::-1][:5]

    for num, pred in enumerate(prediction):
        text = '{}: {}'.format(pred[0], round(float(pred[1]),4))
        # cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, text, (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA) 
    return output_frame

class VideoProcessor:
    def _init_(self):
        self.video_container = st.container()

    def recv(self, frame):
        self.video_container.image(frame.to_ndarray(format="bgr24"), channels="BGR")


#--------------------------------------------------------#

#------------------------Belajar-------------------------#
# Fungsi untuk mendapatkan judul dari YouTube URL"
def get_video_title(youtube_url):
    try:
        yt = YouTube(youtube_url)
        title = yt.title
        # Menghapus "Bisindo" dari judul dan mengambil kata setelahnya
        if "Bisindo" in title:
            title_parts = title.split("Bisindo")
            if len(title_parts) > 1:
                return title_parts[1].strip()
        return title
    except Exception as e:
        st.error(f"Gagal mendapatkan judul dari URL {youtube_url}.")
        st.error(str(e))
        return "Video Tidak Tersedia"

# Fungsi untuk menampilkan video dari YouTube
def display_youtube_video(youtube_url):
    video_id = youtube_url.split("=")[-1] if "=" in youtube_url else youtube_url.split("/")[-1].split("?")[0]
    video_url = f"https://www.youtube.com/embed/{video_id}"
    st.write(f'<iframe width="280" height="157" src="{video_url}" frameborder="0" allowfullscreen style="margin-bottom: 20px;"></iframe>', unsafe_allow_html=True)

#----------------Mini Game------------------#
def initialize_session_state():
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'player_score' not in st.session_state:
        st.session_state.player_score = 0

def update_score(player_choice, correct_answer):
    if player_choice.lower() == correct_answer.lower():
        st.session_state.player_score += 1

def calculate_score(player_choice):
    correct_answer = quiz_questions[st.session_state.current_question]['answer']
    update_score(player_choice, correct_answer)
    st.session_state.current_question += 1

quiz_questions = [
    {
        'image_path': 'gambar_soal/gambar_D.jpg',
        'answer': 'D'
    },
        {
        'image_path': 'gambar_soal/gambar_I.jpg',
        'answer': 'I'
    },
    {
        'image_path': 'gambar_soal/gambar_M.jpg',
        'answer': 'M'
    },
    {
        'image_path': 'gambar_soal/gambar_A.jpg',
        'answer': 'A'
    },
    {
        'image_path': 'gambar_soal/gambar_S.jpg',
        'answer': 'S'
    }
]

#--------------------------------------------#

#----------------Tentang---------------------#
def about_section():

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1 ])
    with col2:
            st.image('./streamlit_files/dimas.jpg', width=150)
            
    with col3:
        st.markdown(
            """
            <style>
                .centered-content {
                    display: flex;
                    align-items: center;
                    justify-content: flex-start;
                    margin-bottom: 20px;
                }

                .content-text {
                    margin-left: -120px;
                    margin-right: -39px;
                }

                .content-text p {
                    margin-bottom: 10px;
                    text-align: justify;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="centered-content">
                <div class="content-text">
                    <p>Menurut Organisasi Kesehatan Dunia (WHO), terdapat sekitar 450 juta orang yang tunarungu atau memiliki kesulitan mendengar. Satu-satunya cara mereka dapat berkomunikasi satu sama lain adalah melalui bahasa isyarat. Namun, bahasa isyarat belum begitu populer di kalangan masyarakat umum. Hal ini membuat komunitas tunarungu sulit mengakses layanan publik dan berkomunikasi dengan orang biasa serta mengembangkan karier mereka. Oleh karena itu, Web App ini lahir untuk memudahkan komunitas tunarungu dalam berkomunikasi dan menjalani kehidupan yang lebih baik.</p>
                    <p><a href="https://docs.google.com/document/d/1HLpP3jvJzZjQL8kz0skGSLmqjiYaHHHtzQiFid2quxc/edit">Informasi Lebih Lanjut - Google Doc</a></p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

#-----------------------------------------------#
        
#------------------Streamlit--------------------#
    
def main():
    # Menyesuaikan margin dan lebar kolom agar tidak menumpuk
    over_theme = {'txc_inactive': 'white','menu_background':'green','txc_active':'green','option_active':'white'}
    font_fmt = {'font-class':'h2','font-size':'150%', 'width':'200px'}


    menu = hc.nav_bar(
        menu_definition=[
            {"label": "Beranda"},
            {"label": "Sign-Talk"}, 
            {"label": "Belajar"},
            {"label": "Mini Games"},           
            {"label": "Tentang"}
        ],
        key='PrimaryOption',
        override_theme=over_theme,
        # font_styling=font_fmt,
        # horizontal_orientation=True,
    )


    st.markdown(
        """
        <style>
            .st-df {
                column-gap: 20px;
                padding: 0px !important;
            }
            .st-cg {
                padding: 0px !important;
                margin: 0px !important;
            }
            header {
                text-align: center;
            }
            .stApp {
                text-align: center;
            }
            .st-emotion-cache-z5fcl4 {
            padding:0!important;
            }

        </style>
        """,
        unsafe_allow_html=True
    )
    if menu == "Beranda":
        st.markdown(
            """
            <div style="text-align:center; margin-top:-75px;">
            <h1 style="font-variant: small-caps; font-size: xx-large; margin-bottom:-45px;">
            <font color=#ea0525>w e b a p p</font>
            </h1>
            <h1>  Real-Time Indonesian Sign Language Interpreter </h1>
            <hr>
            </div>
            """,
            unsafe_allow_html=True,
        )

        centered_style = """
            <style>
            .st-emotion-cache-1kyxreq {
                display: flex;
                flex-flow: wrap;
                row-gap: 0rem;
                justify-content: center;
            }
            </style>
        """

        st.markdown(centered_style, unsafe_allow_html=True)
        st.image('./streamlit_files/bisindo.jpg', width=400)

        st.markdown(
            """
            <div style="max-width: 800px; margin: 0 auto; text-align: center;">
                <p>BISINDO (Bahasa Isyarat Indonesia)</p>
                <p>Bahasa isyarat ini lah yang sering ditemukan di kalangan Teman Tuli maupun Teman Inklusi pengguna bahasa isyarat. BISINDO dibentuk oleh kelompok Tuli dan muncul secara alami berdasarkan pengamatan Teman Tuli. Maka dari itu, BISINDO memiliki variasi “dialek” di berbagai daerah. BISINDO disampaikan dengan gerakan dua tangan.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Tampilan untuk "Beranda"
    elif menu == "Belajar":
        st.title("Pelajari Bahasa Isyarat Indonesia")
        st.markdown("""
             <div style="max-width: 800px; margin: 0 auto; text-align: center;">
                <p>Mari berkomunikasi tanpa hambatan! Di Indonesia terdapat lebih dari 2.500.000 tuli dan Bisindo adalah bentuk komunikasi yang paling efektif serta tidak terbatas hanya untuk Tuli tetapi juga untuk semua orang. Selain untuk mengurangi hambatan dalam berkomunikasi dan mendukung lingkungan yang inklusif, mempelajari BISINDO juga mempunyai banyak manfaat.</p>
            </div>
        """, unsafe_allow_html=True)

        youtube_urls = [
            "https://www.youtube.com/watch?v=GCFfwXFi6hA",
            "https://www.youtube.com/watch?v=S-2Lj8OzPqQ",
            "https://www.youtube.com/watch?v=MvyeZK6hHPg",
            "https://www.youtube.com/watch?v=w2w1RwtyGnI",
            "https://www.youtube.com/watch?v=1zexupTxhjY",
            "https://www.youtube.com/watch?v=yF5D1P61ruc",
            "https://www.youtube.com/watch?v=RSLinLUs0no",
            "https://www.youtube.com/watch?v=dIM8D0UzTcM",
            "https://www.youtube.com/watch?v=0kym9mtmJHY",
            "https://www.youtube.com/watch?v=yiNQZ4qP-gQ",
            "https://www.youtube.com/watch?v=oFnNk1hXvJQ",
            "https://www.youtube.com/watch?v=AEpD4bCMFF8",
            "https://www.youtube.com/watch?v=-xeL9SVqouw",
            "https://www.youtube.com/watch?v=CMFnFbqVVsM",
            "https://www.youtube.com/watch?v=A6VuG_vz5KI",
            "https://www.youtube.com/watch?v=KmGQxAMJJ5M",
            "https://www.youtube.com/watch?v=pnZFh69CM38",
            "https://www.youtube.com/watch?v=MePOVCZDgLk",
            "https://www.youtube.com/watch?v=sQOpGjJTdpo",
            "https://www.youtube.com/watch?v=qzRnU5vuCSo",
            "https://www.youtube.com/watch?v=im2wtb77WnQ",
            "https://www.youtube.com/watch?v=q0zJl-IhTcc",
            "https://www.youtube.com/watch?v=sGjptC-vJ30",
            "https://www.youtube.com/watch?v=1OF6gpG3fFs",
            "https://www.youtube.com/watch?v=3deDWmIofrk",
            "https://www.youtube.com/watch?v=Xc4qx-mOLPY",
            "https://www.youtube.com/watch?v=l94RDnOfmeg",
            "https://www.youtube.com/watch?v=pndvoL6wNck",
            "https://www.youtube.com/watch?v=xZOOLmwVOMI",
            "https://www.youtube.com/watch?v=d96g8WoWlKQ",
            "https://www.youtube.com/watch?v=5hYs_KLpkHg",
            "https://www.youtube.com/watch?v=IamAOXWNltM",
            "https://www.youtube.com/watch?v=vXJ8ZgnHqfk",
            "https://www.youtube.com/watch?v=HjJkym6puho",
            "https://www.youtube.com/watch?v=Mjn58G6caoE",
            "https://www.youtube.com/watch?v=x52qePxNJ3c",
            "https://www.youtube.com/watch?v=CpbMam8sW6o",
            "https://www.youtube.com/watch?v=ZjtxSsfPSCk",
            "https://www.youtube.com/watch?v=d1ZlBAPnvxA",
            "https://www.youtube.com/watch?v=fh7qEh0o3Og",
            "https://www.youtube.com/watch?v=1ikWnb32OYk",
            "https://www.youtube.com/watch?v=ehzPuduoGDA",
            "https://www.youtube.com/watch?v=lBhIR2ZbZ7k",
            "https://www.youtube.com/watch?v=5cSgPUwH244",
            "https://www.youtube.com/watch?v=rOuKqjW5CpE",
            "https://www.youtube.com/watch?v=Kpzc5WCSXVU",
            "https://www.youtube.com/watch?v=9rlOk6HjQMM",
            "https://www.youtube.com/watch?v=rrJo-FV6yf0",
            "https://www.youtube.com/watch?v=sra7h0dqzy0",
            "https://www.youtube.com/watch?v=BqzqzU3Vsb4",
            "https://www.youtube.com/watch?v=5-aZ3vMVa9g",
            "https://www.youtube.com/watch?v=bugvwCHPzpw"
        ]

        col1, col2, col3, col4 = st.columns(4)

        for i, url in enumerate(youtube_urls):
            try:
                video_title = get_video_title(url)
                if i % 4 == 0:
                    with col1:
                        st.markdown(
                            f"<h3 style='text-align: center;'>{video_title}</h3>",
                            unsafe_allow_html=True
                        )
                        display_youtube_video(url)
                elif i % 4 == 1:
                    with col2:
                        st.markdown(
                            f"<h3 style='text-align: center;'>{video_title}</h3>",
                            unsafe_allow_html=True
                        )
                        display_youtube_video(url)
                elif i % 4 == 2:
                    with col3:
                        st.markdown(
                            f"<h3 style='text-align: center;'>{video_title}</h3>",
                            unsafe_allow_html=True
                        )
                        display_youtube_video(url)
                else:
                    with col4:
                        st.markdown(
                            f"<h3 style='text-align: center;'>{video_title}</h3>",
                            unsafe_allow_html=True
                        )
                        display_youtube_video(url)
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam memuat video dari URL {url}.")
                st.error(str(e))

    elif menu == "Mini Games":
        # CSS untuk Mini Game
        centered_style = """
            <style>
            .centered {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }

            div.stTextInput {
                width: 250px;
                margin: auto;
             }

            .st-cg {
                padding: 13px !important;
                margin: 0px !important;
            }
            .centered-image {
                width: 300px; /* Lebar gambar */
                margin-bottom: 20px; /* Jarak dari gambar ke elemen lain */
            }

            .st-emotion-cache-1kyxreq {
                display: flex;
                flex-flow: wrap;
                row-gap: 0rem;
                justify-content: center;
            }
            </style>
        """

        st.markdown(centered_style, unsafe_allow_html=True)

        initialize_session_state()

        ind = st.session_state.current_question

        if ind < len(quiz_questions):
            current_question = quiz_questions[ind]
            image_path = current_question["image_path"]
            
            st.markdown("<div class='centered'><h1>Mini Game</h1></div>", unsafe_allow_html=True)
            st.image(image_path, width=300, caption=f"Question {ind + 1}/{len(quiz_questions)}")  # Mengatur lebar gambar menjadi 300 piksel dan menambahkan keterangan
                
            player_choice = st.text_input("Your Answer", key=f"question_{ind}") 

            # Memposisikan tombol Submit di tengah
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                if st.button("Submit", key=f"submit_{ind}"):
                    calculate_score(player_choice)
                    
                    if st.session_state.current_question < len(quiz_questions):
                        st.experimental_rerun()

                    if st.session_state.current_question >= len(quiz_questions):
                        st.success("Kuis Telah Selesai")
                        st.button("Cek Nilai Anda", on_click=initialize_session_state)
        else:
            st.warning(f"Selamat anda telah menyelesaikan mini game. Nilai Anda:{st.session_state.player_score}")
            if st.button("Main Lagi"):
                st.session_state.current_question = 0
                st.session_state.player_score = 0

    elif menu == "Tentang":
        st.markdown(
            """
            <div style="text-align:center; margin-top:-75px;">
            <h1 style="font-variant: small-caps; font-size: xx-large; margin-bottom:-45px;">
            <font color=#ea0525>t e n t a n g</font>
            </h1>
            <h1>  Real-Time Indonesian Sign Language Interpreter </h1>
            <hr>
            </div>
            """,
            unsafe_allow_html=True,
        )
        about_section()

    elif menu == "Sign-Talk":
        st.markdown(
            """
            <div style="text-align:center; margin-top:-75px;">
            <h1>  SIGN-TALK </h1>
            <h3 style="font-weight: normal;">Realtime Interpreter Bahasa Isyarat Indonesia dengan Text to Speech</h3>
            <hr>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # New detection variables
        sequence = []
        sentence = []
        threshold = 0.9
        tts = False
        actions = os.listdir('./MP_Data')
        label_map = {label:num for num, label in enumerate(actions)}

        # Checkboxes
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1,3])
        with col2:
            show_webcam = st.checkbox('Show webcam')
        with col3:
            show_landmarks = st.checkbox('Show landmarks')
        with col4:
            speak = st.checkbox('Speak')

        # Webcam
        col1, col2, col3 = st.columns([1.5, 3, 1])
        with col2:
            FRAME_WINDOW = st.image([])
            client_settings = ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},  # Matikan audio
            )

            cap = webrtc_streamer(
                    key="example",
                    video_processor_factory=VideoProcessor,
                    async_processing=True,
                    client_settings=client_settings,
                )
            
        # Mediapipe model 
            with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
                while show_webcam:
                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    if show_landmarks:
                        draw_styled_landmarks(image, results)
                    
                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)

                    sequence.append(keypoints)
                    sequence = sequence[-24:]
                    
                    if len(sequence) == 24:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]

                        #3. Viz logic
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    # incase the first word is iloveyou:
                                    if (sentence[0] == '') and (actions[np.argmax(res)] == 'i love you'):
                                        pass
                                    else:
                                        sentence.append(actions[np.argmax(res)])
                                        tts = True
                            else:
                                sentence.append(actions[np.argmax(res)])
                                tts = True

                        if len(sentence) > 5: 
                            sentence = sentence[-5:]

                        # Viz probabilities
                        if show_landmarks:
                            image = prob_viz(res, actions, image)

                        # Text to speak:
                        if speak:
                            if tts: 
                                text_to_speech(sentence[-1])
                                tts = False

                        # show result
                        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image, ' '.join(sentence), (3,30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    # cv2.imshow('OpenCV Feed', image)
                    frameshow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frameshow)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                

if __name__ == "__main__":
    main()
