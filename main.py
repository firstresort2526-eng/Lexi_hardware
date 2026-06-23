# Server_url is 127.0.0.1/8000
import os
PROXY_URL = "http://38.154.203.95:5863"
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import threading
import speech_recognition as sr
#import ollama
from gtts import gTTS
import pygame
from fastapi import FastAPI, Request
import uvicorn
import firebase_admin
from firebase_admin import credentials, firestore
import queue
import ast
import requests, json
import dotenv
from PIL import Image
from io import BytesIO
import numpy as np
import socket
from google import genai
import time
import json
import re
from functools import wraps
from fireworks import Fireworks, AsyncFireworks

def get_rpi_ip():
    """Get the RPi's IP address on the WiFi interface"""
    try:
        # Connect to an external server to get the local IP
        # This doesn't actually send data, just gets the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Error getting IP: {e}")
        return None

# Get your RPi's IP
rpi_ip = get_rpi_ip()

dotenv.load_dotenv()
unsplash_API = os.getenv("ACCESS_KEY")
gemini_API = os.getenv("GEMINI_KEY")

client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])
explanation_chat = None
explanation_timestamp = 0
camera_queue = queue.Queue()
button_pressed = threading.Event()
voice_done = threading.Event()
voice_text = None

class Chat():
    def __init__(self,system_prompt):
        self.history = [{'role':'system','content':system_prompt}]
    
    def send_message(self,question,reasoning='low'):
        self.history.append({'role':'user','content':question})
        completion = client.chat.completions.create(
            model="accounts/fireworks/models/deepseek-v4-flash",
            messages=self.history,
            reasoning_effort=reasoning
        )
        answer = completion.choices[0].message.content
        self.history.append({'role':'assistant','content':answer})
        return answer

def count_time(func):
    @wraps(func)  # Preserves the original function's name and docstring
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Added the '=' sign
        
        # Execute the function and capture its return value
        result = func(*args, **kwargs)   
        
        end_time = time.perf_counter()
        print(f"The function {func.__name__} used {end_time - start_time:.6f} seconds.")
        
        return result  # Ensure the original function's output isn't lost
    return wrapper

#s = rpi_ip.split(".")
#OLLAMA_HOST = f'http://{s[0]}.{s[1]}.{s[2]}.47:11434'
#ollama_client = ollama.Client(host=OLLAMA_HOST)

# Initiate the firestore admin SDK
cred = credentials.Certificate("lexi-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def construct_ref(reflist, parent=db, i=0):
    if i >= len(reflist):
        return parent
    current = reflist[i]
    if i%2 == 0:
        new_parent = parent.collection(current)
    else:
        new_parent = parent.document(current)
    return construct_ref(reflist,parent=new_parent, i=i+1)

def read_data(reflist):
    '''
    This is a helper function that helps read data from the database. 
    Instead of doing the db.collection.document thing, you can simply do read_data([collection_name,document_name, ...])
    '''
    ref = construct_ref(reflist)
    snapshot = ref.get()
    if snapshot.exists:
        return snapshot.to_dict()
    else:
        return None
    
def word_exists(word):
    ref = construct_ref(["word_explanations","Developer","entries",word])
    snapshot = ref.get()
    if snapshot.exists:
        return True
    else:
        return False
    


def upload_data(path:list,json:dict):
    '''
    Upload the json data to the path, returns document id
    '''
    doc_ref = construct_ref(path)
    doc_ref.set(json)
    return json

def upload_explanation_data(word, userid, json: dict) -> str:
    '''
    Upload the json data to the word_explanations entries, returns document id
    '''
    return upload_data(['word_explanations',userid,'entries', word],json)


r = sr.Recognizer()
mic = sr.Microphone()
app = FastAPI()
pygame.mixer.init()
pygame.mixer.music.set_volume(0.15)

def rgb565(img_array):
    """
    Converts an (H, W, 3) RGB888 array to an (H, W, 2) RGB565 byte array.
    """
    # 1. Extract channels
    R = img_array[:,:,0].astype(np.uint16)
    G = img_array[:,:,1].astype(np.uint16)
    B = img_array[:,:,2].astype(np.uint16)

    # 2. Pack into a single 16-bit integer
    # Red: bits 11-15 | Green: bits 5-10 | Blue: bits 0-4
    rgb565 = ((R & 0xF8) << 8) | ((G & 0xFC) << 3) | (B >> 3)

    high_byte = (rgb565 >> 8).astype(np.uint8)
    low_byte = (rgb565 & 0xFF).astype(np.uint8)

    # 4. Stack back into (H, W, 2)
    # Use axis=-1 to create the new depth dimension
    return np.stack((high_byte, low_byte), axis=-1)

def project_image(word):
    url = f"https://api.unsplash.com/search/photos?query={word}&per_page=1"
    header = {'Authorization': f"Client-ID {unsplash_API}"}

    result = requests.get(url,headers=header).json()
    image_url = result['results'][0]['urls']['small']  # 400px width

    image_response = requests.get(image_url)
    img = Image.open(BytesIO(image_response.content))
    img_320 = img.resize((320, 320), Image.Resampling.LANCZOS)
    lcd_img = rgb565(np.array(img_320))
    projector_url = "http://127.0.0.1:8888/display_img"
    respond = requests.post(projector_url, json={'image':lcd_img})
    print(respond.json())

def play_explain_audio(explanation,slow=True, blocking=True):
    tts = gTTS(text=explanation, lang='yue', slow=slow)
    tts.save("output.mp3")

    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    if blocking:
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)

@count_time
def explain_text(cam, voice):
    global explanation_chat, explanation_timestamp
    play_explain_audio("我都諗緊個字嘅意思",slow=False,blocking=False)
    prompt = f"用家指着的字: {cam[0]}\n文字出處: {cam[1]}\n用家語音補充: {voice}"
    print(prompt)
    # Generate response from Ollama
    print("Thinking...")

    explanation_chat = Chat('''你係為讀寫障礙小學生而設嘅學習助手。用家會把鉛筆指向某中文字，請依照文字出處判斷用家希望得到解釋嘅詞語，請用廣東話
回答格式必須為有效JSON，不得包含其他文字。
{"word": 所問詞語, "meaning": 用作flashcard嘅極短解釋（10字以內）, "audio": 簡短、一至兩句小孩亦能明白嘅解釋...}''')
    text_reply = explanation_chat.send_message(prompt)
    reply = json.loads(text_reply)

    audio_reply = f"你指住嘅詞語係{reply['word']}，{reply['audio']}"
    explanation_timestamp = time.perf_counter()

    print(f"Lexi 學習助手: {reply}")
    play_explain_audio(audio_reply)

    word = reply['word']
    meaning = reply['meaning']
    # Upload to firebase

    if word_exists(word):
        ref = construct_ref(["word_explanations","Developer","entries",word])
        current_counter = read_data(["word_explanations","Developer","entries",word]).get("counter")
        ref.update({
            "counter": current_counter + 1,
            f"timestamp{current_counter}": firestore.SERVER_TIMESTAMP
        })
    else:
        upload_explanation_data(userid="Developer", word=word,json={
            "timestamp": firestore.SERVER_TIMESTAMP,
            "word": word,
            "explanation": meaning,
            "counter":  1
        })

    # Delete temp file
    if os.path.exists("/output.mp3"):
       os.remove("/output.mp3")

def process_voice_input():
    global voice_text
    try:
        with mic as source:
            print("Recording...")
            play_explain_audio("Lexi聽緊",slow=False)
            audio = r.listen(source, timeout=10, phrase_time_limit=20) 
            voice_text = r.recognize_google(audio, language="yue-Hant-hk")
        print(f"用家: {voice_text}")
    except:
        voice_text = None
    finally:
        voice_done.set()

@app.post("/button_press")
async def button_press():
    global voice_text
    voice_text = None
    voice_done.clear()
    threading.Thread(target=process_voice_input).start()
    button_pressed.set()
    return {"status": "button_pressed"}

@app.post("/camera_data")
async def process_camera_input(request: Request):
    request_data = await request.json()
    words_list = request_data['words']
    cam_data = words_list[0]['closest_char']
    context = ""
    for i in words_list:
        context += i['text'] 
    camera_queue.put((cam_data,context))
    return {"status": "camera_data_received"}

@count_time
def categorize_instructions(audio_prompt):
    system_prompt = '''用家會向你提出問題，請辨別問題種類。只須回答有效JSON，不可含有其他字句。
            1. 如何寫某字（例如「籃球點寫」、「點樣寫學習」） - 輸出{"type":"write","word":$用家在詢問的詞語}
            2. 某字的意思（例如用家問「呢個字點解？」或「咁呢隻字咩意思」或「我唔識呢隻字」 - 輸出{"type":"explanation","question":$直接抄寫用家的問題}
            3. 追問關於同一隻字（例如用家問「咁呢個詞語點樣造句」或「呢個字嘅詞性係咩」） - 輸出{"type":"follow_up","question":$直接抄寫用家的問題}
            4. 不是直接要求課業答案，而是其他學習相關問題 - 輸出{"type":"other","reply":$精簡地回答}
            5. 要求替他做功課，或與學習無關的問題 - 輸出{"type":"not_related","reply":$精簡地鼓勵用家先完成課業，表明可以指導他寫字/理解句子}'''
    categorizer = Chat(system_prompt)
    return json.loads(categorizer.send_message(audio_prompt))

def explanation_branch(voice_text):
    print("picture will be taken")
    play_explain_audio("知道，幫緊你睇",slow=False,blocking=False)
    response = requests.get('http://127.0.0.1:3141/capture')
    if response.status_code == 200:
        print("picture will be taken")
        print(response.json())
    try:
        cam_data = camera_queue.get(timeout=15)
        print(cam_data)
        explain_text(cam_data, voice_text)
    except Exception as e:
        print(f"Error: {e}")
        play_explain_audio("我偵測唔到你指住嘅文字，請再試多次",slow=False)

def process_explanation():
    global voice_text, explanation_chat, explanation_timestamp
    voice_done.wait()
    if voice_text:
        category = categorize_instructions(voice_text)
        match category['type']:
            case "write":
                play_explain_audio("收到！開緊AR字帖",slow=False,blocking=False)
                response = requests.post("http://127.0.0.1:8888/projector_on", json={"word":category['word']})
                if response.status_code == 200:
                    print("done")
                    return response.json()
            case "explanation":
                explanation_branch(category['question'])
            case "follow_up":
                if time.perf_counter() - explanation_timestamp >= 45:
                    explanation_chat = None

                if explanation_chat:
                    respond = explanation_chat.send_message(category['question'])
                    play_explain_audio(json.loads(respond)['audio'])
                else:
                    explanation_branch(category['question'])
            case "not_related":
                play_explain_audio(category['reply'])
            case _:
                print(f"Unknown type: {category}")
                play_explain_audio("我唔係好明你講咩，請你試吓再講多次。", slow=False)
    else:
        play_explain_audio("我聽唔清楚你講咩，請你再講多次", slow=False)

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Main program

print("系統準備就緒")
print("FastAPI server starting at http://127.0.0.1:8000")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
play_explain_audio("測試")

while True:
    button_pressed.wait()
    button_pressed.clear()
    threading.Thread(target=process_explanation).start()