# Server_url is 127.0.0.1/8000

import threading
import speech_recognition as sr
import ollama
from gtts import gTTS
import pygame
import os 
from fastapi import FastAPI, Request
import uvicorn
import firebase_admin
from firebase_admin import credentials, firestore
import queue
import ast
import requests, json

camera_queue = queue.Queue()
button_pressed = threading.Event()
voice_done = threading.Event()
voice_text = None

OLLAMA_HOST = 'http://172.21.133.47:11434'
ollama_client = ollama.Client(host=OLLAMA_HOST)

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

def explain_text(cam, voice):
    text = f"鏡頭檢測到的詞語: {cam}\n用家語音補充: {voice}\n20字內根據以上資訊解釋，對象是向七歲兒童"
    print(text)
    # Generate response from Ollama
    print("Thinking...")
    response = ollama_client.chat(model="qwen3:4b-instruct", think=False, messages=[
          {'role': 'system', 
             'content': '回答格式：{"word": "詞彙", "meaning": "解釋"} 只輸出JSON，不要其他文字'},
        {'role': 'user',
         'content': f'解釋{text}'}
    ])
    reply = response['message']['content']
    print(f"Lexi 學習助手: {reply}")

    # Text to Speech
    explanation_dict = ast.literal_eval(reply)
    explanation = f"{explanation_dict.get("word")}嘅意思係{explanation_dict.get("meaning")}"
    print(f"解釋{explanation}")
    tts = gTTS(text=explanation, lang='yue', slow=True)
    tts.save("output.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

    # Upload to firebase

    if word_exists(explanation_dict.get("word")):
        ref = construct_ref(["word_explanations","Developer","entries",explanation_dict.get("word")])
        current_counter = read_data(["word_explanations","Developer","entries",explanation_dict.get("word")]).get("counter")
        ref.update({
            "counter": current_counter + 1,
            f"timestamp{current_counter}": firestore.SERVER_TIMESTAMP
        })
    else:
        upload_explanation_data(userid="Developer", word=explanation_dict.get("word"),json={
            "timestamp": firestore.SERVER_TIMESTAMP,
            "word": explanation_dict.get("word"),
            "explanation": explanation_dict.get("meaning"),
            "counter":  1
        })

    # Delete temp file
    if os.path.exists("output.mp3"):
        os.remove("output.mp3")

def process_voice_input():
    global voice_text
    try:
        with mic as source:
            print("Recording...")
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
    cam_data = request_data['words'][-1]
    camera_queue.put(cam_data)
    return {"status": "camera_data_received"}

def process_explanation():
    global voice_text
    voice_done.wait()
    
    #if xie in voice _text, send to ollama then send endpoint to ollama call caspar end point 
    if "寫" in voice_text:
        response = ollama_client.chat(model="qwen3:4b-instruct", think=False, messages=[
          {'role': 'system', 
             'content': '回答格式：{"word": "字"} 只輸出JSON，不要其他文字'},
        {'role': 'user',
         'content': f'用家想寫哪一個詞語？用家說：{voice_text}'}
    ])
        reply = response['message']['content']
        print(reply)

        response = requests.post("http://127.0.0.1:8888/projector_on", json=json.loads(reply))

        if response.status_code == 200:
            print("done")
            print(response.json())

    else:
        response = requests.get('http://127.0.0.1:3141/capture')
        if response.status_code == 200:
            print("picture will be taken")
            print(response.json())

        if voice_text is not None:
            current_voice_text = voice_text 
        else:
            current_voice_text = "成句句子係咩意思"
        
        try:
            cam_data = camera_queue.get(timeout=10)
        except:
            cam_data = "未檢測到詞語"
        
        explain_text(cam_data, current_voice_text)
    

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Main program

print("系統準備就緒")
print("FastAPI server starting at http://127.0.0.1:8000")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

while True:
    button_pressed.wait()
    button_pressed.clear()
    threading.Thread(target=process_explanation).start()