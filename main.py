import threading
import speech_recognition as sr
import ollama
from pynput import keyboard
from gtts import gTTS
import pygame
from playsound import playsound
import os
import datetime as dt
import json  
from fastapi import FastAPI, Request
import uvicorn
import firebase_admin
from firebase_admin import credentials, firestore

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
    return None

def upload_data(path:list,json:dict):
    '''
    Upload the json data to the path, returns document id
    '''
    doc_ref = construct_ref(path)
    document_result = doc_ref.add(json)
    return document_result[1].id

def upload_explanation_data(userid, json: dict) -> str:
    '''
    Upload the json data to the word_explanations entries, returns document id
    '''
    return upload_data(['word_explanations',userid,'entries'],json)

print(upload_explanation_data("Developer",{'timestamp':firestore.SERVER_TIMESTAMP,'word':'成功'})) # Example data

r = sr.Recognizer()
mic = sr.Microphone()

def saveHistory(prompts, response):
    new_entry = {
        "timestamp": dt.datetime.now().isoformat(),
        "prompt": prompts,
        "response": response
    }

    history = []
    if os.path.exists('history.txt'):
        try:
            with open('history.txt', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # If file is not empty
                    history = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            history = []
    
    history.append(new_entry)
    
    with open('history.txt', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        
    print(f"History saved at {new_entry['timestamp']}")
    
def explain_text(text):
    # Generate response from Ollama
    print("Thinking...")
    response = ollama.chat(model="qwen2.5:1.5b-instruct", messages=[
          {'role': 'system', 
             'content': '用粵語口語在四十字解釋以下詞語。如果有多個詞語，請分別簡短解釋每個詞語和整個句子的意思'},
        {'role': 'user',
         'content': f'解釋{text}'}
    ])
    reply = response['message']['content']
    print(f"Lexi 學習助手: {reply}")

    # Save history
    saveHistory(text, reply)

    # Text to Speech
    tts = gTTS(text=reply, lang='yue', slow=False)
    tts.save("output.mp3")
    playsound("output.mp3")

    # Delete temp file
    if os.path.exists("output.mp3"):
        os.remove("output.mp3")

app = FastAPI()

@app.post("/explain")
async def process_camera_input(request: Request):
    request_data = await request.json()
    text = request_data["words"][-1]["text"]
    print(f"讀取鏡頭數據：{text}")
    
    # Run explain_text in a separate thread to avoid blocking
    thread = threading.Thread(target=explain_text, args=(text,))
    thread.start()
    
    return {"status": "processing", "text": text}

def process_voice_input():
    try:
        with mic as source:
            print("Recording...")
            audio = r.listen(source, timeout=10, phrase_time_limit=20) 
            text = r.recognize_google(audio, language="yue-Hant-hk")
        print(f"用家: {text}")
        explain_text(text)
    except sr.UnknownValueError:
        tts = gTTS(text="唔好意思，Lexi聽唔清楚你的查詢", lang='yue', slow=False)
        tts.save("output.mp3")
        playsound("output.mp3")
    except Exception as e:
        print(f"Error in voice processing: {e}")

def on_press(key):
    try:
        if hasattr(key, 'char') and key.char == 's':  
            print("Voice input triggered...")
            threading.Thread(target=process_voice_input).start()
    except AttributeError:
        pass

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Main program

print("系統準備就緒")
print("按 's' - 用語音輸入查詢字義")
print("FastAPI server starting at http://0.0.0.0:8000")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()