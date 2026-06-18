from google import genai
import dotenv, os
import time
dotenv.load_dotenv()
API_key = os.getenv("GEMINI_KEY")

client = genai.Client(api_key=API_key)

start_time = time.perf_counter()
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="你好呀，可唔可以同我傾下計？你得唔得閒呀？"
)
end_time = time.perf_counter()
print(response.text)
print(end_time-start_time)