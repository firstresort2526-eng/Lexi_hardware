import dotenv, os
os.environ["HTTP_PROXY"] = "http://38.154.203.95:5863"
os.environ["HTTPS_PROXY"] = "http://38.154.203.95:5863"
from google import genai
import time
dotenv.load_dotenv()
API_key = os.getenv("GEMINI_KEY")

client = genai.Client(api_key=API_key)

start_time = time.perf_counter()
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents='''用家指着的單字：庭
文字出處: 的姿態大相徑庭。
用家語音補充: 呢個字係咩意思
對象：9歲小孩

請運用以上資訊，精簡地解釋詞語在句字中的意思'''
)
end_time = time.perf_counter()
print(response.text)
print(end_time-start_time)