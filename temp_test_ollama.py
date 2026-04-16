import ollama
import time

OLLAMA_HOST = 'http://10.129.130.47:11434'
ollama_client = ollama.Client(host=OLLAMA_HOST)

starttime = time.perf_counter()
# Test default connection
response = ollama_client.chat(
    model="qwen3:4b-instruct",
    messages=[{"role": "user", "content": "Hi how r u"}]
)

endtime = time.perf_counter()
print(response['message']['content'])
print(starttime-endtime)