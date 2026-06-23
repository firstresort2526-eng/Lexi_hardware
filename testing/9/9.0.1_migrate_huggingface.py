import os
import dotenv
from fireworks import Fireworks, AsyncFireworks
dotenv.load_dotenv()

client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])

system_prompt = '''用家會向你提出問題，請辨別問題種類。只須回答有效JSON，不可含有其他字句。
            1. 如何寫某字（例如「籃球點寫」、「點樣寫學習」） - 輸出{"type":"write","word":$用家在詢問的詞語}
            2. 某字的意思（例如用家問「呢個字點解？」或「咁呢隻字咩意思」或「我唔識呢隻字」 - 輸出{"type":"explanation","question":$直接抄寫用家的問題}
            3. 追問關於同一隻字（例如用家問「咁呢個詞語點樣造句」或「呢個字嘅詞性係咩」） - 輸出{"type":"follow_up","question":$直接抄寫用家的問題}
            4. 不是直接要求課業答案，而是其他學習相關問題 - 輸出{"type":"other","reply":$精簡地回答}
            5. 要求替他做功課，或與學習無關的問題 - 輸出{"type":"not_related","reply":$精簡地鼓勵用家先完成課業，表明可以指導他寫字/理解句子}'''

completion = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input("What's your question?\n>>")}
    ],
    reasoning_effort="none"
)

print(completion.choices[0].message.content)