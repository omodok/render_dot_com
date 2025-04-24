#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install flask


# In[2]:


#!pip install line-bot-sdk
#!pip install python-dotenv


# In[3]:


from flask import Flask, request, abort
import os
from dotenv import load_dotenv
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, ReplyMessageRequest
from linebot.v3.messaging import TextMessage, MessagingApi
from linebot.v3.webhooks import MessageEvent
from line03_autoregressive_model import *
#from rag01_hyakunin_cho import *


# In[4]:


load_dotenv()  # .env ファイルを読み込む

YOUR_CHANNEL_SECRET = os.environ.get('YOUR_CHANNEL_SECRET')
YOUR_CHANNEL_ACCESS_TOKEN = os.environ.get('YOUR_CHANNEL_ACCESS_TOKEN')

configuration = Configuration(access_token=YOUR_CHANNEL_ACCESS_TOKEN)
line_bot_api = ApiClient(configuration)
messaging_api = MessagingApi(line_bot_api)  
handler = WebhookHandler(YOUR_CHANNEL_SECRET)


# In[5]:


app = Flask(__name__)

@app.route("/webhook",  methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK', 200

# メッセージイベントをハンドル
@handler.add(MessageEvent)
def handle_message(event):
    response_text = ai_chat_bot(event.message.text)
    messaging_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=response_text)]
        )
    )

# Flask側にヘルスチェックエンドポイントを追加
@app.route('/health')
def health_check():
    return jsonify({"status": "active"}), 200

#app.run(host="localhost", port=5000)
app.run(host="0.0.0.0", port=int(os.environ["PORT"]))


# In[ ]:




