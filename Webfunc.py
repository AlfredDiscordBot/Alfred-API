import requests
import time
import os
from threading import Thread

#https://alfred-discord-bot.yashvardhan13.repl.co

def check():
    a = requests.get("https://alfred-discord-bot.yashvardhan13.repl.co")
    if a.status_code > 300:
        return False
    return True

def t():
    while True:
        try:
            if not check():
                send_message("Alfred is having a downtime.")
            time.sleep(30)
        except:
            print("Alfred request error")

def self_check():
    while True:
        try:
            time.sleep(30)
            r = requests.get("https://suicide-detector-api-1.yashvardhan13.repl.co/")
            if r.status_code>300:
                os.system("busybox reboot")
        except:
            print("Self Check error")

def send_message(message, color=16742520):
    json={
        'embeds':[
            {
                'title':"Server",
                'description':message,
                'color': color             
            }
        ]
    }
    requests.post( "https://discord.com/api/webhooks/978532333332344862/n47VPtIj1MX7na_EmUn_v7qLWhZ8rAOwDeDIb3RHcsfO05TF8gin_7ZBErboqEDdSvM0", json=json)

if check():
    send_message("Alfred's server is online", color=3066993)
else:
    send_message("Alfred's server is offline")
th = Thread(target = t)
th.start()
th1 = Thread(target = self_check)
th1.start()
