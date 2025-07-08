from pyrogram import Client, filters
from pyrogram.types import Message, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, ReplyKeyboardRemove, KeyboardButton
from run_langgraph import main, ChatHistory, receiver_agent_response
import json
import os
from datetime import datetime
from tg_keys import get_api_keys

user_histories = {}

app = Client(
    name="my_account",
    api_id=get_api_keys()[1],
    api_hash=get_api_keys()[2]
)

def check_sending_data():
    with open("chats.txt", "r", encoding="utf-8") as file:
        data = file.read()
    if data:
        data = data.split("###")
        user_id = data[0]
        message = data[1]
        return user_id, message
    else:
        return False, False

def load_customers_history():
    if os.path.exists("customers_history.json"):
        try:
            with open("customers_history.json", "r", encoding="utf-8") as file:
                return json.load(file)
        except:
            return {}
    return {}

def save_customers_history(data):
    with open("customers_history.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def add_agent_message(customer_id, message):
    data = load_customers_history()
    
    if customer_id not in data:
        data[customer_id] = {
            "chat_history": [],
            "created_at": datetime.now().isoformat()
        }
    
    data[customer_id]["chat_history"].append({
        "type": "agent",
        "message": message,
        "timestamp": datetime.now().isoformat()
    })
    
    save_customers_history(data)

def add_customer_message(customer_id, message):
    data = load_customers_history()
    
    if customer_id not in data:
        data[customer_id] = {
            "chat_history": [],
            "created_at": datetime.now().isoformat()
        }
    
    data[customer_id]["chat_history"].append({
        "type": "customer", 
        "message": message,
        "timestamp": datetime.now().isoformat()
    })
    
    save_customers_history(data)

def format_chat_history_for_ai(customer_id):
    data = load_customers_history()
    
    if customer_id not in data:
        return None
    
    chat_history = data[customer_id]["chat_history"]
    if not chat_history:
        return None
    
    formatted_chat = f"تاریخچه گفتگو با مشتری '{customer_id}':\n\n"
    
    for i, chat in enumerate(chat_history, 1):
        message_type = chat.get('type', 'نامشخص')
        message = chat.get('message', '')
        timestamp = chat.get('timestamp', '')
        
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y/%m/%d %H:%M")
        except:
            formatted_time = timestamp
        
        if message_type == 'agent':
            formatted_chat += f"{i}. دستیار فروش ({formatted_time}): {message}\n"
        elif message_type == 'customer':
            formatted_chat += f"{i}. مشتری ({formatted_time}): {message}\n"
        else:
            formatted_chat += f"{i}. ({formatted_time}): {message}\n"
    
    return formatted_chat

def check_existed_in_json(customer_id):
    data = load_customers_history()
    return customer_id in data


@app.on_message(filters.command("start") & filters.private)
async def start_command(client: Client, message: Message):
    await message.reply_text("سلام چطوری")


@app.on_message(filters.private & ~filters.command("start"))
async def message_handler(client: Client, message: Message):

    sender_id = message.from_user.id
    
    if sender_id == 7883900921:
        if message.text:
            print(" Account owner message : <<", message.text, ">>")
            user_id = message.from_user.id
            if user_id not in user_histories:
                user_histories[user_id] = ChatHistory()
            user_histories[user_id].add_message("user", message.text)
            response = main(message.text, chat_history=user_histories[user_id])
            customer_id, message_to_send = check_sending_data()
            if customer_id == False:
                await message.reply_text(response)
            else:
                print(" Sending message to customer : ", customer_id, "message : ", message_to_send)
                await client.send_message(customer_id, message_to_send)
                open("chats.txt", "w").write("")
                
                add_agent_message(customer_id, message_to_send)
                
                await message.reply_text(response)
    else:
        print(" >> message from customer : ", message.text)
        await message.reply_text("ممنون از پاسخ شما , پیام شما به طاها باختری ارسال شد")
        
        customer_username = message.from_user.username or str(message.from_user.id)
        add_customer_message(customer_username, message.text)
        
        chat_history_str = format_chat_history_for_ai(customer_username)
        
        if chat_history_str:
            if client.me.id not in user_histories:
                user_histories[client.me.id] = ChatHistory()
            
            ai_prompt = f"{chat_history_str}"
            print("  chat_history_str :\n", chat_history_str)
            user_histories[client.me.id].add_message("user", ai_prompt)
            ai_analysis = receiver_agent_response(ai_prompt, chat_history=user_histories[client.me.id])
            try:
                await client.send_message(7883900921, f"{customer_username} جواب داد :\n\n{ai_analysis}")
            except Exception as e:
                print(f"خطا در ارسال تحلیل به خودتان: {e}")

if __name__ == "__main__":
    app.run()