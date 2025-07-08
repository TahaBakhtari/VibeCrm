import json
import os
from datetime import datetime

def load_customers_history():
    """Load customers history from JSON file"""
    if os.path.exists("customers_history.json"):
        try:
            with open("customers_history.json", "r", encoding="utf-8") as file:
                return json.load(file)
        except:
            return {}
    return {}

def display_chat_history():
    """Display all chat histories in a readable format"""
    data = load_customers_history()
    
    if not data:
        print("هیچ تاریخچه چتی وجود ندارد.")
        return
    
    print("=" * 50)
    print("تاریخچه چت‌های مشتریان")
    print("=" * 50)
    
    for customer_id, customer_data in data.items():
        print(f"\n🔹 مشتری: {customer_id}")
        print(f"📅 تاریخ ایجاد: {customer_data.get('created_at', 'نامشخص')}")
        print("-" * 30)
        
        chat_history = customer_data.get('chat_history', [])
        if not chat_history:
            print("هیچ پیامی وجود ندارد.")
            continue
        
        for i, chat in enumerate(chat_history, 1):
            timestamp = chat.get('timestamp', 'نامشخص')
            message_type = chat.get('type', 'نامشخص')
            message = chat.get('message', '')
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime("%Y/%m/%d %H:%M:%S")
            except:
                formatted_time = timestamp
            
            if message_type == 'agent':
                print(f"  {i}. 👨‍💼 ایجنت ({formatted_time}):")
                print(f"     {message}")
            elif message_type == 'customer':
                print(f"  {i}. 👤 مشتری ({formatted_time}):")
                print(f"     {message}")
            else:
                print(f"  {i}. ❓ نامشخص ({formatted_time}):")
                print(f"     {message}")
        
        print("\n" + "=" * 50)

def display_customer_chat(customer_id):
    """Display chat history for a specific customer"""
    data = load_customers_history()
    
    if customer_id not in data:
        print(f"هیچ تاریخچه‌ای برای مشتری '{customer_id}' یافت نشد.")
        return
    
    customer_data = data[customer_id]
    print(f"\n🔹 مشتری: {customer_id}")
    print(f"📅 تاریخ ایجاد: {customer_data.get('created_at', 'نامشخص')}")
    print("-" * 30)
    
    chat_history = customer_data.get('chat_history', [])
    if not chat_history:
        print("هیچ پیامی وجود ندارد.")
        return
    
    for i, chat in enumerate(chat_history, 1):
        timestamp = chat.get('timestamp', 'نامشخص')
        message_type = chat.get('type', 'نامشخص')
        message = chat.get('message', '')
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y/%m/%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        if message_type == 'agent':
            print(f"  {i}. 👨‍💼 ایجنت ({formatted_time}):")
            print(f"     {message}")
        elif message_type == 'customer':
            print(f"  {i}. 👤 مشتری ({formatted_time}):")
            print(f"     {message}")

if __name__ == "__main__":
    print("انتخاب کنید:")
    print("1. مشاهده تمام تاریخچه‌ها")
    print("2. مشاهده تاریخچه یک مشتری خاص")
    
    choice = input("انتخاب شما (1 یا 2): ").strip()
    
    if choice == "1":
        display_chat_history()
    elif choice == "2":
        customer_id = input("شناسه مشتری را وارد کنید: ").strip()
        display_customer_chat(customer_id)
    else:
        print("انتخاب نامعتبر!") 