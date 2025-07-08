import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import sqlite3
from contextlib import contextmanager
from typing import Annotated

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

class ChatHistory:
    def __init__(self):
        self.messages = []
        
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        
    def get_messages(self):
        return self.messages.copy()
        
    def clear(self):
        self.messages = []

chat_history = ChatHistory()

os.environ["OPENAI_API_KEY"] = "x"

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DB_FILE = DATA_DIR / "crm.db"

customer_index = None
customer_data = []
plan_index = None
plan_data = []
sale_index = None
sale_data = []
product_index = None
product_data = []

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    try:
        yield conn
    finally:
        conn.close()

def initialize_database():
    """Initialize the SQLite database schema"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            address TEXT,
            extra_info TEXT,
            created_at TEXT NOT NULL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            with_who TEXT,
            subject TEXT,
            plan_time TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT NOT NULL,
            price TEXT,
            customer TEXT,
            sale_date TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            price TEXT,
            quantity TEXT,
            description TEXT,
            created_at TEXT NOT NULL
        )
        ''')
        
        conn.commit()

initialize_database()

def normalize_text(text):
    """No-op normalization since hazm is removed."""
    return text

def get_all_data(table_name):
    """Get all data from a specified table"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        return [dict(row) for row in cursor.fetchall()]

def initialize_vector_indexes():
    """Initialize FAISS vector indexes for semantic search."""
    global customer_index, customer_data, plan_index, plan_data, sale_index, sale_data
    global product_index, product_data
    
    try:
        customer_data = get_all_data('customers')
        
        if customer_data:
            customer_texts = []
            
            for customer in customer_data:
                text = f"{customer['name']} {customer['phone']} {customer['email']} {customer['address']} {customer['extra_info']}"
                customer_texts.append(normalize_text(text))
            
            customer_embeddings = model.encode(customer_texts)
            vector_dimension = customer_embeddings.shape[1]
            customer_index = faiss.IndexFlatL2(vector_dimension)
            customer_index.add(np.array(customer_embeddings).astype('float32'))
    
        plan_data = get_all_data('plans')
        
        if plan_data:
            plan_texts = []
            
            for plan in plan_data:
                text = f"{plan['title']} {plan['with_who']} {plan['subject']} {plan['plan_time']} {plan['notes']}"
                plan_texts.append(normalize_text(text))
            
            plan_embeddings = model.encode(plan_texts)
            vector_dimension = plan_embeddings.shape[1]
            plan_index = faiss.IndexFlatL2(vector_dimension)
            plan_index.add(np.array(plan_embeddings).astype('float32'))
    
        sale_data = get_all_data('sales')
        
        if sale_data:
            sale_texts = []
            
            for sale in sale_data:
                text = f"{sale['item']} {sale['price']} {sale['customer']} {sale['sale_date']} {sale['notes']}"
                sale_texts.append(normalize_text(text))
            
            sale_embeddings = model.encode(sale_texts)
            vector_dimension = sale_embeddings.shape[1]
            sale_index = faiss.IndexFlatL2(vector_dimension)
            sale_index.add(np.array(sale_embeddings).astype('float32'))
    
        product_data = get_all_data('products')
        
        if product_data:
            product_texts = []
            
            for product in product_data:
                text = f"{product['name']} {product['category']} {product['price']} {product['quantity']} {product['description']}"
                product_texts.append(normalize_text(text))
            
            product_embeddings = model.encode(product_texts)
            vector_dimension = product_embeddings.shape[1]
            product_index = faiss.IndexFlatL2(vector_dimension)
            product_index.add(np.array(product_embeddings).astype('float32'))
    
        return "Vector indexes initialized successfully"
    except Exception as e:
        return f"Error initializing vector indexes: {str(e)}"

initialize_vector_indexes()

real_now = datetime.now()
real_date = real_now.strftime("%Y-%m-%d")

reference_date = datetime(2025, 5, 20)
current_date = reference_date.strftime("%Y-%m-%d")
current_time = real_now.strftime("%H:%M:%S")
current_weekday = reference_date.strftime("%A")

system_prompt = f"""
You help the business owner manage customers, plans, sales, and products. Replies must be short and do exactly what is asked, talk so friendly. User is your admin.
and his name is Taha (طاها باختری)
Always use all tools and functions to get the best result, even if not requested.
You can:
- Add, find, update, or remove customers
- Make, find, update, delete, or list plans/reminders
- Add, find, update, delete, or list sales
- Add, find, update, delete, or list products
- Generate detailed sales reports and analytics
- Send messages to customers

CHAT HISTORY ANALYSIS:
- Sometimes you'll receive chat histories that include messages you previously sent as an agent
- Provide short insights on customer status, needs, concerns, and suggest next actions
- Remember that "ایجنت" messages in chat histories are your own previous responses

Reporting capabilities:
- Total sales for any time period (today, this week, this month, this year, or custom dates)
- Sales breakdowns by customer or product
- Sales trend analysis (daily, weekly, or monthly views)
- Top performers (best customers and best-selling products)
- Filter reports by customer, product, or date ranges

IMPORTANT ABOUT DATES:
- This is a test environment with sample data from May 2025
- The simulated current date is {current_date}
- When user asks about "today", compare with the simulated date {current_date}
- Do NOT match today with real-world current date unless specifically requested
- Although sample data is from May 26-30, 2025, you can add, update, or delete sales for ANY date the user specifies
- Allow sales to be recorded for any date, including past or future dates, regardless of the language used (English, Persian, etc.)

You can search by anything, not just names.
IMPORTANT:
- If a name is mentioned, first check if they're a customer and use their info if found.
- Only create new customers when explicitly asked to do so by the user.
- If info (like phone/email) is needed, search for it yourself.
- Use as many functions as needed to fully complete the request.        
- If info can't be found, politely inform the user that it doesn't exist.
- For update requests, get the current value first and then update with the new value.
- For reporting requests, determine the appropriate time period and report type.
- When asked for top performers, items, or customers, use the top_performers_report function.

Current simulated date: {current_date}
Current simulated weekday: {current_weekday}

allways answer short and concise. allways answer in persian. and friendly.
"""

@tool
def save_new_customer(name: str = "", phone: str = "", email: str = "", address: str = "", extra_info: str = "") -> str:
    """Save a new customer or update an existing one.
    
    Args:
        name: Customer name
        phone: Customer phone number
        email: Customer email address
        address: Customer address
        extra_info: Additional customer information
    """
    print(" >> save new customer used")
    created_at = datetime.now().isoformat()
    
    try:
        global customer_data
        updated = False
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM customers WHERE LOWER(name) = LOWER(?)",
                (name.strip(),)
            )
            existing_customer = cursor.fetchone()
            
            if existing_customer:
                cursor.execute(
                    """
                    UPDATE customers 
                    SET phone = ?, email = ?, address = ?, extra_info = ?
                    WHERE id = ?
                    """,
                    (
                        phone or existing_customer['phone'],
                        email or existing_customer['email'],
                        address or existing_customer['address'],
                        extra_info or existing_customer['extra_info'],
                        existing_customer['id']
                    )
                )
                updated = True
            else:
                cursor.execute(
                    """
                    INSERT INTO customers (name, phone, email, address, extra_info, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, phone, email, address, extra_info, created_at)
                )
            
            conn.commit()
        
        customer_data = get_all_data('customers')
        initialize_vector_indexes()
        
        if updated:
            return f"Customer '{name}' updated successfully."
        else:
            return f"Successfully saved customer: {name}"
    except Exception as e:
        return f"Error saving customer: {str(e)}"

@tool
def search_customers(query: str = "", dimension: str = "") -> str:
    """Search for customers by query and dimension.
    
    Args:
        query: Search query
        dimension: Field to search in (name, phone, email, address, extra_info)
    """
    print(" >> search customers used")
    if not query:
        return "Please provide a search query."
    
    valid_dimensions = ['name', 'phone', 'email', 'address', 'extra_info']
    if dimension and dimension not in valid_dimensions:
        return f"Please provide a valid dimension to search in: {', '.join(valid_dimensions)}."
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            results = []
            if not dimension:
                sql_query = """
                SELECT * FROM customers 
                WHERE name LIKE ? OR phone LIKE ? OR email LIKE ? OR address LIKE ? OR extra_info LIKE ?
                """
                search_term = f"%{query}%"
                cursor.execute(sql_query, (search_term, search_term, search_term, search_term, search_term))
            else:
                sql_query = f"SELECT * FROM customers WHERE {dimension} LIKE ?"
                cursor.execute(sql_query, (f"%{query}%",))
            
            results = cursor.fetchall()
            
            if not results:
                if dimension:
                    return f"No customers found in '{dimension}' matching '{query}'. Use save_new_customer to create this customer if needed."
                else:
                    return f"No customers found matching '{query}'. Use save_new_customer to create this customer if needed."
            
            formatted_results = []
            for i, customer in enumerate(results, 1):
                customer_dict = dict(customer)
                customer_info = [f"{i}. {customer_dict['name']}:"]
                customer_info.append(f"   Phone: {customer_dict['phone']}")
                customer_info.append(f"   Email: {customer_dict['email']}")
                customer_info.append(f"   Address: {customer_dict['address']}")
                if customer_dict['extra_info']:
                    customer_info.append(f"   Notes: {customer_dict['extra_info']}")
                formatted_results.append("\n".join(customer_info))
            
            return f"Found {len(results)} matching customers:\n\n" + "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def remove_customer(search_term: str = "") -> str:
    """Remove a customer from the database.
    
    Args:
        search_term: Search term to identify the customer to remove
    """
    print(" >> remove customer used")   
    if not search_term:
        return "Please provide a search term to identify the customer to remove."
    
    try:
        global customer_data
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            search_term_like = f"%{search_term}%"
            cursor.execute(
                """
                SELECT * FROM customers 
                WHERE name LIKE ? OR phone LIKE ? OR email LIKE ? OR address LIKE ? OR extra_info LIKE ?
                """,
                (search_term_like, search_term_like, search_term_like, search_term_like, search_term_like)
            )
            
            matches = cursor.fetchall()
            
            if not matches:
                return f"No customer found matching: {search_term}"
            
            if len(matches) > 1:
                result = f"Found {len(matches)} customers matching '{search_term}'. Please provide more specific details:\n\n"
                for i, customer in enumerate(matches, 1):
                    result += f"{i}. {customer['name']} - {customer['phone']} - {customer['email']}\n"
                return result
            
            customer_to_remove = dict(matches[0])
            cursor.execute("DELETE FROM customers WHERE id = ?", (customer_to_remove['id'],))
            conn.commit()
        
        customer_data = get_all_data('customers')
        initialize_vector_indexes()
        
        return f"Successfully removed customer: {customer_to_remove['name']}"
        
    except Exception as e:
        return f"Error removing customer: {str(e)}"

@tool
def create_plan(title: str = "Meeting", with_who: str = "", subject: str = "", plan_time: str = "", notes: str = "") -> str:
    """Add or update a plan in the database.
    
    Args:
        title: Plan title
        with_who: Who the plan is with
        subject: Plan subject
        plan_time: When the plan is scheduled
        notes: Additional notes
    """
    print(" >> create plan used")
    created_at = datetime.now().isoformat()
    
    try:
        global plan_data
        updated = False
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT * FROM plans 
                WHERE LOWER(title) = LOWER(?) AND LOWER(with_who) = LOWER(?) AND plan_time = ?
                """,
                (title.strip(), with_who.strip(), plan_time.strip())
            )
            
            existing_plan = cursor.fetchone()
            
            if existing_plan:
                cursor.execute(
                    """
                    UPDATE plans 
                    SET subject = ?, notes = ?
                    WHERE id = ?
                    """,
                    (
                        subject or existing_plan['subject'],
                        notes or existing_plan['notes'],
                        existing_plan['id']
                    )
                )
                updated = True
            else:
                cursor.execute(
                    """
                    INSERT INTO plans (title, with_who, subject, plan_time, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (title, with_who, subject, plan_time, notes, created_at)
                )
            
            conn.commit()
        
        plan_data = get_all_data('plans')
        initialize_vector_indexes()
        
        if updated:
            return f"Plan '{title}' with {with_who} at {plan_time} updated successfully."
        else:
            return f"Successfully saved plan: {title} with {with_who} at {plan_time}"
    except Exception as e:
        return f"Error saving plan: {str(e)}"

@tool
def search_plans(query: str = "", dimension: str = "") -> str:
    """Search for plans by a query and a specific dimension.
    
    Args:
        query: Search query
        dimension: Field to search in (title, with_who, subject, plan_time, notes)
    """
    print(" >> search plans used")
    if not query:
        return "Please provide a search query."
    
    valid_fields = ['title', 'with_who', 'subject', 'plan_time', 'notes']
    if not dimension or dimension not in valid_fields:
        return f"Please provide a valid dimension to search in: {', '.join(valid_fields)}."
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            sql_query = f"SELECT * FROM plans WHERE {dimension} LIKE ?"
            cursor.execute(sql_query, (f"%{query}%",))
            
            results = cursor.fetchall()
            
            if not results:
                return f"No plans found in '{dimension}' matching '{query}'"
            
            formatted_results = []
            for i, plan in enumerate(results, 1):
                plan_dict = dict(plan)
                plan_info = [f"{i}. {plan_dict['title']}:"]
                plan_info.append(f"   With: {plan_dict['with_who']}")
                if plan_dict['subject']:
                    plan_info.append(f"   Subject: {plan_dict['subject']}")
                plan_info.append(f"   When: {plan_dict['plan_time']}")
                if plan_dict['notes']:
                    plan_info.append(f"   Notes: {plan_dict['notes']}")
                formatted_results.append("\n".join(plan_info))
            
            return f"Found {len(results)} matching plans in '{dimension}':\n\n" + "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def see_all_plans() -> str:
    """Retrieve and list all saved plans/meetings."""
    print(" >> see all plans used")
    try:
        plans = get_all_data('plans')
        if not plans:
            return "No plans or meetings found."
        
        sorted_plans = sorted(plans, key=lambda x: x.get('plan_time', ''))
        
        formatted_results = []
        for i, plan in enumerate(sorted_plans, 1):
            plan_info = [f"{i}. {plan['title']}:"]
            plan_info.append(f"   With: {plan['with_who']}")
            if plan['subject']:
                plan_info.append(f"   Subject: {plan['subject']}")
            plan_info.append(f"   When: {plan['plan_time']}")
            if plan['notes']:
                plan_info.append(f"   Notes: {plan['notes']}")
            formatted_results.append("\n".join(plan_info))
        
        return "All saved plans/meetings:\n\n" + "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error retrieving plans: {str(e)}"

@tool
def delete_plan(search_term: str = "") -> str:
    """Remove a plan from the database.
    
    Args:
        search_term: Search term to identify the plan to remove
    """
    print(" >> delete plan used")
    if not search_term:
        return "Please provide a search term to identify the plan to remove."
    
    try:
        global plan_data
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            search_term_like = f"%{search_term}%"
            cursor.execute(
                """
                SELECT * FROM plans 
                WHERE title LIKE ? OR with_who LIKE ? OR subject LIKE ? OR plan_time LIKE ? OR notes LIKE ?
                """,
                (search_term_like, search_term_like, search_term_like, search_term_like, search_term_like)
            )
            
            matches = cursor.fetchall()
            
            if not matches:
                return f"No plan found matching: {search_term}"
            
            if len(matches) > 1:
                result = f"Found {len(matches)} plans matching '{search_term}'. Please provide more specific details:\n\n"
                for i, plan in enumerate(matches, 1):
                    result += f"{i}. {plan['title']} with {plan['with_who']} at {plan['plan_time']}\n"
                return result
            
            plan_to_remove = dict(matches[0])
            cursor.execute("DELETE FROM plans WHERE id = ?", (plan_to_remove['id'],))
            conn.commit()
        
        plan_data = get_all_data('plans')
        initialize_vector_indexes()
        
        return f"Successfully removed plan: {plan_to_remove['title']} with {plan_to_remove['with_who']} at {plan_to_remove['plan_time']}"
        
    except Exception as e:
        return f"Error removing plan: {str(e)}"

@tool
def list_of_customers() -> str:
    """Retrieve and list all saved customers."""
    print(" >> list of customers used")
    lines = []
    customers = get_all_data('customers')
    if not customers:
        return "No customers found."
    
    for c in customers:
        details = (
            f"Name: {c.get('name','')}, "
            f"Phone: {c.get('phone','')}, "
            f"Email: {c.get('email','')}, "
            f"Address: {c.get('address','')}, "
            f"Extra Info: {c.get('extra_info','')}, "
            f"Created At: {c.get('created_at','')}"
        )
        lines.append(details)
    
        return "\n".join(lines) 

@tool
def save_the_sales(item: str = "", price: str = "", customer: str = "", sale_date: str = "", notes: str = "") -> str:
    """Saves a sale to the database.
    
    Args:
        item: Item sold
        price: Sale price
        customer: Customer who bought the item
        sale_date: Date of sale
        notes: Additional notes
    """
    print(" >> save the sales used")
    created_at = datetime.now().isoformat()
    
    if not customer:
        customer = "Unknown"
    
    if not sale_date:
        sale_date = current_date
    
    try:
        global sale_data
        updated = False

        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT * FROM sales 
                WHERE LOWER(item) = LOWER(?) AND LOWER(customer) = LOWER(?) AND sale_date = ?
                """,
                (item.strip(), customer.strip(), sale_date.strip())
            )
            
            existing_sale = cursor.fetchone()
            
            if existing_sale:
                cursor.execute(
                    """
                    UPDATE sales 
                    SET price = ?, notes = ?
                    WHERE id = ?
                    """,
                    (
                        price or existing_sale['price'],
                        notes or existing_sale['notes'],
                        existing_sale['id']
                    )
                )
                updated = True
            else:
                cursor.execute(
                    """
                    INSERT INTO sales (item, price, customer, sale_date, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (item, price, customer, sale_date, notes, created_at)
                )
            
            conn.commit()
        
        sale_data = get_all_data('sales')
        initialize_vector_indexes()

        if updated:
            return f"Sale for '{item}' to '{customer}' on '{sale_date}' updated successfully."
        else:
            return f"Successfully saved sale: {item} to {customer} on {sale_date}"
    except Exception as e:
        return f"Error saving sale: {str(e)}"

@tool
def search_sales(query: str = "", dimension: str = "") -> str:
    """Search for sales by query and dimension.
    
    Args:
        query: Search query
        dimension: Field to search in (item, price, customer, sale_date, notes)
    """
    print(" >> search sales used")
    if not query:
        return "Please provide a search query."
    
    valid_dimensions = ['item', 'price', 'customer', 'sale_date', 'notes']
    if not dimension or dimension not in valid_dimensions:
        return f"Please provide a valid dimension to search in: {', '.join(valid_dimensions)}."
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            sql_query = f"SELECT * FROM sales WHERE {dimension} LIKE ?"
            cursor.execute(sql_query, (f"%{query}%",))
            
            results = cursor.fetchall()
            
            if not results:
                return f"No sales found in '{dimension}' matching '{query}'"
            
            formatted_results = []
            for i, sale in enumerate(results, 1):
                sale_dict = dict(sale)
                sale_info = [f"{i}. {sale_dict['item']}:"]
                sale_info.append(f"   Price: {sale_dict['price']}")
                sale_info.append(f"   Customer: {sale_dict['customer']}")
                sale_info.append(f"   Date: {sale_dict['sale_date']}")
                if sale_dict['notes']:
                    sale_info.append(f"   Notes: {sale_dict['notes']}")
                formatted_results.append("\n".join(sale_info))
            
            return f"Found {len(results)} matching sales in '{dimension}':\n\n" + "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def remove_the_sales(search_term: str = "") -> str:
    """Remove a sale from the database.
    
    Args:
        search_term: Search term to identify the sale to remove
    """
    print(" >> remove the sales used")
    if not search_term:
        return "Please provide a search term to identify the sale to remove."
    
    try:
        global sale_data

        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            search_term_like = f"%{search_term}%"
            cursor.execute(
                """
                SELECT * FROM sales 
                WHERE item LIKE ? OR price LIKE ? OR customer LIKE ? OR sale_date LIKE ? OR notes LIKE ?
                """,
                (search_term_like, search_term_like, search_term_like, search_term_like, search_term_like)
            )
            
            matches = cursor.fetchall()
            
            if not matches:
                return f"No sale found matching: {search_term}"
            
            if len(matches) > 1:
                result = f"Found {len(matches)} sales matching '{search_term}'. Please provide more specific details:\n\n"
                for i, sale in enumerate(matches, 1):
                    result += f"{i}. {sale['item']} - {sale['customer']} - {sale['sale_date']} - {sale['price']}\n"
                return result
            
            sale_to_remove = dict(matches[0])
            cursor.execute("DELETE FROM sales WHERE id = ?", (sale_to_remove['id'],))
            conn.commit()

        sale_data = get_all_data('sales')
        initialize_vector_indexes()

        return f"Successfully removed sale: {sale_to_remove['item']} to {sale_to_remove['customer']} on {sale_to_remove['sale_date']}"
    
    except Exception as e:
        return f"Error removing sale: {str(e)}"

@tool
def save_product(name: str = "", category: str = "", price: str = "", quantity: str = "", description: str = "") -> str:
    """Saves a product to the database.
    
    Args:
        name: Product name
        category: Product category
        price: Product price
        quantity: Product quantity
        description: Product description
    """
    print(" >> save product used")
    created_at = datetime.now().isoformat()
    
    try:
        global product_data
        updated = False

        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT * FROM products 
                WHERE LOWER(name) = LOWER(?) AND LOWER(category) = LOWER(?)
                """,
                (name.strip(), category.strip())
            )
            
            existing_product = cursor.fetchone()
            
            if existing_product:
                cursor.execute(
                    """
                    UPDATE products 
                    SET price = ?, quantity = ?, description = ?
                    WHERE id = ?
                    """,
                    (
                        price or existing_product['price'],
                        quantity or existing_product['quantity'],
                        description or existing_product['description'],
                        existing_product['id']
                    )
                )
                updated = True
            else:
                cursor.execute(
                    """
                    INSERT INTO products (name, category, price, quantity, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, category, price, quantity, description, created_at)
                )
            
            conn.commit()
        
        product_data = get_all_data('products')
        initialize_vector_indexes()

        if updated:
            return f"Product '{name}' in category '{category}' updated successfully."
        else:
            return f"Successfully saved product: {name} in category {category}"
    except Exception as e:
        return f"Error saving product: {str(e)}"

@tool
def search_products(query: str = "", dimension: str = "") -> str:
    """Search for products by query and dimension.
    
    Args:
        query: Search query
        dimension: Field to search in (name, category, price, quantity, description)
    """
    print(" >> search products used")
    if not query:
        return "Please provide a search query."
    
    valid_dimensions = ['name', 'category', 'price', 'quantity', 'description']
    if not dimension or dimension not in valid_dimensions:
        return f"Please provide a valid dimension to search in: {', '.join(valid_dimensions)}."
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            sql_query = f"SELECT * FROM products WHERE {dimension} LIKE ?"
            cursor.execute(sql_query, (f"%{query}%",))
            
            results = cursor.fetchall()
            
            if not results:
                return f"No products found in '{dimension}' matching '{query}'"
            
            formatted_results = []
            for i, product in enumerate(results, 1):
                product_dict = dict(product)
                product_info = [f"{i}. {product_dict['name']}:"]
                product_info.append(f"   Category: {product_dict['category']}")
                product_info.append(f"   Price: {product_dict['price']}")
                product_info.append(f"   Quantity: {product_dict['quantity']}")
                if product_dict['description']:
                    product_info.append(f"   Description: {product_dict['description']}")
                formatted_results.append("\n".join(product_info))
            
            return f"Found {len(results)} matching products in '{dimension}':\n\n" + "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def remove_product(search_term: str = "") -> str:
    """Remove a product from the database.
    
    Args:
        search_term: Search term to identify the product to remove
    """
    print(" >> remove product used")
    if not search_term:
        return "Please provide a search term to identify the product to remove."
    
    try:
        global product_data
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Search for matching products
            search_term_like = f"%{search_term}%"
            cursor.execute(
                """
                SELECT * FROM products 
                WHERE name LIKE ? OR category LIKE ? OR price LIKE ? OR quantity LIKE ? OR description LIKE ?
                """,
                (search_term_like, search_term_like, search_term_like, search_term_like, search_term_like)
            )
            
            matches = cursor.fetchall()
            
            if not matches:
                return f"No product found matching: {search_term}"
            
            # If multiple matches, provide details and ask for clarification
            if len(matches) > 1:
                result = f"Found {len(matches)} products matching '{search_term}'. Please provide more specific details:\n\n"
                for i, product in enumerate(matches, 1):
                    result += f"{i}. {product['name']} - {product['category']} - {product['price']}\n"
                return result
            
            # Remove the product
            product_to_remove = dict(matches[0])
            cursor.execute("DELETE FROM products WHERE id = ?", (product_to_remove['id'],))
            conn.commit()
        
        # Reload product data and reinitialize vector indexes
        product_data = get_all_data('products')
        initialize_vector_indexes()
        
        return f"Successfully removed product: {product_to_remove['name']} in category {product_to_remove['category']}"
    
    except Exception as e:
        return f"Error removing product: {str(e)}"

@tool
def list_of_products() -> str:
    """Returns a list of all products."""
    print(" >> list of products used")
    products = get_all_data('products')
    if not products:
        return "No products found."
    
    lines = []
    for p in products:
        details = (
            f"Name: {p.get('name','')}, "
            f"Category: {p.get('category','')}, "
            f"Price: {p.get('price','')}, "
            f"Quantity: {p.get('quantity','')}, "
            f"Description: {p.get('description','')}"
        )
        lines.append(details)
    
    return "\n".join(lines)

@tool
def send_message_to_customer(user_id: str, message: str) -> str:
    """Send a message to a customer.
    
    Args:
        user_id: The user's Telegram ID
        message: The message to send
    """
    print(" >> send message to customer used")
    data = f"{user_id}###{message}"
    with open("chats.txt", "a", encoding="utf-8") as file:
        file.write(data)
    print(" >> message saved for customer : " , data)
    return "Sent message to customer"

@tool
def update_customer(search_term: str = "", field_to_update: str = "", new_value: str = "") -> str:
    """Update a specific field for an existing customer.
    
    Args:
        search_term: Search term to identify the customer to update
        field_to_update: Field to update (name, phone, email, address, extra_info)
        new_value: New value for the field
    """
    print(" >> update customer used")
    if not search_term:
        return "Please provide a search term to identify the customer to update."
    
    if not field_to_update:
        return "Please specify which field to update (name, phone, email, address, extra_info)."
        
    if not new_value:
        return "Please provide a new value for the specified field."
    
    valid_fields = ['name', 'phone', 'email', 'address', 'extra_info']
    if field_to_update not in valid_fields:
        return f"Invalid field. Please choose from: {', '.join(valid_fields)}"
    
    try:
        global customer_data
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Search for matching customers
            search_term_like = f"%{search_term}%"
            cursor.execute(
                """
                SELECT * FROM customers 
                WHERE name LIKE ? OR phone LIKE ? OR email LIKE ? OR address LIKE ? OR extra_info LIKE ?
                """,
                (search_term_like, search_term_like, search_term_like, search_term_like, search_term_like)
            )
            
            matches = cursor.fetchall()
            
            if not matches:
                return f"No customer found matching: {search_term}"
            
            # If multiple matches, provide details and ask for clarification
            if len(matches) > 1:
                result = f"Found {len(matches)} customers matching '{search_term}'. Please provide more specific details:\n\n"
                for i, customer in enumerate(matches, 1):
                    result += f"{i}. {customer['name']} - {customer['phone']} - {customer['email']}\n"
                return result
            
            # Update the customer
            customer_to_update = dict(matches[0])
            old_value = customer_to_update[field_to_update]
            
            # Update the specific field
            update_query = f"UPDATE customers SET {field_to_update} = ? WHERE id = ?"
            cursor.execute(update_query, (new_value, customer_to_update['id']))
            conn.commit()
        
        # Reload customer data and reinitialize vector indexes
        customer_data = get_all_data('customers')
        initialize_vector_indexes()
        
        return f"Successfully updated customer {customer_to_update['name']}'s {field_to_update} from '{old_value}' to '{new_value}'"
        
    except Exception as e:
        return f"Error updating customer: {str(e)}"

@tool
def update_plan(search_term: str = "", field_to_update: str = "", new_value: str = "") -> str:
    """Update a specific field for an existing plan.
    
    Args:
        search_term: Search term to identify the plan to update
        field_to_update: Field to update (title, with_who, subject, plan_time, notes)
        new_value: New value for the field
    """
    print(" >> update plan used")
    if not search_term:
        return "Please provide a search term to identify the plan to update."
    
    if not field_to_update:
        return "Please specify which field to update (title, with_who, subject, plan_time, notes)."
        
    if not new_value:
        return "Please provide a new value for the specified field."
    
    valid_fields = ['title', 'with_who', 'subject', 'plan_time', 'notes']
    if field_to_update not in valid_fields:
        return f"Invalid field. Please choose from: {', '.join(valid_fields)}"
    
    try:
        global plan_data
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Search for matching plans
            search_term_like = f"%{search_term}%"
            cursor.execute(
                """
                SELECT * FROM plans 
                WHERE title LIKE ? OR with_who LIKE ? OR subject LIKE ? OR plan_time LIKE ? OR notes LIKE ?
                """,
                (search_term_like, search_term_like, search_term_like, search_term_like, search_term_like)
            )
            
            matches = cursor.fetchall()
            
            if not matches:
                return f"No plan found matching: {search_term}"
            
            # If multiple matches, provide details and ask for clarification
            if len(matches) > 1:
                result = f"Found {len(matches)} plans matching '{search_term}'. Please provide more specific details:\n\n"
                for i, plan in enumerate(matches, 1):
                    result += f"{i}. {plan['title']} with {plan['with_who']} at {plan['plan_time']}\n"
                return result
            
            # Update the plan
            plan_to_update = dict(matches[0])
            old_value = plan_to_update[field_to_update]
            
            # Update the specific field
            update_query = f"UPDATE plans SET {field_to_update} = ? WHERE id = ?"
            cursor.execute(update_query, (new_value, plan_to_update['id']))
            conn.commit()
        
        # Reload plan data and reinitialize vector indexes
        plan_data = get_all_data('plans')
        initialize_vector_indexes()
        
        return f"Successfully updated plan '{plan_to_update['title']}' {field_to_update} from '{old_value}' to '{new_value}'"
        
    except Exception as e:
        return f"Error updating plan: {str(e)}"

@tool
def update_sale(search_term: str = "", field_to_update: str = "", new_value: str = "") -> str:
    """Update a specific field for an existing sale.
    
    Args:
        search_term: Search term to identify the sale to update
        field_to_update: Field to update (item, price, customer, sale_date, notes)
        new_value: New value for the field
    """
    print(" >> update sale used")
    if not search_term:
        return "Please provide a search term to identify the sale to update."
    
    if not field_to_update:
        return "Please specify which field to update (item, price, customer, sale_date, notes)."
        
    if not new_value:
        return "Please provide a new value for the specified field."
    
    valid_fields = ['item', 'price', 'customer', 'sale_date', 'notes']
    if field_to_update not in valid_fields:
        return f"Invalid field. Please choose from: {', '.join(valid_fields)}"
    
    try:
        global sale_data
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Search for matching sales
            search_term_like = f"%{search_term}%"
            cursor.execute(
                """
                SELECT * FROM sales 
                WHERE item LIKE ? OR price LIKE ? OR customer LIKE ? OR sale_date LIKE ? OR notes LIKE ?
                """,
                (search_term_like, search_term_like, search_term_like, search_term_like, search_term_like)
            )
            
            matches = cursor.fetchall()
            
            if not matches:
                return f"No sale found matching: {search_term}"
            
            # If multiple matches, provide details and ask for clarification
            if len(matches) > 1:
                result = f"Found {len(matches)} sales matching '{search_term}'. Please provide more specific details:\n\n"
                for i, sale in enumerate(matches, 1):
                    result += f"{i}. {sale['item']} - {sale['customer']} - {sale['sale_date']} - {sale['price']}\n"
                return result
            
            # Update the sale
            sale_to_update = dict(matches[0])
            old_value = sale_to_update[field_to_update]
            
            # Update the specific field
            update_query = f"UPDATE sales SET {field_to_update} = ? WHERE id = ?"
            cursor.execute(update_query, (new_value, sale_to_update['id']))
            conn.commit()
        
        # Reload sale data and reinitialize vector indexes
        sale_data = get_all_data('sales')
        initialize_vector_indexes()
        
        return f"Successfully updated sale of '{sale_to_update['item']}' {field_to_update} from '{old_value}' to '{new_value}'"
        
    except Exception as e:
        return f"Error updating sale: {str(e)}"

@tool
def update_product(search_term: str = "", field_to_update: str = "", new_value: str = "") -> str:
    """Update a specific field for an existing product.
    
    Args:
        search_term: Search term to identify the product to update
        field_to_update: Field to update (name, category, price, quantity, description)
        new_value: New value for the field
    """
    print(" >> update product used")
    if not search_term:
        return "Please provide a search term to identify the product to update."
    
    if not field_to_update:
        return "Please specify which field to update (name, category, price, quantity, description)."
        
    if not new_value:
        return "Please provide a new value for the specified field."
    
    valid_fields = ['name', 'category', 'price', 'quantity', 'description']
    if field_to_update not in valid_fields:
        return f"Invalid field. Please choose from: {', '.join(valid_fields)}"
    
    try:
        global product_data
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Search for matching products
            search_term_like = f"%{search_term}%"
            cursor.execute(
                """
                SELECT * FROM products 
                WHERE name LIKE ? OR category LIKE ? OR price LIKE ? OR quantity LIKE ? OR description LIKE ?
                """,
                (search_term_like, search_term_like, search_term_like, search_term_like, search_term_like)
            )
            
            matches = cursor.fetchall()
            
            if not matches:
                return f"No product found matching: {search_term}"
            
            # If multiple matches, provide details and ask for clarification
            if len(matches) > 1:
                result = f"Found {len(matches)} products matching '{search_term}'. Please provide more specific details:\n\n"
                for i, product in enumerate(matches, 1):
                    result += f"{i}. {product['name']} - {product['category']} - {product['price']}\n"
                return result
            
            # Update the product
            product_to_update = dict(matches[0])
            old_value = product_to_update[field_to_update]
            
            # Update the specific field
            update_query = f"UPDATE products SET {field_to_update} = ? WHERE id = ?"
            cursor.execute(update_query, (new_value, product_to_update['id']))
            conn.commit()
        
        # Reload product data and reinitialize vector indexes
        product_data = get_all_data('products')
        initialize_vector_indexes()
        
        return f"Successfully updated product '{product_to_update['name']}' {field_to_update} from '{old_value}' to '{new_value}'"
        
    except Exception as e:
        return f"Error updating product: {str(e)}"

@tool
def get_total_sales(period: str = "", start_date: str = "", end_date: str = "", customer: str = "", item: str = "", use_real_date: bool = False) -> str:
    """Get a summary of sales for a specific period or date range.
    
    Args:
        period: Period can be 'today', 'this_week', 'this_month', 'this_year', or 'custom'
        start_date: Start date for custom period (YYYY-MM-DD)
        end_date: End date for custom period (YYYY-MM-DD)
        customer: Optional customer filter
        item: Optional item filter
        use_real_date: Use actual current date instead of reference date
    """
    print(" >> get total sales used")
    
    # Determine which date to use as base
    if use_real_date:
        today = real_now.date()
        date_context = "actual today"
    else:
        # Use reference_date for historical/sample data context
        today = reference_date.date()
        date_context = "reference period"
    
    if period == "today":
        start_date = today.strftime("%Y-%m-%d")
        end_date = start_date
    elif period == "this_week":
        # Start from Monday of current week
        monday = today - timedelta(days=today.weekday())
        start_date = monday.strftime("%Y-%m-%d")
        end_date = (monday + timedelta(days=6)).strftime("%Y-%m-%d")
    elif period == "this_month":
        start_date = today.replace(day=1).strftime("%Y-%m-%d")
        # Get last day of month
        if today.month == 12:
            end_date = today.replace(day=31).strftime("%Y-%m-%d")
        else:
            end_date = (today.replace(month=today.month+1, day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
    elif period == "this_year":
        start_date = today.replace(month=1, day=1).strftime("%Y-%m-%d")
        end_date = today.replace(month=12, day=31).strftime("%Y-%m-%d")
    elif period == "custom" and start_date and end_date:
        # Use provided custom date range
        pass
    else:
        # Default to all time if no valid period specified
        start_date = "1900-01-01"
        end_date = "2999-12-31"
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Base query
            sql = """
            SELECT * FROM sales 
            WHERE (sale_date BETWEEN ? AND ? OR sale_date LIKE ? OR sale_date LIKE ?)
            """
            # Add wildcards to catch partial matches for non-standard formats
            start_wildcard = f"%{start_date}%"
            end_wildcard = f"%{end_date}%"
            params = [start_date, end_date, start_wildcard, end_wildcard]
            
            # Add customer filter if provided
            if customer:
                sql += " AND customer LIKE ?"
                params.append(f"%{customer}%")
                
            # Add item filter if provided
            if item:
                sql += " AND item LIKE ?"
                params.append(f"%{item}%")
            
            # Execute query with parameters
            cursor.execute(sql, params)
            filtered_sales = cursor.fetchall()
    
        if not filtered_sales:
            if period:
                return f"No sales found for {period} ({start_date} to {end_date})."
            else:
                return f"No sales found between {start_date} and {end_date}."
        
        # Calculate total revenue
        total_revenue = 0
        for sale in filtered_sales:
            try:
                price = float(sale['price'])
                total_revenue += price
            except (ValueError, TypeError):
                pass  # Skip if price can't be converted to float
        
        # Generate report
        period_str = ""
        if period == "today":
            period_str = f"today ({start_date})"
        elif period == "this_week":
            period_str = f"this week ({start_date} to {end_date})"
        elif period == "this_month":
            period_str = f"this month ({start_date} to {end_date})"
        elif period == "this_year":
            period_str = f"this year ({start_date} to {end_date})"
        else:
            period_str = f"between {start_date} and {end_date}"
        
        report = [f"Sales Report {period_str}:"]
        report.append(f"Total Sales: ${total_revenue:.2f}")
        report.append(f"Number of Sales: {len(filtered_sales)}")
        
        # Add details of each sale
        report.append("\nSale Details:")
        for i, sale in enumerate(filtered_sales, 1):
            sale_dict = dict(sale)
            report.append(f"{i}. {sale_dict['item']} - ${sale_dict['price']} - Sold to {sale_dict['customer']} on {sale_dict['sale_date']}")
        
        return "\n".join(report)
    except Exception as e:
        return f"Error generating sales report: {str(e)}"

@tool
def sales_by_customer(period: str = "", use_real_date: bool = False) -> str:
    """Generate a report of sales grouped by customer for the given period.
    
    Args:
        period: Period can be 'today', 'this_week', 'this_month', 'this_year', or empty for all time
        use_real_date: Use actual current date instead of reference date
    """
    print(" >> sales by customer used")
    
    # Determine which date to use as base
    if use_real_date:
        today = real_now.date()
    else:
        # Use reference_date for historical/sample data context
        today = reference_date.date()
    
    # Determine date range based on period
    if period == "today":
        start_date = today.strftime("%Y-%m-%d")
        end_date = start_date
        period_str = f"today ({start_date})"
    elif period == "this_week":
        # Start from Monday of current week
        monday = today - timedelta(days=today.weekday())
        start_date = monday.strftime("%Y-%m-%d")
        end_date = (monday + timedelta(days=6)).strftime("%Y-%m-%d")
        period_str = f"this week ({start_date} to {end_date})"
    elif period == "this_month":
        start_date = today.replace(day=1).strftime("%Y-%m-%d")
        # Get last day of month
        if today.month == 12:
            end_date = today.replace(day=31).strftime("%Y-%m-%d")
        else:
            end_date = (today.replace(month=today.month+1, day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
        period_str = f"this month ({start_date} to {end_date})"
    elif period == "this_year":
        start_date = today.replace(month=1, day=1).strftime("%Y-%m-%d")
        end_date = today.replace(month=12, day=31).strftime("%Y-%m-%d")
        period_str = f"this year ({start_date} to {end_date})"
    else:
        # Default to all time if no valid period specified
        start_date = "1900-01-01"
        end_date = "2999-12-31"
        period_str = "all time"
    
    try:
        with get_db_connection() as conn:
            conn.create_function("CAST_FLOAT", 1, lambda x: float(x) if x and x.strip() else 0)
            cursor = conn.cursor()
            
            # Get all sales within date range
            cursor.execute(
                """
                SELECT * FROM sales 
                WHERE sale_date BETWEEN ? AND ?
                ORDER BY customer, sale_date
                """, 
                (start_date, end_date)
            )
            
            filtered_sales = cursor.fetchall()
            
            if not filtered_sales:
                return f"No sales found for {period_str}."
            
            # Use SQL to summarize sales by customer
            cursor.execute(
                """
                SELECT 
                    customer, 
                    COUNT(*) as count, 
                    SUM(CAST_FLOAT(price)) as total 
                FROM sales 
                WHERE sale_date BETWEEN ? AND ? 
                GROUP BY customer 
                ORDER BY total DESC
                """, 
                (start_date, end_date)
            )
            
            customer_summaries = cursor.fetchall()
        
        # Generate report
        report = [f"Sales by Customer for {period_str}:"]
        
        for customer_summary in customer_summaries:
            customer = customer_summary['customer']
            total = customer_summary['total']
            count = customer_summary['count']
            
            report.append(f"\n{customer}:")
            report.append(f"  Total Revenue: ${total:.2f}")
            report.append(f"  Number of Purchases: {count}")
            report.append("  Items Purchased:")
            
            # Filter the full sales list for this customer
            customer_sales = [s for s in filtered_sales if s['customer'] == customer]
            for sale in customer_sales:
                sale_dict = dict(sale)
                try:
                    price = float(sale_dict['price'])
                    report.append(f"    - {sale_dict['item']} (${price:.2f}) on {sale_dict['sale_date']}")
                except (ValueError, TypeError):
                    report.append(f"    - {sale_dict['item']} (${sale_dict['price']}) on {sale_dict['sale_date']}")
        
        return "\n".join(report)
    
    except Exception as e:
        return f"Error generating customer sales report: {str(e)}"

@tool
def sales_by_item(period: str = "", use_real_date: bool = False) -> str:
    """Generate a report of sales grouped by item for the given period.
    
    Args:
        period: Period can be 'today', 'this_week', 'this_month', 'this_year', or empty for all time
        use_real_date: Use actual current date instead of reference date
    """
    print(" >> sales by item used")
    
    # Determine which date to use as base
    if use_real_date:
        today = real_now.date()
    else:
        # Use reference_date for historical/sample data context
        today = reference_date.date()
    
    # Determine date range based on period
    if period == "today":
        start_date = today.strftime("%Y-%m-%d")
        end_date = start_date
        period_str = f"today ({start_date})"
    elif period == "this_week":
        # Start from Monday of current week
        monday = today - timedelta(days=today.weekday())
        start_date = monday.strftime("%Y-%m-%d")
        end_date = (monday + timedelta(days=6)).strftime("%Y-%m-%d")
        period_str = f"this week ({start_date} to {end_date})"
    elif period == "this_month":
        start_date = today.replace(day=1).strftime("%Y-%m-%d")
        # Get last day of month
        if today.month == 12:
            end_date = today.replace(day=31).strftime("%Y-%m-%d")
        else:
            end_date = (today.replace(month=today.month+1, day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
        period_str = f"this month ({start_date} to {end_date})"
    elif period == "this_year":
        start_date = today.replace(month=1, day=1).strftime("%Y-%m-%d")
        end_date = today.replace(month=12, day=31).strftime("%Y-%m-%d")
        period_str = f"this year ({start_date} to {end_date})"
    else:
        # Default to all time if no valid period specified
        start_date = "1900-01-01"
        end_date = "2999-12-31"
        period_str = "all time"
    
    try:
        with get_db_connection() as conn:
            conn.create_function("CAST_FLOAT", 1, lambda x: float(x) if x and x.strip() else 0)
            cursor = conn.cursor()
            
            # Get all sales within date range
            cursor.execute(
                """
                SELECT * FROM sales 
                WHERE sale_date BETWEEN ? AND ?
                ORDER BY item, sale_date
                """, 
                (start_date, end_date)
            )
            
            filtered_sales = cursor.fetchall()
            
            if not filtered_sales:
                return f"No sales found for {period_str}."
            
            # Use SQL to summarize sales by item
            cursor.execute(
                """
                SELECT 
                    item, 
                    COUNT(*) as count, 
                    SUM(CAST_FLOAT(price)) as total,
                    SUM(CAST_FLOAT(price))/COUNT(*) as avg_price
                FROM sales 
                WHERE sale_date BETWEEN ? AND ? 
                GROUP BY item 
                ORDER BY total DESC
                """, 
                (start_date, end_date)
            )
            
            item_summaries = cursor.fetchall()
        
        # Generate report
        report = [f"Sales by Item for {period_str}:"]
        
        for item_summary in item_summaries:
            item = item_summary['item']
            total = item_summary['total']
            count = item_summary['count']
            avg_price = item_summary['avg_price']
            
            report.append(f"\n{item}:")
            report.append(f"  Total Revenue: ${total:.2f}")
            report.append(f"  Units Sold: {count}")
            report.append(f"  Average Price: ${avg_price:.2f}")
            report.append("  Customers:")
            
            # Filter the full sales list for this item
            item_sales = [s for s in filtered_sales if s['item'] == item]
            for sale in item_sales:
                sale_dict = dict(sale)
                try:
                    price = float(sale_dict['price'])
                    report.append(f"    - {sale_dict['customer']} (${price:.2f}) on {sale_dict['sale_date']}")
                except (ValueError, TypeError):
                    report.append(f"    - {sale_dict['customer']} (${sale_dict['price']}) on {sale_dict['sale_date']}")
        
        return "\n".join(report)
    
    except Exception as e:
        return f"Error generating item sales report: {str(e)}"

@tool
def sales_trend_analysis(period_type: str = "monthly", year: str = "", use_real_date: bool = False) -> str:
    """Generate a sales trend analysis by day, week, or month.
    
    Args:
        period_type: Analysis type can be 'daily', 'weekly', or 'monthly'
        year: Optional year parameter to limit analysis to a specific year
        use_real_date: Use actual current date instead of reference date
    """
    print(" >> sales trend analysis used")
    
    # If year is not provided, use appropriate reference year
    if not year:
        if use_real_date:
            year = str(real_now.year)
        else:
            year = str(reference_date.year)
    
    try:
        with get_db_connection() as conn:
            conn.create_function("CAST_FLOAT", 1, lambda x: float(x) if x and x.strip() else 0)
            cursor = conn.cursor()
            
            # Base query to get all sales for the year
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"
            cursor.execute(
                "SELECT * FROM sales WHERE sale_date BETWEEN ? AND ?", 
                (year_start, year_end)
            )
            
            year_filtered_sales = cursor.fetchall()
            
            if not year_filtered_sales:
                return f"No sales found for year {year}."
            
            # Format the grouping based on period_type
            if period_type == "daily":
                # Get all sales grouped by date
                cursor.execute(
                    """
                    SELECT 
                        sale_date as period_key,
                        sale_date as period_label,
                        COUNT(*) as count,
                        SUM(CAST_FLOAT(price)) as total
                    FROM sales 
                    WHERE sale_date BETWEEN ? AND ?
                    GROUP BY sale_date
                    ORDER BY sale_date
                    """,
                    (year_start, year_end)
                )
                
            elif period_type == "weekly":
                # For weekly grouping, we need to extract the week number
                cursor.execute(
                    """
                    SELECT 
                        substr(sale_date, 1, 4) || '-W' || printf('%02d', CAST(strftime('%W', sale_date) AS INTEGER)) as period_key,
                        'Week ' || CAST(strftime('%W', sale_date) AS INTEGER) || ', ' || substr(sale_date, 1, 4) as period_label,
                        COUNT(*) as count,
                        SUM(CAST_FLOAT(price)) as total
                    FROM sales 
                    WHERE sale_date BETWEEN ? AND ?
                    GROUP BY substr(sale_date, 1, 4) || '-W' || printf('%02d', CAST(strftime('%W', sale_date) AS INTEGER))
                    ORDER BY sale_date
                    """,
                    (year_start, year_end)
                )
                
            else:  # monthly default
                # Group by month (YYYY-MM)
                cursor.execute(
                    """
                    SELECT 
                        substr(sale_date, 1, 7) as period_key,
                        CASE 
                            WHEN substr(sale_date, 6, 2) = '01' THEN 'January ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '02' THEN 'February ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '03' THEN 'March ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '04' THEN 'April ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '05' THEN 'May ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '06' THEN 'June ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '07' THEN 'July ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '08' THEN 'August ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '09' THEN 'September ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '10' THEN 'October ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '11' THEN 'November ' || substr(sale_date, 1, 4)
                            WHEN substr(sale_date, 6, 2) = '12' THEN 'December ' || substr(sale_date, 1, 4)
                        END as period_label,
                        COUNT(*) as count,
                        SUM(CAST_FLOAT(price)) as total
                    FROM sales 
                    WHERE sale_date BETWEEN ? AND ?
                    GROUP BY substr(sale_date, 1, 7)
                    ORDER BY period_key
                    """,
                    (year_start, year_end)
                )
                
            # Get all period data
            period_sales = cursor.fetchall()
            
            # Get total revenue and sales for the entire period
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_sales,
                    SUM(CAST_FLOAT(price)) as total_revenue
                FROM sales 
                WHERE sale_date BETWEEN ? AND ?
                """,
                (year_start, year_end)
            )
            
            totals = cursor.fetchone()
            total_revenue = totals['total_revenue']
            total_sales = totals['total_sales']
        
        # Generate report
        report = [f"Sales Trend Analysis ({period_type}) for {year}:"]
        
        report.append(f"\nTotal Revenue: ${total_revenue:.2f}")
        report.append(f"Total Sales Count: {total_sales}")
        if total_sales > 0:
            report.append(f"Average Sale Value: ${total_revenue/total_sales:.2f}")
        
        report.append("\nBreakdown by Period:")
        
        for period in period_sales:
            period_dict = dict(period)
            report.append(f"\n{period_dict['period_label']}:")
            report.append(f"  Revenue: ${period_dict['total']:.2f}")
            report.append(f"  Sales Count: {period_dict['count']}")
            if period_dict['count'] > 0:
                report.append(f"  Average Sale: ${period_dict['total']/period_dict['count']:.2f}")
        
        return "\n".join(report)
    
    except Exception as e:
        return f"Error generating sales trend analysis: {str(e)}"

@tool
def top_performers_report(period: str = "", top_count: int = 5, use_real_date: bool = False) -> str:
    """Generate a report of top performing products and customers.
    
    Args:
        period: Period can be 'today', 'this_week', 'this_month', 'this_year', or empty for all time
        top_count: Number of top performers to show (default 5)
        use_real_date: Use actual current date instead of reference date
    """
    print(" >> top performers report used")
    
    # Determine which date to use as base
    if use_real_date:
        today = real_now.date()
    else:
        # Use reference_date for historical/sample data context
        today = reference_date.date()
    
    # Determine date range based on period
    if period == "today":
        start_date = today.strftime("%Y-%m-%d")
        end_date = start_date
        period_str = f"today ({start_date})"
    elif period == "this_week":
        # Start from Monday of current week
        monday = today - timedelta(days=today.weekday())
        start_date = monday.strftime("%Y-%m-%d")
        end_date = (monday + timedelta(days=6)).strftime("%Y-%m-%d")
        period_str = f"this week ({start_date} to {end_date})"
    elif period == "this_month":
        start_date = today.replace(day=1).strftime("%Y-%m-%d")
        # Get last day of month
        if today.month == 12:
            end_date = today.replace(day=31).strftime("%Y-%m-%d")
        else:
            end_date = (today.replace(month=today.month+1, day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
        period_str = f"this month ({start_date} to {end_date})"
    elif period == "this_year":
        start_date = today.replace(month=1, day=1).strftime("%Y-%m-%d")
        end_date = today.replace(month=12, day=31).strftime("%Y-%m-%d")
        period_str = f"this year ({start_date} to {end_date})"
    else:
        # Default to all time if no valid period specified
        start_date = "1900-01-01"
        end_date = "2999-12-31"
        period_str = "all time"
    
    try:
        with get_db_connection() as conn:
            conn.create_function("CAST_FLOAT", 1, lambda x: float(x) if x and x.strip() else 0)
            cursor = conn.cursor()
            
            # Get top customers by revenue
            cursor.execute(
                """
                SELECT 
                    customer,
                    COUNT(*) as count,
                    SUM(CAST_FLOAT(price)) as revenue,
                    SUM(CAST_FLOAT(price))/COUNT(*) as avg_purchase
                FROM sales 
                WHERE sale_date BETWEEN ? AND ?
                GROUP BY customer
                ORDER BY revenue DESC
                LIMIT ?
                """,
                (start_date, end_date, top_count)
            )
            
            top_customers = cursor.fetchall()
            
            # Get top products by revenue
            cursor.execute(
                """
                SELECT 
                    item,
                    COUNT(*) as count,
                    SUM(CAST_FLOAT(price)) as revenue,
                    SUM(CAST_FLOAT(price))/COUNT(*) as avg_price
                FROM sales 
                WHERE sale_date BETWEEN ? AND ?
                GROUP BY item
                ORDER BY revenue DESC
                LIMIT ?
                """,
                (start_date, end_date, top_count)
            )
            
            top_products = cursor.fetchall()
            
            if not top_customers and not top_products:
                return f"No sales found for {period_str}."
        
        # Generate report
        report = [f"Top Performers Report for {period_str}:"]
        
        # Top customers section
        report.append(f"\nTop {len(top_customers)} Customers by Revenue:")
        for i, customer in enumerate(top_customers, 1):
            customer_dict = dict(customer)
            report.append(f"{i}. {customer_dict['customer']}")
            report.append(f"   Revenue: ${customer_dict['revenue']:.2f}")
            report.append(f"   Purchases: {customer_dict['count']}")
            report.append(f"   Average Purchase: ${customer_dict['avg_purchase']:.2f}")
        
        # Top products section
        report.append(f"\nTop {len(top_products)} Products by Revenue:")
        for i, product in enumerate(top_products, 1):
            product_dict = dict(product)
            report.append(f"{i}. {product_dict['item']}")
            report.append(f"   Revenue: ${product_dict['revenue']:.2f}")
            report.append(f"   Units Sold: {product_dict['count']}")
            report.append(f"   Average Price: ${product_dict['avg_price']:.2f}")
        
        return "\n".join(report)
    
    except Exception as e:
        return f"Error generating top performers report: {str(e)}"

@tool
def parse_date_range(text: str, use_real_date: bool = False) -> str:
    """Parse natural language date ranges into period, start_date, and end_date.
    
    Args:
        text: Natural language date range text
        use_real_date: Use actual current date instead of reference date
    """
    text = text.lower()
    
    # Determine which date context to use
    if use_real_date:
        ref_date = real_now
        context = "real date"
    else:
        ref_date = reference_date
        context = "simulated date"
    
    # Handle specific periods
    if "today" in text:
        return f"Period: today, Context: {context}"
    elif "this week" in text:
        return f"Period: this_week, Context: {context}"
    elif "this month" in text:
        return f"Period: this_month, Context: {context}"
    elif "this year" in text:
        return f"Period: this_year, Context: {context}"
    elif "all time" in text or "overall" in text:
        return f"Period: all time, Context: {context}"
    
    # Handle relative periods
    if "yesterday" in text:
        yesterday = (ref_date - timedelta(days=1)).strftime("%Y-%m-%d")
        return f"Period: custom, Start: {yesterday}, End: {yesterday}, Context: {context}"
    elif "last week" in text:
        today = ref_date.date()
        # Start from last Monday
        last_monday = today - timedelta(days=today.weekday() + 7)
        return f"Period: custom, Start: {last_monday.strftime('%Y-%m-%d')}, End: {(last_monday + timedelta(days=6)).strftime('%Y-%m-%d')}, Context: {context}"
    elif "last month" in text:
        today = ref_date.date()
        first_of_month = today.replace(day=1)
        last_month_end = first_of_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return f"Period: custom, Start: {last_month_start.strftime('%Y-%m-%d')}, End: {last_month_end.strftime('%Y-%m-%d')}, Context: {context}"
    elif "last year" in text:
        today = ref_date.date()
        last_year = today.year - 1
        return f"Period: custom, Start: {last_year}-01-01, End: {last_year}-12-31, Context: {context}"
    
    # Handle explicit date reference - try to detect if user wants actual today
    if "real" in text or "actual" in text:
        if "today" in text:
            real_today = real_now.date().strftime("%Y-%m-%d")
            return f"Period: custom, Start: {real_today}, End: {real_today}, Context: real date"
    
    # If no match, return empty to use default (all time)
    return f"Period: all time, Context: {context}"

@tool
def list_of_plans() -> str:
    """Returns a list of all plans/meetings."""
    print(" >> list of plans used")
    plans = get_all_data('plans')
    if not plans:
        return "No plans found."
    
    lines = []
    for p in plans:
        details = (
            f"Title: {p.get('title','')}, "
            f"With: {p.get('with_who','')}, "
            f"Subject: {p.get('subject','')}, "
            f"When: {p.get('plan_time','')}, "
            f"Notes: {p.get('notes','')}, "
            f"Created At: {p.get('created_at','')}"
        )
        lines.append(details)
    
    return "\n".join(lines)


def create_crm_agent():
    """Create the CRM agent using LangGraph"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True
    )

    checkpointer = InMemorySaver()
    

    tools = [
        save_new_customer,
        search_customers,
        remove_customer,
        create_plan,
        search_plans,
        see_all_plans,
        delete_plan,
        list_of_customers,
        list_of_plans,
        save_the_sales,
        search_sales,
        remove_the_sales,
        save_product,
        search_products,
        remove_product,
        list_of_products,
        send_message_to_customer,
        update_customer,
        update_plan,
        update_sale,
        update_product,
        get_total_sales,
        sales_by_customer,
        sales_by_item,
        sales_trend_analysis,
        top_performers_report,
        parse_date_range,
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer
    )
    
    return agent


def create_receiver_agent():
    """Create the receiver agent using LangGraph"""
    receiver_system_prompt = """
    You are a helpful assistant that receives messages from customers.
    you see a conversation between a customer and a bot.
    you need to analyze the the last message that you receved from the customer and write a so short and friendly report about it.
    Do not include date and time information in your responses unless specifically requested by the user.
    Answer short and concise.

    If the customer is asking for a product, you need to search for the product in the database and return the product information.
    but if the customer is asking for any other information, you need to say 
    """
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Create the receiver agent
    receiver_agent = create_react_agent(
        model=llm,
        tools=[],  # No tools for receiver agent
        prompt=receiver_system_prompt
    )
    
    return receiver_agent

# Initialize agents
main_agent = create_crm_agent()
receiver_agent = create_receiver_agent()

def receiver_agent_response(user_prompt, chat_history=None):
    """Handle receiver agent responses"""
    history = chat_history or chat_history
    
    # Convert chat history to LangGraph format
    messages = []
    for msg in history.get_messages():
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the user prompt
    messages.append({"role": "user", "content": user_prompt})
    
    config = {"configurable": {"thread_id": "receiver_thread"}}
    
    response = receiver_agent.invoke(
        {"messages": messages},
        config=config
    )
    
    # Extract the last message
    last_message = response["messages"][-1]
    history.add_message(last_message.type, last_message.content)
    
    return last_message.content

def main(user_prompt, chat_history=None):
    """Main function to handle user interactions"""
    history = chat_history or chat_history
    history.add_message("user", user_prompt)
    
    # Convert chat history to LangGraph format
    messages = []
    for msg in history.get_messages():
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    config = {"configurable": {"thread_id": "main_thread"}}
    
    response = main_agent.invoke(
        {"messages": messages},
        config=config
    )
    
    # Extract the last message
    last_message = response["messages"][-1]
    history.add_message(last_message.type, last_message.content)
    
    return last_message.content

if __name__ == "__main__":
    # Test the agent
    print("LangGraph CRM Agent initialized successfully!")
    print("Testing with a simple query...")
    
    # Test query
    test_response = main("List all customers", chat_history)
    print(f"Test response: {test_response}")
    
    print("\nAgent is ready to use!")
    print("You can now use main(user_prompt, chat_history) to interact with the agent.") 