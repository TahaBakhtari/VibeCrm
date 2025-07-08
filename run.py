import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from swarm import Swarm, Agent
import re
import sqlite3
from contextlib import contextmanager

# Chat history management
class ChatHistory:
    def __init__(self):
        self.messages = []
        
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        
    def get_messages(self):
        return self.messages.copy()
        
    def clear(self):
        self.messages = []

# Global chat history instance
chat_history = ChatHistory()

# Set OpenAI API key as environment variable
os.environ["OPENAI_API_KEY"] = "x"
client = Swarm()

# Initialize sentence transformer model for vector search
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Initialize data storage paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# SQLite database file
DB_FILE = DATA_DIR / "crm.db"

# Create FAISS indexes for different data types
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
        
        # Create customers table
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
        
        # Create plans table
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
        
        # Create sales table
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
        
        # Create products table
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

# Initialize database at startup
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
        # Load customer data
        customer_data = get_all_data('customers')
        
        if customer_data:
            # Create customer embeddings
            customer_texts = []
            
            for customer in customer_data:
                # Create concatenated text for embedding
                text = f"{customer['name']} {customer['phone']} {customer['email']} {customer['address']} {customer['extra_info']}"
                customer_texts.append(normalize_text(text))
            
            # Generate embeddings
            customer_embeddings = model.encode(customer_texts)
            # Initialize FAISS index
            vector_dimension = customer_embeddings.shape[1]
            customer_index = faiss.IndexFlatL2(vector_dimension)
            # Add vectors to index
            customer_index.add(np.array(customer_embeddings).astype('float32'))
    
        # Load plan data
        plan_data = get_all_data('plans')
        
        if plan_data:
            # Create plan embeddings
            plan_texts = []
            
            for plan in plan_data:
                # Create concatenated text for embedding
                text = f"{plan['title']} {plan['with_who']} {plan['subject']} {plan['plan_time']} {plan['notes']}"
                plan_texts.append(normalize_text(text))
            
            # Generate embeddings
            plan_embeddings = model.encode(plan_texts)
            # Initialize FAISS index
            vector_dimension = plan_embeddings.shape[1]
            plan_index = faiss.IndexFlatL2(vector_dimension)
            # Add vectors to index
            plan_index.add(np.array(plan_embeddings).astype('float32'))
    
        # Load sale data
        sale_data = get_all_data('sales')
        
        if sale_data:
            # Create sale embeddings
            sale_texts = []
            
            for sale in sale_data:
                # Create concatenated text for embedding
                text = f"{sale['item']} {sale['price']} {sale['customer']} {sale['sale_date']} {sale['notes']}"
                sale_texts.append(normalize_text(text))
            
            # Generate embeddings
            sale_embeddings = model.encode(sale_texts)
            # Initialize FAISS index
            vector_dimension = sale_embeddings.shape[1]
            sale_index = faiss.IndexFlatL2(vector_dimension)
            # Add vectors to index
            sale_index.add(np.array(sale_embeddings).astype('float32'))
    
        # Load product data
        product_data = get_all_data('products')
        
        if product_data:
            # Create product embeddings
            product_texts = []
            
            for product in product_data:
                # Create concatenated text for embedding
                text = f"{product['name']} {product['category']} {product['price']} {product['quantity']} {product['description']}"
                product_texts.append(normalize_text(text))
            
            # Generate embeddings
            product_embeddings = model.encode(product_texts)
            # Initialize FAISS index
            vector_dimension = product_embeddings.shape[1]
            product_index = faiss.IndexFlatL2(vector_dimension)
            # Add vectors to index
            product_index.add(np.array(product_embeddings).astype('float32'))
    
        return "Vector indexes initialized successfully"
    except Exception as e:
        return f"Error initializing vector indexes: {str(e)}"

# Initialize vector indexes at startup
initialize_vector_indexes()

# Get current system date and time for the system prompt
real_now = datetime.now()
real_date = real_now.strftime("%Y-%m-%d")

# For testing: hardcode a reference date to match the sample data
# But use a date that doesn't match any sales exactly (May 20, 2025)
reference_date = datetime(2025, 5, 20)  # Not matching any specific sale
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

def save_new_customer(name = "", phone = "", email = "", address = "", extra_info = ""):
    print(" >> save new customer used")
    created_at = datetime.now().isoformat()
    
    try:
        global customer_data
        updated = False
        
        # Check if customer already exists
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM customers WHERE LOWER(name) = LOWER(?)",
                (name.strip(),)
            )
            existing_customer = cursor.fetchone()
            
            if existing_customer:
                # Update the existing record
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
                # Insert new customer record
                cursor.execute(
                    """
                    INSERT INTO customers (name, phone, email, address, extra_info, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, phone, email, address, extra_info, created_at)
                )
            
            conn.commit()
        
        # Reload customer data and reinitialize vector indexes
        customer_data = get_all_data('customers')
        initialize_vector_indexes()
        
        if updated:
            return f"Customer '{name}' updated successfully."
        else:
            return f"Successfully saved customer: {name}"
    except Exception as e:
        return f"Error saving customer: {str(e)}"

def search_customers(query="", dimension=""):
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
            # If no specific dimension is provided, search across all dimensions
            if not dimension:
                # Search in all columns
                sql_query = """
                SELECT * FROM customers 
                WHERE name LIKE ? OR phone LIKE ? OR email LIKE ? OR address LIKE ? OR extra_info LIKE ?
                """
                search_term = f"%{query}%"
                cursor.execute(sql_query, (search_term, search_term, search_term, search_term, search_term))
            else:
                # Search in a specific column
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

def remove_customer(search_term = ""):
    print(" >> remove customer used")   
    if not search_term:
        return "Please provide a search term to identify the customer to remove."
    
    try:
        global customer_data
        
        # Find matching customers
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
            
            # Remove the customer
            customer_to_remove = dict(matches[0])
            cursor.execute("DELETE FROM customers WHERE id = ?", (customer_to_remove['id'],))
            conn.commit()
        
        # Reload customer data and reinitialize vector indexes
        customer_data = get_all_data('customers')
        initialize_vector_indexes()
        
        return f"Successfully removed customer: {customer_to_remove['name']}"
        
    except Exception as e:
        return f"Error removing customer: {str(e)}"

def create_plan(title="Meeting", with_who="", subject="", plan_time="", notes=""):
    """
    Add or update a plan in the database.
    If a plan with the same title, with_who, and plan_time exists, update it. Otherwise, add new.
    """
    print(" >> create plan used")
    created_at = datetime.now().isoformat()
    
    try:
        global plan_data
        updated = False
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if plan already exists
            cursor.execute(
                """
                SELECT * FROM plans 
                WHERE LOWER(title) = LOWER(?) AND LOWER(with_who) = LOWER(?) AND plan_time = ?
                """,
                (title.strip(), with_who.strip(), plan_time.strip())
            )
            
            existing_plan = cursor.fetchone()
            
            if existing_plan:
                # Update the existing record
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
                # Insert new plan record
                cursor.execute(
                    """
                    INSERT INTO plans (title, with_who, subject, plan_time, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (title, with_who, subject, plan_time, notes, created_at)
                )
            
            conn.commit()
        
        # Reload plan data and reinitialize vector indexes
        plan_data = get_all_data('plans')
        initialize_vector_indexes()
        
        if updated:
            return f"Plan '{title}' with {with_who} at {plan_time} updated successfully."
        else:
            return f"Successfully saved plan: {title} with {with_who} at {plan_time}"
    except Exception as e:
        return f"Error saving plan: {str(e)}"

def search_plans(query="", dimension=""):
    """
    Search for plans by a query and a specific dimension (field).
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
            
            # Search in a specific column
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
    
def see_all_plans():
    """
    Retrieve and list all saved plans/meetings.
    """
    print(" >> see all plans used")
    try:
        plans = get_all_data('plans')
        if not plans:
            return "No plans or meetings found."
        
        # Sort plans by plan_time
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

def delete_plan(search_term=""):
    """
    Remove a plan from the database.
    """
    print(" >> delete plan used")
    if not search_term:
        return "Please provide a search term to identify the plan to remove."
    
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
            
            # Remove the plan
            plan_to_remove = dict(matches[0])
            cursor.execute("DELETE FROM plans WHERE id = ?", (plan_to_remove['id'],))
            conn.commit()
        
        # Reload plan data and reinitialize vector indexes
        plan_data = get_all_data('plans')
        initialize_vector_indexes()
        
        return f"Successfully removed plan: {plan_to_remove['title']} with {plan_to_remove['with_who']} at {plan_to_remove['plan_time']}"
        
    except Exception as e:
        return f"Error removing plan: {str(e)}"

def list_of_customers():
    """
    Returns a list of all customers.
    """
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

def list_of_plans():
    """
    Returns a list of all plans/meetings.
    """
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

def save_the_sales(item="", price="", customer="", sale_date="", notes=""):
    """
    Saves a sale to the database.
    
    Note: This function accepts any date format. If the date isn't in YYYY-MM-DD format,
    it will be stored as provided but reporting may not work accurately for that entry.
    """
    print(" >> save the sales used")
    created_at = datetime.now().isoformat()
    
    # If no customer is specified, use "Unknown" as default
    if not customer:
        customer = "Unknown"
    
    # If no date is provided, use the current simulated date
    if not sale_date:
        sale_date = current_date
    
    # Handle non-ASCII characters in item, customer, and notes
    # SQLite can store UTF-8 so we don't need to remove non-ASCII, just ensure they're handled properly
    
    try:
        global sale_data
        updated = False

        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if sale already exists (by item, customer, sale_date)
            cursor.execute(
                """
                SELECT * FROM sales 
                WHERE LOWER(item) = LOWER(?) AND LOWER(customer) = LOWER(?) AND sale_date = ?
                """,
                (item.strip(), customer.strip(), sale_date.strip())
            )
            
            existing_sale = cursor.fetchone()
            
            if existing_sale:
                # Update the existing record
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
                # Insert new sale record
                cursor.execute(
                    """
                    INSERT INTO sales (item, price, customer, sale_date, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (item, price, customer, sale_date, notes, created_at)
                )
            
            conn.commit()
        
        # Reload sale data and reinitialize vector indexes
        sale_data = get_all_data('sales')
        initialize_vector_indexes()

        if updated:
            return f"Sale for '{item}' to '{customer}' on '{sale_date}' updated successfully."
        else:
            return f"Successfully saved sale: {item} to {customer} on {sale_date}"
    except Exception as e:
        return f"Error saving sale: {str(e)}"

def search_sales(query="", dimension=""):
    print(" >> search sales used")
    if not query:
        return "Please provide a search query."
    
    valid_dimensions = ['item', 'price', 'customer', 'sale_date', 'notes']
    if not dimension or dimension not in valid_dimensions:
        return f"Please provide a valid dimension to search in: {', '.join(valid_dimensions)}."
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Search in a specific column
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

def remove_the_sales(search_term=""):
    print(" >> remove the sales used")
    if not search_term:
        return "Please provide a search term to identify the sale to remove."
    
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
            
            # Remove the sale
            sale_to_remove = dict(matches[0])
            cursor.execute("DELETE FROM sales WHERE id = ?", (sale_to_remove['id'],))
            conn.commit()

        # Reload sale data and reinitialize vector indexes
        sale_data = get_all_data('sales')
        initialize_vector_indexes()

        return f"Successfully removed sale: {sale_to_remove['item']} to {sale_to_remove['customer']} on {sale_to_remove['sale_date']}"
    
    except Exception as e:
        return f"Error removing sale: {str(e)}"

def update_customer(search_term="", field_to_update="", new_value=""):
    """
    Update a specific field for an existing customer.
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

def update_plan(search_term="", field_to_update="", new_value=""):
    """
    Update a specific field for an existing plan.
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

def update_sale(search_term="", field_to_update="", new_value=""):
    """
    Update a specific field for an existing sale.
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

def update_product(search_term="", field_to_update="", new_value=""):
    """
    Update a specific field for an existing product.
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
                    result += f"{i}. {product['name']} - {product['category']} - ${product['price']}\n"
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

def save_product(name="", category="", price="", quantity="", description=""):
    """
    Saves a product to the database.
    """
    print(" >> save product used")
    created_at = datetime.now().isoformat()
    
    try:
        global product_data
        updated = False

        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if product already exists (by name and category)
            cursor.execute(
                """
                SELECT * FROM products 
                WHERE LOWER(name) = LOWER(?) AND LOWER(category) = LOWER(?)
                """,
                (name.strip(), category.strip())
            )
            
            existing_product = cursor.fetchone()
            
            if existing_product:
                # Update the existing record
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
                # Insert new product record
                cursor.execute(
                    """
                    INSERT INTO products (name, category, price, quantity, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, category, price, quantity, description, created_at)
                )
            
            conn.commit()
        
        # Reload product data and reinitialize vector indexes
        product_data = get_all_data('products')
        initialize_vector_indexes()

        if updated:
            return f"Product '{name}' in category '{category}' updated successfully."
        else:
            return f"Successfully saved product: {name} in category {category}"
    except Exception as e:
        return f"Error saving product: {str(e)}"

def search_products(query="", dimension=""):
    """
    Search for products by a query and a specific dimension (field).
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
            
            # Search in a specific column
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

def remove_product(search_term=""):
    """
    Remove a product from the database.
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

def list_of_products():
    """
    Returns a list of all products.
    """
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

def get_total_sales(period="", start_date="", end_date="", customer="", item="", use_real_date=False):
    """
    Get a summary of sales for a specific period or date range.
    period can be 'today', 'this_week', 'this_month', 'this_year', or 'custom'
    If period is 'custom', start_date and end_date must be provided.
    Optional filters for customer or item.
    Set use_real_date to True to use the actual current date instead of reference date.
    
    Note: This function works best with dates in YYYY-MM-DD format.
    Non-standard date formats may not be included in date-based reports.
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
            
            # Base query - Note: This uses SQL date comparison which works well with YYYY-MM-DD format
            # For non-standard formats, we'll try a broader approach
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

def sales_by_customer(period="", use_real_date=False):
    """
    Generate a report of sales grouped by customer for the given period.
    period can be 'today', 'this_week', 'this_month', 'this_year', or empty for all time.
    Set use_real_date to True to use the actual current date instead of reference date.
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

def sales_by_item(period="", use_real_date=False):
    """
    Generate a report of sales grouped by item for the given period.
    period can be 'today', 'this_week', 'this_month', 'this_year', or empty for all time.
    Set use_real_date to True to use the actual current date instead of reference date.
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

def sales_trend_analysis(period_type="monthly", year="", use_real_date=False):
    """
    Generate a sales trend analysis by day, week, or month.
    period_type can be 'daily', 'weekly', or 'monthly'.
    Optional year parameter to limit analysis to a specific year.
    Set use_real_date to True to use the actual current date instead of reference date.
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
                # SQLite doesn't have date formatting like strftime directly
                # We'll use the actual sale_date for grouping
                
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
                # Using strftime with SQLite for week of year
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

def top_performers_report(period="", top_count=5, use_real_date=False):
    """
    Generate a report of top performing products and customers.
    period can be 'today', 'this_week', 'this_month', 'this_year', or empty for all time.
    top_count is the number of top performers to show (default 5).
    Set use_real_date to True to use the actual current date instead of reference date.
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

def parse_date_range(text, use_real_date=False):
    """
    Parse natural language date ranges into period, start_date, and end_date.
    Returns a tuple of (period, start_date, end_date).
    Set use_real_date to True to use the actual current date instead of reference date.
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
        return "today", "", "", context
    elif "this week" in text:
        return "this_week", "", "", context
    elif "this month" in text:
        return "this_month", "", "", context
    elif "this year" in text:
        return "this_year", "", "", context
    elif "all time" in text or "overall" in text:
        return "", "", "", context
    
    # Handle relative periods
    if "yesterday" in text:
        yesterday = (ref_date - timedelta(days=1)).strftime("%Y-%m-%d")
        return "custom", yesterday, yesterday, context
    elif "last week" in text:
        today = ref_date.date()
        # Start from last Monday
        last_monday = today - timedelta(days=today.weekday() + 7)
        return "custom", last_monday.strftime("%Y-%m-%d"), (last_monday + timedelta(days=6)).strftime("%Y-%m-%d"), context
    elif "last month" in text:
        today = ref_date.date()
        first_of_month = today.replace(day=1)
        last_month_end = first_of_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return "custom", last_month_start.strftime("%Y-%m-%d"), last_month_end.strftime("%Y-%m-%d"), context
    elif "last year" in text:
        today = ref_date.date()
        last_year = today.year - 1
        return "custom", f"{last_year}-01-01", f"{last_year}-12-31", context
    
    # Handle explicit date reference - try to detect if user wants actual today
    if "real" in text or "actual" in text:
        if "today" in text:
            real_today = real_now.date().strftime("%Y-%m-%d")
            return "custom", real_today, real_today, "real date"
    
    # If no match, return empty to use default (all time)
    return "", "", "", context

def send_message_to_customer(user_id, message):
    """
    user_id: The user's Telegram ID
    message: The message to send
    """
    print(" >> send message to customer used")
    data = f"{user_id}###{message}"
    with open("chats.txt", "a", encoding="utf-8") as file:
        file.write(data)
    print(" >> message saved for customer : " , data)
    return "Sent message to customer"


main_agent = Agent(
    name="CRM Agent",
    instructions=system_prompt,
    functions=[
        save_new_customer,
        search_customers,
        remove_customer,
        create_plan,
        search_plans,
        delete_plan,
        list_of_customers,
        list_of_plans,
        save_the_sales,
        search_sales,
        remove_the_sales,
        update_customer,
        update_plan,
        update_sale,
        update_product,
        save_product,
        search_products,
        remove_product,
        list_of_products,
        get_total_sales,
        sales_by_customer,
        sales_by_item,
        sales_trend_analysis,
        top_performers_report,
        parse_date_range,
        send_message_to_customer,
    ],
)

#26 agents


receiver_system_prompt = """
You are a helpful assistant that receives messages from customers.
you see a conversation between a customer and a bot.
you need to analyze the the last message that you receved from the customer and write a so short and friendly report about it.
Do not include date and time information in your responses unless specifically requested by the user.
Answer short and concise.

If the customer is asking for a product, you need to search for the product in the database and return the product information.
but if the customer is asking for any other information, you need to say 
"""

receiver_agent = Agent(
    name="Receiver Agent",
    instructions=receiver_system_prompt,
)

def receiver_agent_response(user_prompt, chat_history=None):
    history = chat_history or chat_history
    messages = history.get_messages()
    response = client.run(
        agent=receiver_agent,
        messages=messages,
    )
    last_message = response.messages[-1]
    history.add_message(last_message["role"], last_message["content"])

    return last_message["content"]

def main(user_prompt, chat_history=None):
    history = chat_history or chat_history
    history.add_message("user", user_prompt)
    messages = history.get_messages()
    response = client.run(
        agent=main_agent,
        messages=messages,
    )
    last_message = response.messages[-1]
    history.add_message(last_message["role"], last_message["content"])
    
    return last_message["content"]
