import os
import json
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, g
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
import calendar
import run  # Import the functions from run.py
import re

app = Flask(__name__)
app.secret_key = "x"  # Required for flash messages

# Add float to template context
app.jinja_env.globals.update(float=float)

# Custom filter to parse price string
@app.template_filter('parse_price')
def parse_price(price_str):
    if not price_str:
        return 0
    # Extract numeric part using regex
    numeric_str = re.search(r'\d+', str(price_str))
    if numeric_str:
        return float(numeric_str.group())
    return 0

# Custom filter to parse quantity string
@app.template_filter('parse_quantity')
def parse_quantity(quantity_str):
    if not quantity_str:
        return 0
    # Extract numeric part using regex
    numeric_str = re.search(r'\d+', str(quantity_str))
    if numeric_str:
        return int(numeric_str.group())
    return 0

# Initialize data storage paths
DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "crm.db"

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    try:
        yield conn
    finally:
        conn.close()

def get_sales_data(period='monthly'):
    """Get sales data for charts"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if period == 'weekly':
            # Get last 7 days sales
            dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
            sales_data = []
            for date in dates:
                cursor.execute("""
                    SELECT COALESCE(SUM(price), 0) as total
                    FROM sales 
                    WHERE date(sale_date) = date(?)
                """, (date,))
                total = cursor.fetchone()['total']
                sales_data.append(total)
            
            # Convert dates to Persian weekday names
            weekdays = ['شنبه', 'یکشنبه', 'دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنج‌شنبه', 'جمعه']
            labels = weekdays
            
        else:  # monthly
            # Get last 6 months sales
            months_data = []
            persian_months = [
                'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور',
                'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'
            ]
            current_month = datetime.now().month
            labels = []
            sales_data = []
            
            for i in range(5, -1, -1):
                month = (current_month - i - 1) % 12 + 1
                year = datetime.now().year - ((current_month - i - 1) // 12)
                
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, month + 1, 1) - timedelta(days=1)
                
                cursor.execute("""
                    SELECT COALESCE(SUM(price), 0) as total
                    FROM sales 
                    WHERE date(sale_date) BETWEEN date(?) AND date(?)
                """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                
                total = cursor.fetchone()['total']
                sales_data.append(total)
                labels.append(persian_months[month - 1])
        
        # Get sales distribution data by joining with products table
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN p.category IS NOT NULL THEN p.category
                    ELSE 'سایر'
                END as category,
                COUNT(*) as count,
                COALESCE(SUM(s.price), 0) as total
            FROM sales s
            LEFT JOIN products p ON s.item = p.name
            GROUP BY 
                CASE 
                    WHEN p.category IS NOT NULL THEN p.category
                    ELSE 'سایر'
                END
        """)
        distribution_data = cursor.fetchall()
        
        # If no data, provide default categories
        if not distribution_data:
            distribution_data = [
                {'category': 'محصولات', 'total': 0},
                {'category': 'خدمات', 'total': 0},
                {'category': 'سایر', 'total': 0}
            ]
        
        # Map English categories to Persian
        category_map = {
            'product': 'محصولات',
            'service': 'خدمات',
            'other': 'سایر',
            None: 'سایر'
        }
        
        # Transform categories to Persian
        distribution_data = [
            {
                'category': category_map.get(row['category'], row['category']),
                'total': row['total']
            }
            for row in distribution_data
        ]
        
        return {
            'labels': labels,
            'sales_data': sales_data,
            'distribution': {
                'labels': [row['category'] for row in distribution_data],
                'data': [row['total'] for row in distribution_data]
            }
        }

# Home page - Dashboard
@app.route('/')
def home():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Get counts for dashboard
        customers_count = cursor.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        plans_count = cursor.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
        sales_count = cursor.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        products_count = cursor.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    
    # Get chart data
    chart_data = get_sales_data('monthly')
    
    # If it's an AJAX request, return JSON data
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        period = request.args.get('period', 'monthly')
        return jsonify(get_sales_data(period))
    
    return render_template('index.html', 
                          customers_count=customers_count,
                          plans_count=plans_count,
                          sales_count=sales_count,
                          products_count=products_count,
                          chart_data=chart_data)

# -------- CUSTOMERS SECTION --------
@app.route('/customers')
def list_customers():
    customers = run.get_all_data('customers')
    return render_template('customers.html', customers=customers)

@app.route('/customers/add', methods=['GET', 'POST'])
def add_customer():
    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        email = request.form.get('email')
        address = request.form.get('address')
        extra_info = request.form.get('extra_info')
        
        result = run.save_new_customer(name, phone, email, address, extra_info)
        flash(result)
        return redirect(url_for('list_customers'))
    
    return render_template('customer_form.html')

@app.route('/customers/edit/<int:id>', methods=['GET', 'POST'])
def edit_customer(id):
    if request.method == 'POST':
        field = request.form.get('field')
        value = request.form.get('value')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE customers SET {field} = ? WHERE id = ?", (value, id))
            conn.commit()
            
        flash("اطلاعات مشتری با موفقیت به‌روزرسانی شد")
        return redirect(url_for('list_customers'))
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        customer = cursor.execute("SELECT * FROM customers WHERE id = ?", (id,)).fetchone()
    
    if not customer:
        flash("مشتری مورد نظر یافت نشد")
        return redirect(url_for('list_customers'))
    
    return render_template('customer_edit.html', customer=customer)

@app.route('/customers/delete/<int:id>')
def delete_customer(id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        customer = cursor.execute("SELECT name FROM customers WHERE id = ?", (id,)).fetchone()
        
        if customer:
            cursor.execute("DELETE FROM customers WHERE id = ?", (id,))
            conn.commit()
            flash(f"مشتری {customer['name']} با موفقیت حذف شد")
        else:
            flash("مشتری مورد نظر یافت نشد")
    
    return redirect(url_for('list_customers'))

@app.route('/customers/search', methods=['GET', 'POST'])
def search_customers():
    results = []
    if request.method == 'POST':
        query = request.form.get('query')
        dimension = request.form.get('dimension')
        
        if query:
            result_text = run.search_customers(query, dimension)
            flash(result_text)
            
            # Also get the actual results for display
            with get_db_connection() as conn:
                cursor = conn.cursor()
                sql = "SELECT * FROM customers WHERE 1=0"
                
                if dimension:
                    sql = f"SELECT * FROM customers WHERE {dimension} LIKE ?"
                    results = cursor.execute(sql, (f"%{query}%",)).fetchall()
                else:
                    sql = """
                    SELECT * FROM customers 
                    WHERE name LIKE ? OR phone LIKE ? OR email LIKE ? 
                    OR address LIKE ? OR extra_info LIKE ?
                    """
                    search_term = f"%{query}%"
                    results = cursor.execute(sql, (search_term, search_term, search_term, search_term, search_term)).fetchall()
    
    return render_template('customer_search.html', results=results)

# -------- PLANS SECTION --------
@app.route('/plans')
def list_plans():
    plans = run.get_all_data('plans')
    return render_template('plans.html', plans=plans)

@app.route('/plans/add', methods=['GET', 'POST'])
def add_plan():
    if request.method == 'POST':
        title = request.form.get('title')
        with_who = request.form.get('with_who')
        subject = request.form.get('subject')
        plan_time = request.form.get('plan_time')
        notes = request.form.get('notes')
        
        result = run.create_plan(title, with_who, subject, plan_time, notes)
        flash(result)
        return redirect(url_for('list_plans'))
    
    return render_template('plan_form.html')

@app.route('/plans/edit/<int:id>', methods=['GET', 'POST'])
def edit_plan(id):
    if request.method == 'POST':
        field = request.form.get('field')
        value = request.form.get('value')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE plans SET {field} = ? WHERE id = ?", (value, id))
            conn.commit()
            
        flash(f"Plan field {field} updated successfully")
        return redirect(url_for('list_plans'))
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        plan = cursor.execute("SELECT * FROM plans WHERE id = ?", (id,)).fetchone()
    
    if not plan:
        flash("Plan not found")
        return redirect(url_for('list_plans'))
    
    return render_template('plan_edit.html', plan=plan)

@app.route('/plans/delete/<int:id>')
def delete_plan(id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        plan = cursor.execute("SELECT title FROM plans WHERE id = ?", (id,)).fetchone()
        
        if plan:
            cursor.execute("DELETE FROM plans WHERE id = ?", (id,))
            conn.commit()
            flash(f"Plan {plan['title']} deleted successfully")
        else:
            flash("Plan not found")
    
    return redirect(url_for('list_plans'))

@app.route('/plans/search', methods=['GET', 'POST'])
def search_plans():
    results = []
    if request.method == 'POST':
        query = request.form.get('query')
        dimension = request.form.get('dimension')
        
        if query and dimension:
            result_text = run.search_plans(query, dimension)
            flash(result_text)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                sql = f"SELECT * FROM plans WHERE {dimension} LIKE ?"
                results = cursor.execute(sql, (f"%{query}%",)).fetchall()
    
    return render_template('plan_search.html', results=results)

# -------- SALES SECTION --------
@app.route('/sales')
def list_sales():
    sales = run.get_all_data('sales')
    return render_template('sales.html', sales=sales)

@app.route('/sales/add', methods=['GET', 'POST'])
def add_sale():
    if request.method == 'POST':
        item = request.form.get('item')
        price = request.form.get('price')
        customer = request.form.get('customer')
        sale_date = request.form.get('sale_date')
        notes = request.form.get('notes')
        
        result = run.save_the_sales(item, price, customer, sale_date, notes)
        flash(result)
        return redirect(url_for('list_sales'))
    
    # Get customers for the dropdown
    customers = run.get_all_data('customers')
    products = run.get_all_data('products')
    
    return render_template('sale_form.html', customers=customers, products=products)

@app.route('/sales/edit/<int:id>', methods=['GET', 'POST'])
def edit_sale(id):
    if request.method == 'POST':
        field = request.form.get('field')
        value = request.form.get('value')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE sales SET {field} = ? WHERE id = ?", (value, id))
            conn.commit()
            
        flash(f"Sale field {field} updated successfully")
        return redirect(url_for('list_sales'))
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        sale = cursor.execute("SELECT * FROM sales WHERE id = ?", (id,)).fetchone()
    
    if not sale:
        flash("Sale not found")
        return redirect(url_for('list_sales'))
    
    return render_template('sale_edit.html', sale=sale)

@app.route('/sales/delete/<int:id>')
def delete_sale(id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        sale = cursor.execute("SELECT item FROM sales WHERE id = ?", (id,)).fetchone()
        
        if sale:
            cursor.execute("DELETE FROM sales WHERE id = ?", (id,))
            conn.commit()
            flash(f"Sale of {sale['item']} deleted successfully")
        else:
            flash("Sale not found")
    
    return redirect(url_for('list_sales'))

@app.route('/sales/search', methods=['GET', 'POST'])
def search_sales():
    results = []
    if request.method == 'POST':
        query = request.form.get('query')
        dimension = request.form.get('dimension')
        
        if query and dimension:
            result_text = run.search_sales(query, dimension)
            flash(result_text)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                sql = f"SELECT * FROM sales WHERE {dimension} LIKE ?"
                results = cursor.execute(sql, (f"%{query}%",)).fetchall()
    
    return render_template('sale_search.html', results=results)

# -------- PRODUCTS SECTION --------
@app.route('/products')
def list_products():
    products = run.get_all_data('products')
    return render_template('products.html', products=products)

@app.route('/products/add', methods=['GET', 'POST'])
def add_product():
    if request.method == 'POST':
        name = request.form.get('name')
        category = request.form.get('category')
        price = request.form.get('price')
        quantity = request.form.get('quantity')
        description = request.form.get('description')
        
        result = run.save_product(name, category, price, quantity, description)
        flash(result)
        return redirect(url_for('list_products'))
    
    return render_template('product_form.html')

@app.route('/products/edit/<int:id>', methods=['GET', 'POST'])
def edit_product(id):
    if request.method == 'POST':
        field = request.form.get('field')
        value = request.form.get('value')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE products SET {field} = ? WHERE id = ?", (value, id))
            conn.commit()
            
        flash(f"Product field {field} updated successfully")
        return redirect(url_for('list_products'))
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        product = cursor.execute("SELECT * FROM products WHERE id = ?", (id,)).fetchone()
    
    if not product:
        flash("Product not found")
        return redirect(url_for('list_products'))
    
    return render_template('product_edit.html', product=product)

@app.route('/products/delete/<int:id>')
def delete_product(id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        product = cursor.execute("SELECT name FROM products WHERE id = ?", (id,)).fetchone()
        
        if product:
            cursor.execute("DELETE FROM products WHERE id = ?", (id,))
            conn.commit()
            flash(f"Product {product['name']} deleted successfully")
        else:
            flash("Product not found")
    
    return redirect(url_for('list_products'))

@app.route('/products/search', methods=['GET', 'POST'])
def search_products():
    results = []
    if request.method == 'POST':
        query = request.form.get('query')
        dimension = request.form.get('dimension')
        
        if query and dimension:
            result_text = run.search_products(query, dimension)
            flash(result_text)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                sql = f"SELECT * FROM products WHERE {dimension} LIKE ?"
                results = cursor.execute(sql, (f"%{query}%",)).fetchall()
    
    return render_template('product_search.html', results=results)

# -------- REPORTS SECTION --------
@app.route('/reports')
def reports_dashboard():
    return render_template('reports.html')

@app.route('/reports/total-sales', methods=['GET', 'POST'])
def total_sales_report():
    report_text = ""
    if request.method == 'POST':
        period = request.form.get('period')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        customer = request.form.get('customer')
        item = request.form.get('item')
        use_real_date = 'use_real_date' in request.form
        
        report_text = run.get_total_sales(period, start_date, end_date, customer, item, use_real_date)
    
    customers = run.get_all_data('customers')
    products = run.get_all_data('products')
    
    return render_template('total_sales_report.html', 
                          report_text=report_text,
                          customers=customers,
                          products=products)

@app.route('/reports/sales-by-customer', methods=['GET', 'POST'])
def sales_by_customer_report():
    report_text = ""
    if request.method == 'POST':
        period = request.form.get('period')
        use_real_date = 'use_real_date' in request.form
        
        report_text = run.sales_by_customer(period, use_real_date)
    
    return render_template('sales_by_customer_report.html', report_text=report_text)

@app.route('/reports/sales-by-item', methods=['GET', 'POST'])
def sales_by_item_report():
    report_text = ""
    if request.method == 'POST':
        period = request.form.get('period')
        use_real_date = 'use_real_date' in request.form
        
        report_text = run.sales_by_item(period, use_real_date)
    
    return render_template('sales_by_item_report.html', report_text=report_text)

@app.route('/reports/trend-analysis', methods=['GET', 'POST'])
def trend_analysis_report():
    report_text = ""
    if request.method == 'POST':
        period_type = request.form.get('period_type')
        year = request.form.get('year')
        use_real_date = 'use_real_date' in request.form
        
        report_text = run.sales_trend_analysis(period_type, year, use_real_date)
    
    current_year = datetime.now().year
    years = list(range(current_year - 5, current_year + 1))
    
    return render_template('trend_analysis_report.html', 
                          report_text=report_text,
                          years=years)

@app.route('/reports/top-performers', methods=['GET', 'POST'])
def top_performers_report():
    report_text = ""
    if request.method == 'POST':
        period = request.form.get('period')
        top_count = int(request.form.get('top_count', 5))
        use_real_date = 'use_real_date' in request.form
        
        report_text = run.top_performers_report(period, top_count, use_real_date)
    
    return render_template('top_performers_report.html', report_text=report_text)

# Run the Flask app
if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    template_dir = Path('templates')
    template_dir.mkdir(exist_ok=True)
    
    app.run(debug=True) 