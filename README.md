# CRM Telegram Bot Documentation

This documentation explains how to set up and run the CRM Telegram Bot system consisting of two main components:
- `__main__.py` - Telegram bot interface 
- `run_langgraph.py` - CRM backend system with LangGraph

## System Overview

The system is a customer relationship management (CRM) bot that operates through Telegram. It consists of:
- **Telegram Bot Frontend** (`__main__.py`): Handles user interactions via Telegram
- **CRM Backend** (`run_langgraph.py`): Manages customers, plans, sales, and products using AI agents

## Prerequisites

### Python Version
- Python 3.8 or higher

### Required Dependencies

Install the following packages using pip:

```bash
pip install pyrogram
pip install langchain-openai
pip install langgraph
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu if you have GPU support
pip install numpy
pip install sqlite3  # Usually built-in with Python
```

Or install all at once:
```bash
pip install pyrogram langchain-openai langgraph sentence-transformers faiss-cpu numpy
```

## Setup Instructions

### 1. API Keys Configuration

#### OpenAI API Key
In `run_langgraph.py`, you need to set your OpenAI API key:
```python
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
```

#### Telegram API Keys
Create a file named `tg_keys.py` in the same directory with the following structure:
```python
def get_api_keys():
    return [
        "app_name",      # Your app name
        12345678,         # Your Telegram API ID
        "your_api_hash"   # Your Telegram API hash
    ]
```

To get Telegram API credentials:
1. Go to https://my.telegram.org/auth
2. Log in with your phone number
3. Go to "API Development tools"
4. Create a new application to get API ID and API hash
5. Create a bot via @BotFather on Telegram to get the bot token

### 2. File Structure

Ensure your project has the following structure:
```
project_folder/
├── __main__.py
├── run_langgraph.py
├── tg_keys.py
├── data/              # Will be created automatically
├── chats.txt          # Will be created automatically
└── customers_history.json  # Will be created automatically
```

### 3. Database Setup

The system uses SQLite database which will be created automatically in the `data/` folder when you first run the system.

## Running the System

### Option 1: Run the Telegram Bot (Recommended)

To start the complete system with Telegram interface:

```bash
python __main__.py
```

This will:
- Initialize the CRM backend system
- Start the Telegram bot
- Create necessary database tables
- Initialize vector search indexes

### Option 2: Run Backend Only (For Testing)

To test the CRM backend without Telegram:

```bash
python run_langgraph.py
```

This will:
- Initialize the CRM system
- Run a test query
- Show that the system is ready

## System Features

### Customer Management
- Add new customers
- Search customers by any field
- Update customer information
- Remove customers
- List all customers

### Plans/Meetings Management
- Create new plans/meetings
- Search plans by any field
- Update plan information
- Delete plans
- List all plans

### Sales Management
- Record new sales
- Search sales by any criteria
- Update sales information
- Remove sales records
- Generate sales reports

### Product Management
- Add new products
- Search products by any field
- Update product information
- Remove products
- List all products

### Reporting Features
- Total sales reports by period
- Sales by customer analysis
- Sales by item analysis
- Sales trend analysis
- Top performers reports

## Configuration

### Bot Owner ID
In `__main__.py`, update the owner ID to your Telegram user ID:
```python
if sender_id == 7883900921:  # Replace with your Telegram ID
```

### Date Configuration
The system uses a reference date for testing. In `run_langgraph.py`:
```python
reference_date = datetime(2025, 5, 20)  # Adjust as needed
```

## Usage Examples

### For Bot Owner
Send messages to the bot such as:
- "Add customer John Doe with phone 123-456-7890"
- "Show all customers"
- "Create plan meeting with John tomorrow at 3 PM"
- "Add sale laptop $1000 to John today"
- "Show sales report for this month"

### For Customers
Regular users can send messages which will be forwarded to the bot owner with AI analysis.

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed
2. **API Key Error**: Verify your OpenAI API key and Telegram credentials
3. **Database Error**: Check if the `data/` folder has write permissions
4. **Connection Error**: Ensure internet connection for API calls

### Log Files
The system outputs debug information to console. Monitor the console output for any errors.

### Data Files
- `customers_history.json`: Stores customer chat histories
- `chats.txt`: Temporary file for message passing
- `data/crm.db`: SQLite database with all CRM data

## Advanced Configuration

### Vector Search
The system uses sentence-transformers for semantic search. You can modify the model in `run_langgraph.py`:
```python
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### AI Model
Change the OpenAI model in `run_langgraph.py`:
```python
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Change to gpt-4, gpt-3.5-turbo, etc.
    temperature=0.7,
    streaming=True
)
```

## Security Notes

- Keep your API keys secure and never commit them to version control
- The bot owner ID acts as authentication - only this user can access CRM functions
- All data is stored locally in SQLite database

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Ensure API keys are correctly configured
4. Check file permissions for data storage

## Development

To extend the system:
1. Add new tools in `run_langgraph.py`
2. Modify the system prompts for different behavior
3. Add new message handlers in `__main__.py`
4. Extend database schema as needed 
