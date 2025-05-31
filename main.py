import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, date
from flask import Flask, jsonify, request, send_file, make_response, render_template, redirect, url_for, flash
from flask_cors import CORS
import subprocess
import traceback
import logging
import os
from dotenv import load_dotenv
import json
import shutil
import requests
from werkzeug.utils import secure_filename
import tempfile
import fitz  # PyMuPDF
import re
import openai
from billcalendar import get_calendar_data
import billcalendar
from typing import List
import uuid  # Import the UUID library

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="static/pages")
CORS(app)

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --------- Configuration ---------
CSV_PATH = "simulated_revenue.csv"
SEQ_LENGTH = 100
BATCH_SIZE = 16
HIDDEN_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.002
PREDICT_DAYS = 50
USERS_FILE = 'users.json'
DATA_DIR = 'user_data'  # Define a base directory for user data

def get_user_data_path(user_id, filename):
    """Generates a user-specific data file path."""
    user_dir = os.path.join(DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)  # Ensure user's directory exists
    return os.path.join(user_dir, filename)

# --- Expenses Helper Functions ---
def get_user_expenses_path(user_id):
    """Return the path to the user's expenses.json file."""
    user_dir = os.path.join(DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "expenses.json")

def load_user_expenses(user_id):
    path = get_user_expenses_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_user_expenses(user_id, expenses):
    path = get_user_expenses_path(user_id)
    with open(path, "w") as f:
        json.dump(expenses, f, indent=2)

# --- Inventory Helper Functions ---
def get_user_inventory_path(user_id):
    user_dir = os.path.join(DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "inventory.json")

def get_user_restock_actions_path(user_id):
    user_dir = os.path.join(DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "restock_actions.json")

def save_user_inventory(user_id, inventory):
    with open(get_user_inventory_path(user_id), "w") as f:
        json.dump(inventory, f, indent=2)

def load_user_inventory(user_id):
    path = get_user_inventory_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_user_restock_actions(user_id, restock_actions):
    with open(get_user_restock_actions_path(user_id), "w") as f:
        json.dump(restock_actions, f, indent=2)

def load_user_restock_actions(user_id):
    path = get_user_restock_actions_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

# --- Employee Helper Functions ---
def get_user_employees_path(user_id):
    user_dir = os.path.join(DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "employees.json")

def load_user_employees(user_id):
    path = get_user_employees_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_user_employees(user_id, employees):
    path = get_user_employees_path(user_id)
    with open(path, "w") as f:
        json.dump(employees, f, indent=2)

# --------- Helper Functions ---------
def ensure_directories():
    """Ensure all required directories exist"""
    directories = ['static', 'static/pages', 'templates', 'templates/pages']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def initialize_app():
    """Initialize the application and run required setup"""
    ensure_directories()
    # Load environment variables from .env
    load_dotenv()
    # Set OpenAI API key from .env
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OPENAI_API_KEY not set in environment variables")

    # Check for other required environment variables
    required_vars = ["GOOGLE_API_KEY", "GOOGLE_CX"]
    for var in required_vars:
        if not os.getenv(var):
            logger.error(f"{var} not set in environment variables")


# --------- User Management ---------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

@app.route('/register', methods=['POST'])
def register():
    # Removed empty try statement as it had no except/finally clause
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'message': 'Invalid request.'}), 400

        username = data.get('username', '').strip()
        password = data.get('password', '')
        email = data.get('email', '').strip()  # Get email
        company_name = data.get('company_name', '').strip()  # Get company name

        if not username or not password or not email or not company_name:
            return jsonify({'message': 'Username, password, email, and company name required.'}), 400

        users = load_users()
        if username in users:
            return jsonify({'message': 'Username already exists.'}), 409

        user_id = str(uuid.uuid4())  # Generate a unique user ID

        user_data = {
            'password': password,
            'user_id': user_id,
            'email': email,
            'company_name': company_name
        }

        users[username] = user_data  # Store all user data

        save_users(users)
        return jsonify({'message': 'Registration successful.', 'user_id': user_id}), 200  # Return user ID

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'message': 'Invalid request.'}), 400
    username = data.get('username', '').strip()
    password = data.get('password', '')
    users = load_users()
    if username in users and users[username]['password'] == password:
        user = users[username]
        return jsonify({
            'message': 'Login successful.',
            'user_id': user.get('user_id'),
            'email': user.get('email'),
            'company_name': user.get('company_name')
        }), 200
    else:
        return jsonify({'message': 'Invalid username or password.'}), 401

# --------- Employee Management ---------
@app.route('/employees', methods=['GET'])
def list_employees():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400
    employees = load_user_employees(user_id)
    return jsonify(employees)

@app.route('/employees', methods=['POST'])
def add_employee():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'message': 'Invalid request.'}), 400
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400
    employees = load_user_employees(user_id)
    employees.append(data)
    save_user_employees(user_id, employees)
    return jsonify({'message': 'Employee added successfully.'}), 201

@app.route('/employees/<int:employee_id>', methods=['GET'])
def get_employee(employee_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400
    employees = load_user_employees(user_id)
    if employee_id < 0 or employee_id >= len(employees):
        return jsonify({'message': 'Employee not found.'}), 404
    return jsonify(employees[employee_id])

@app.route('/employees/<int:employee_id>', methods=['PUT'])
def update_employee(employee_id):
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'message': 'Invalid request.'}), 400
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400
    employees = load_user_employees(user_id)
    if employee_id < 0 or employee_id >= len(employees):
        return jsonify({'message': 'Employee not found.'}), 404
    employees[employee_id] = data
    save_user_employees(user_id, employees)
    return jsonify({'message': 'Employee updated successfully.'}), 200

@app.route('/employees/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400
    employees = load_user_employees(user_id)
    if employee_id < 0 or employee_id >= len(employees):
        return jsonify({'message': 'Employee not found.'}), 404
    del employees[employee_id]
    save_user_employees(user_id, employees)
    return jsonify({'message': 'Employee deleted successfully.'}), 200

@app.route('/api/employees/<int:employee_id>', methods=['DELETE'])
def api_delete_employee(employee_id):
    # Accept user_id from query param or JSON body
    user_id = request.args.get('user_id')
    if not user_id and request.is_json:
        data = request.get_json(force=True, silent=True)
        user_id = data.get('user_id') if data else None
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400
    employees = load_user_employees(user_id)
    if employee_id < 0 or employee_id >= len(employees):
        return jsonify({'message': 'Employee not found.'}), 404
    del employees[employee_id]
    save_user_employees(user_id, employees)
    return jsonify({'message': 'Employee deleted successfully.'}), 200

@app.route('/manage_employees.html')
def serve_manage_employees_html():
    return send_file('static/pages/manage_employees.html')

# --------- Dataset Class ---------
class RevenueDataset(Dataset):
    def __init__(self, user_id, seq_length):
        self.seq_length = seq_length
        csv_path = get_user_data_path(user_id, 'simulated_revenue.csv')
        df = pd.read_csv(csv_path)

        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except KeyError:
            print("Error: 'Date' column not found in DataFrame.")
            raise
        except ValueError as e:
            print(f"Error parsing dates: {e}")
            raise

        df.sort_values('Date', inplace=True)

        # Keep only the last 200 rows (or less)
        df = df.iloc[-seq_length:]

        avg_revenue_by_day = df.groupby('DayOfWeek')['Revenue'].mean()
        df['DayOfWeek'] = df['DayOfWeek'].map(avg_revenue_by_day)
        self.feature_cols = ['DayOfWeek']
        self.target_col = 'Revenue'

        self.scaler = MinMaxScaler()
        try:
            scaled = self.scaler.fit_transform(df[self.feature_cols + [self.target_col]])
        except KeyError as e:
            print(f"Error: Column(s) missing in DataFrame: {e}")
            raise

        self.features = scaled[:, :-1]
        self.targets = scaled[:, -1]

        self.original_df = df.reset_index(drop=True)
        self.scaler_target = MinMaxScaler()
        self.scaler_target.fit(df[[self.target_col]])

    def __len__(self):
        return max(1, len(self.features) - 1)

    def __getitem__(self, idx):
        end_idx = idx + 1
        start_idx = max(0, end_idx - self.seq_length)
        seq_x = self.features[start_idx:end_idx]

        if len(seq_x) < self.seq_length:
            padding = np.zeros((self.seq_length - len(seq_x), seq_x.shape[1]))
            seq_x = np.vstack([padding, seq_x])

        y = self.targets[end_idx - 1]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --------- RNN Model ---------
class RevenueRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RevenueRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.squeeze()

# --------- Training Function ---------
def train_model(model, dataloader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# --------- Forecasting Function ---------
def forecast_next_month(model, dataset):
    model.eval()
    seq = dataset.features[-SEQ_LENGTH:]
    if len(seq) < SEQ_LENGTH:
        pad = np.zeros((SEQ_LENGTH - len(seq), seq.shape[1]))
        seq = np.vstack([pad, seq])
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    predictions = []
    last_date = dataset.original_df['Date'].max()

    for _ in range(PREDICT_DAYS):
        with torch.no_grad():
            pred = model(seq).item()
        predictions.append(pred)

        # Next date
        next_date = last_date + timedelta(days=1)
        next_day = next_date.strftime('%A')
        next_day_num = pd.Series([next_day]).astype('category').cat.codes[0]
        new_feature = np.array([next_day_num])

        # Update sequence
        seq = seq.squeeze(0).numpy()
        seq = np.vstack([seq[1:], new_feature])
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        last_date = next_date

    predicted_scaled = np.array(predictions).reshape(-1, 1)
    predicted_real = dataset.scaler_target.inverse_transform(predicted_scaled)
    dates = pd.date_range(start=dataset.original_df['Date'].max() + timedelta(days=1), periods=PREDICT_DAYS)

    result_df = pd.DataFrame({
        'date': dates,
        'predicted_revenue': predicted_real.flatten()
    })

    result_df.to_csv("predicted_revenue.csv", index=False)
    print("📈 Forecast saved to predicted_revenue.csv")

# --------- Inventory Check Function ---------
def run_inventory_check(tolerance=7, restock_days=14):
    pricing_data = pd.read_csv('pricingdata.csv')
    pricing_data['Date'] = pd.to_datetime(pricing_data['Date'])
    inventory_data = pd.read_csv('store_inventory.csv')
    inventory_data.rename(columns={'InitialInventory': 'Inventory'}, inplace=True)

    avg_sales = pricing_data.groupby('ItemID').agg(
        total_sold=('QuantitySold', 'sum'),
        days_sold=('Date', lambda x: (x.max() - x.min()).days + 1),
        avg_cost=('CostPerItem', 'mean')
    ).reset_index()
    avg_sales['avg_daily_sales'] = avg_sales['total_sold'] / avg_sales['days_sold']

    last_date = pricing_data['Date'].max()
    restock_actions = []
    avg_sales_dict = avg_sales.set_index('ItemID').to_dict('index')

    inventory_list = []
    for _, inv_row in inventory_data.iterrows():
        item = inv_row['ItemID']
        current_inventory = inv_row['Inventory']
        sales_info = avg_sales_dict.get(item, {})
        avg_daily_sale = sales_info.get('avg_daily_sales', 0)
        avg_cost = sales_info.get('avg_cost', 0)
        if avg_daily_sale > 0:
            days_until_runout = current_inventory / avg_daily_sale
            restock_date = last_date + timedelta(days=int(days_until_runout) - tolerance)
            restock_amount = int(avg_daily_sale * restock_days + 0.5)
            restock_cost = restock_amount * avg_cost
            need_restock = days_until_runout <= tolerance
            inventory_list.append({
                "ItemID": item,
                "Inventory": current_inventory,
                "restock_date": str(restock_date.date()),
                "restock_amount": restock_amount,
                "restock_cost": round(restock_cost, 2),
                "need_restock": need_restock,
                "avg_cost": round(avg_cost, 2)  # Add average cost
            })
            if need_restock:
                new_inv = current_inventory + restock_amount
                inventory_data.loc[inventory_data['ItemID'] == item, 'Inventory'] = new_inv
                restock_actions.append({
                    "item": item,
                    "restock_date": str(restock_date.date()),
                    "restock_amount": restock_amount,
                    "restock_cost": round(restock_cost, 2),
                    "new_inventory": new_inv
                })
        else:
            inventory_list.append({
                "ItemID": item,
                "Inventory": current_inventory,
                "restock_date": None,
                "restock_amount": 0,
                "restock_cost": 0,
                "need_restock": False
            })
    inventory_data.to_csv('store_inventory.csv', index=False)
    return {
        "restock_actions": restock_actions,
        "inventory": inventory_list
    }

@app.route('/run_inventory', methods=['POST'])
def run_inventory():
    try:
        # --- Get user_id from form or JSON ---
        user_id = None
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            user_id = request.form.get('user_id')
            inventory_file = request.files.get('inventory_csv')
            tolerance = int(request.form.get('tolerance', 7))
            restock_days = int(request.form.get('restock_days', 14))
            if inventory_file:
                inventory_file.save('store_inventory.csv')
        else:
            data = request.get_json(force=True, silent=True) or {}
            user_id = data.get('user_id')
            tolerance = int(data.get('tolerance', 7))
            restock_days = int(data.get('restock_days', 14))
        if not user_id:
            return jsonify({'message': 'User ID is required.'}), 400

        result = run_inventory_check(tolerance=tolerance, restock_days=restock_days)
        # Save inventory and restock actions for this user
        save_user_inventory(user_id, result.get("inventory", []))
        save_user_restock_actions(user_id, result.get("restock_actions", []))
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /run_inventory: {e}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/api/inventory', methods=['GET'])
def api_inventory():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400
    try:
        inventory = load_user_inventory(user_id)
        return jsonify({'inventory': inventory})
    except Exception as e:
        logger.error(f"Error loading inventory: {e}")
        return jsonify({'message': 'Failed to load inventory.'}), 500

@app.route('/api/restock_actions', methods=['GET'])
def api_restock_actions():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400
    try:
        restock_actions = load_user_restock_actions(user_id)
        return jsonify({'restock_actions': restock_actions})
    except Exception as e:
        logger.error(f"Error loading restock actions: {e}")
        return jsonify({'message': 'Failed to load restock actions.'}), 500

# --------- Main ---------
def main():
    df = pd.read_csv(CSV_PATH)
    dataset = RevenueDataset(df, SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_size = len(dataset.feature_cols)
    model = RevenueRNN(input_size=input_size, hidden_size=HIDDEN_SIZE)
    train_model(model, dataloader, EPOCHS, LEARNING_RATE)

    torch.save(model.state_dict(), "revenue_rnn.pth")
    print("✅ Model trained and saved as revenue_rnn.pth")

    forecast_next_month(model, dataset)

# --------- API Endpoint ---------
@app.route('/')
def index():
    return send_file("static/pages/home.html")

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/process_revenue')
def process_revenue_route():
    try:
        main()
        return jsonify({'message': 'Revenue processed successfully!'})
    except Exception as e:
        logger.error(f"Error in /process_revenue: {e}\n{traceback.format_exc()}")
        return jsonify({'message': f'Error: {str(e)}'})

@app.route('/run_all')
def run_all():
    try:
        # Define the environment variables
        env = os.environ.copy()

        # Run finance_advice.py
        process = subprocess.run(
            ['python', 'finance_advice.py'],
            check=False,
            capture_output=True,
            text=True,
            env=env  # Pass the environment variables
        )

        if process.returncode != 0:
            logger.error(f"finance_advice.py failed with return code {process.returncode}")
            logger.error(f"stdout: {process.stdout}")
            logger.error(f"stderr: {process.stderr}")
            return jsonify({'message': f'Error running finance_advice.py: {process.stderr}'})

        # Train the model
        df = pd.read_csv(CSV_PATH)
        dataset = RevenueDataset(df, SEQ_LENGTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        input_size = len(dataset.feature_cols)
        model = RevenueRNN(input_size=input_size, hidden_size=HIDDEN_SIZE)
        train_model(model, dataloader, EPOCHS, LEARNING_RATE)

        torch.save(model.state_dict(), "revenue_rnn.pth")
        print("✅ Model trained and saved as revenue_rnn.pth")

        # Run the forecasting logic
        forecast_next_month(model, dataset)

        return jsonify({'message': 'Both scripts ran successfully!'})
    except FileNotFoundError as e:
        logger.error(f"finance_advice.py not found: {e}")
        return jsonify({'message': f'Error: finance_advice.py not found.'})
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
        return jsonify({'message': f'Error: {str(e)}'})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400

    revenue_file = request.files.get('revenue_csv')
    past_data_file = request.files.get('past_data_csv')
    revenue_json = request.form.get('revenue_json')
    past_data_json = request.form.get('past_data_json')

    if not (revenue_file or revenue_json) or not (past_data_file or past_data_json):
        return jsonify({'message': 'You must provide either a revenue CSV or manual revenue data, and a past data CSV or manual past data.'}), 400

    try:
        # Handle revenue data
        if revenue_file:
            revenue_file.save(get_user_data_path(user_id, 'simulated_revenue.csv'))
        elif revenue_json:
            try:
                revenue_data = json.loads(revenue_json)
                df = pd.DataFrame(revenue_data)
                df['Date'] = pd.to_datetime(df['Date'])
                df['DayOfWeek'] = df['DayOfWeek'].astype(str)
                df['Revenue'] = pd.to_numeric(df['Revenue'])
                df.to_csv(get_user_data_path(user_id, 'simulated_revenue.csv'), index=False)
            except Exception as e:
                logger.error(f"Error parsing manual revenue data: {e}")
                return jsonify({'message': 'Invalid manual revenue data.'}), 400

        # Handle past data
        if past_data_file:
            past_data_file.save(get_user_data_path(user_id, 'pricingdata.csv'))
        elif past_data_json:
            try:
                past_data = json.loads(past_data_json)
                df = pd.DataFrame(past_data)
                df['Date'] = pd.to_datetime(df['Date'])
                df['ItemID'] = df['ItemID'].astype(str)
                df['QuantitySold'] = pd.to_numeric(df['QuantitySold'])
                df['CostPerItem'] = pd.to_numeric(df['CostPerItem'])
                df['OriginalCostToManufacture'] = pd.to_numeric(df['OriginalCostToManufacture'])
                df.to_csv(get_user_data_path(user_id, 'pricingdata.csv'), index=False)
            except Exception as e:
                logger.error(f"Error parsing manual past data: {e}")
                return jsonify({'message': 'Invalid manual past data.'}), 400

        return jsonify({'message': 'Files uploaded successfully!'})
    except Exception as e:
        logger.error(f"Error saving files: {e}")
        return jsonify({'message': 'Failed to save files.'}), 500

@app.route('/predict_revenue', methods=['POST'])
def predict_revenue():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400
    try:
        dataset = RevenueDataset(user_id, SEQ_LENGTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        input_size = len(dataset.feature_cols)
        model = RevenueRNN(input_size=input_size, hidden_size=HIDDEN_SIZE)
        train_model(model, dataloader, EPOCHS, LEARNING_RATE)

        torch.save(model.state_dict(), "revenue_rnn.pth")
        forecast_next_month(model, dataset)
        # Run finance_advice.py to generate df_revised.csv
        process = subprocess.run(
            ['python', 'finance_advice.py'],
            check=False,
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            logger.error(f"finance_advice.py failed: {process.stderr}")
            return jsonify({'message': f'Error running finance_advice.py: {process.stderr}'}), 500
        # Return the df_revised.csv file
        return send_file("df_revised.csv", as_attachment=True)
    except Exception as e:
        logger.error(f"Error in /predict_revenue: {e}\n{traceback.format_exc()}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/df_revised.csv')
def download_df_revised():
    path = "df_revised.csv"
    if not os.path.exists(path):
        return "df_revised.csv not found.", 404
    return send_file(path, as_attachment=True)

@app.route('/get_advice', methods=['POST'])
def get_advice():
    try:
        # Assumes predicted_revenue.csv has already been generated by /predict_revenue
        # Get user parameters from request
        params = request.get_json(force=True, silent=True) or {}
        env = os.environ.copy()
        # Set environment variables for finance_advice.py
        for key, env_key in [
            ("profit_margin", "PROFIT_MARGIN"),
            ("money_starting", "MONEY_STARTING"),
            ("interest_rate", "INTEREST_RATE"),
            ("loan_term_days", "LOAN_TERM_DAYS"),
            ("min_cash_reserve_ratio", "MIN_CASH_RESERVE_RATIO"),
        ]:
            if key in params:
                env[env_key] = str(params[key])
        # Pass expenses as JSON string
        if "expenses" in params:
            env["EXPENSES_JSON"] = json.dumps(params["expenses"])

        # Only run finance_advice.py and pricingoffical.py, do NOT re-run prediction
        process1 = subprocess.run(
            ['python', 'finance_advice.py'],
            check=False,
            capture_output=True,
            text=True,
            env=env
        )
        if process1.returncode != 0:
            logger.error(f"finance_advice.py failed: {process1.stderr}")
            return jsonify({'message': f'Error running finance_advice.py: {process1.stderr}'}), 500

        process2 = subprocess.run(
            ['python', 'pricingoffical.py'],
            check=False,
            capture_output=True,
            text=True
        )
        if process2.returncode != 0:
            logger.error(f"pricingoffical.py failed: {process2.stderr}")
            return jsonify({'message': f'Error running pricingoffical.py: {process2.stderr}'}), 500

        # Read advice summary
        advice_text = ""
        advice_path = "suggested_price_changes_relative.csv"
        summary_path = "advice_summary.txt"
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                advice_text = f.read()
        else:
            advice_text = "No summary available."

        if not os.path.exists(advice_path):
            return jsonify({'message': 'Advice file not found.'}), 500

        # Return JSON with advice and a download endpoint for the CSV
        return jsonify({
            'advice': advice_text,
            'csv_download_url': '/download_advice_csv'
        })
    except Exception as e:
        logger.error(f"Error in /get_advice: {e}\n{traceback.format_exc()}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/download_advice_csv')
def download_advice_csv():
    advice_path = "suggested_price_changes_relative.csv"
    if not os.path.exists(advice_path):
        return "Advice file not found.", 404
    return send_file(advice_path, as_attachment=True)

@app.route('/<page>.html')
def serve_html_page(page):
    # Serve any HTML file from static/pages/ (e.g., /jobapp.html, /admanager.html)
    return send_file(f'static/pages/{page}.html')

@app.route('/calendar.html')
def serve_calendar_html():
    return send_file('static/pages/calendar.html')

@app.route('/api/expenses', methods=['GET', 'POST'])
def api_expenses():
    user_id = request.args.get('user_id') if request.method == 'GET' else (request.json.get('user_id') if request.is_json else request.form.get('user_id'))
    if not user_id:
        return jsonify({'message': 'User ID is required.'}), 400

    if request.method == 'GET':
        try:
            expenses = load_user_expenses(user_id)
            return jsonify({'expenses': expenses})
        except Exception as e:
            logger.error(f"Error loading expenses: {e}")
            return jsonify({'message': 'Failed to load expenses.'}), 500

    if request.method == 'POST':
        try:
            if request.is_json:
                data = request.get_json(force=True, silent=True)
            else:
                data = request.form
            expenses = data.get('expenses')
            if isinstance(expenses, str):
                expenses = json.loads(expenses)
            if not isinstance(expenses, list):
                return jsonify({'message': 'Expenses must be a list.'}), 400
            save_user_expenses(user_id, expenses)
            return jsonify({'message': 'Expenses saved.'})
        except Exception as e:
            logger.error(f"Error saving expenses: {e}")
            return jsonify({'message': 'Failed to save expenses.'}), 500

# --------- Job Application Analyzer Routes and Logic ---------
@app.route("/jobapp")
def jobapp_index():
    return send_file("static/pages/jobapp.html")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def analyze_application(app_text, job_description):
    prompt = f"""
You are an expert hiring assistant. Here is a job description:

{job_description}

Now analyze the following candidate's application:

{app_text}

Return the following:
1. Pros
2. Cons
3. Summary
4. Overall fit score (0-10)
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def extract_section(text, section_name, stop_section=None):
    """
    Extracts the section content between section_name and stop_section.
    If stop_section is None, extracts until the next section header or end of text.
    """
    # Normalize line endings and remove extra whitespace
    text = re.sub(r'\r\n?', '\n', text)
    # Build regex for section start (e.g., "Pros:" or "Pros")
    section_pattern = rf"{section_name}\s*:?"
    # Build regex for stop section (e.g., "Cons:" or "Cons")
    if stop_section:
        stop_pattern = rf"{stop_section}\s*:?"
        # Match section_name at line start, capture everything until stop_section at line start
        pattern = re.compile(
            rf"^{section_pattern}\s*\n(.*?)(?:^\s*{stop_pattern}\s*\n|$)",
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
    else:
        # Until next ALL CAPS section or end of text
        pattern = re.compile(
            rf"^{section_pattern}\s*\n(.*?)(?:^\s*[A-Z][a-zA-Z ]{{1,30}}:\s*\n|$)",
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
    match = pattern.search(text)
    return match.group(1).strip() if match else "Not found"

def extract_score(text):
    match = re.search(r"(final score|fit score)\s*[:\-]?\s*(\d+(\.\d+)?)", text, re.IGNORECASE)
    return float(match.group(2)) if match else 0

def extract_summary(text):
    # Extract the summary section between "Summary" and "Final score" or "fit score"
    summary = extract_section(text, "Summary", "Final score")
    if summary == "Not found":
        summary = extract_section(text, "Summary", "fit score")
    return summary

def extract_final_score_justification(text):
    # Extract the justification sentence after "Final score"
    match = re.search(r"Final score\s*[:\-]?\s*\d+(\.\d+)?[^\n]*\n(.*)", text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    # Fallback: try to get the line after "fit score"
    match = re.search(r"fit score\s*[:\-]?\s*\d+(\.\d+)?[^\n]*\n(.*)", text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return ""

def process_all_applications(job_description, folder_path):
    results = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            text = extract_text_from_pdf(file_path)
            analysis = analyze_application(text, job_description)
            score = extract_score(analysis)
            results.append({
                "file": file,
                "score": score,
                "analysis": analysis
            })

    best = max(results, key=lambda x: x['score'], default={"file": "None", "score": 0})
    return results, best

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()  # Temporary folder for uploads

@app.route("/analyze", methods=["POST"])
def jobapp_analyze():
    job_description = request.form.get("job_description", "")
    files = request.files.getlist("applications")
    if not job_description or not files or files[0].filename == "":
        flash("Please provide a job description and at least one PDF.")
        return redirect(url_for("jobapp_index"))

    upload_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
    logger.info(f"Upload directory created: {upload_dir}")
    saved_files = []
    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            saved_files.append(filename)

    results, best = process_all_applications(job_description, upload_dir)

    # Pass only the relative upload_dir name for download links
    rel_upload_dir = os.path.relpath(upload_dir, app.config['UPLOAD_FOLDER'])

    return render_template(
        "jobapp_results.html",
        results=results,
        best=best,
        upload_dir=rel_upload_dir
    )

@app.route("/jobapp/download/<path:filename>")
def jobapp_download_file(filename):
    try:
        logger.info(f"Attempting to download file: {filename}")
        # filename is like: <rel_upload_dir>/annotated_<file>
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(full_path):
            logger.error(f"File not found: {full_path}")
            flash("File not found.")
            return redirect(url_for("jobapp_index"))
        return send_file(full_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        flash("Error downloading file.")
        return redirect(url_for("jobapp_index"))

# --------- Advertising Opportunities Routes and Logic ---------
@app.route("/admanager")
def admanager_index():
    return render_template("pages/admanager.html")

@app.route("/api/test", methods=["GET"])
def api_test():
    return jsonify({"success": True, "message": "API is working."})

def find_ads_google_custom_search(business_type, county, state, api_key, cx):
    query = f"Small advertising for {business_type} near {county}, {state}"
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": 5,
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return {"success": False, "error": f"Google API error: {response.status_code}"}
        try:
            data = response.json()
        except Exception as e:
            logging.error(f"Error parsing JSON from Google API: {e}")
            return {"success": False, "error": "Invalid JSON from Google API"}
        ads = []
        for item in data.get("items", []):
            ads.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            })
        return {"success": True, "ads": ads}
    except Exception as e:
        logging.error(f"Exception in find_ads_google_custom_search: {e}")
        return {"success": False, "error": f"Exception: {str(e)}"}

@app.route('/api/find_ads', methods=['POST'])
def api_find_ads():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid or missing JSON in request"}), 400
        business_type = data.get('business_type')
        county = data.get('county')
        state = data.get('state')
        if not business_type or not county or not state:
            return jsonify({"success": False, "error": "Missing business_type, county, or state"}), 400
        api_key = os.getenv("GOOGLE_API_KEY")  # Load from environment variables
        cx = os.getenv("GOOGLE_CX")  # Load from environment variables
        if not api_key or not cx:
            return jsonify({"success": False, "error": "API key or CX is missing"}), 500
        result = find_ads_google_custom_search(business_type, county, state, api_key, cx)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Exception in api_find_ads: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

# --------- Lawyer Tool Routes and Logic ---------
@app.route("/lawyertipeting")
def lawyertipeting_index():
    return render_template("pages/lawyertipeting.html")

@app.route("/api/lawyer_simplify", methods=["POST"])
def api_lawyer_simplify():
    try:
        from lawyertipeting import read_contract, simplify_contract
        file = request.files.get("contract")
        if not file or file.filename == "":
            return jsonify({"success": False, "error": "No file uploaded."}), 400
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        contract_text = read_contract(file_path)
        simplified = simplify_contract(contract_text)
        shutil.rmtree(temp_dir)
        return jsonify({"success": True, "simplified": simplified})
    except Exception as e:
        logger.error(f"Error in /api/lawyer_simplify: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/calendar')
def api_calendar():
    try:
        month = int(request.args.get('month', datetime.now().month))
        year = int(request.args.get('year', datetime.now().year))
        data = get_calendar_data(month, year)
        return jsonify({'calendar': data})
    except Exception as e:
        logger.error(f"Error in /api/calendar: {e}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/api/calendar/add_event', methods=['POST'])
def api_calendar_add_event():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'success': False, 'error': 'Missing JSON'}), 400
        date_str = data.get('date')
        title = data.get('title', 'Restock')
        desc = data.get('desc', '')
        if not date_str:
            return jsonify({'success': False, 'error': 'Missing date'}), 400
        # Add event to billcalendar.events
        billcalendar.add_event(date_str, title, desc)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error in /api/calendar/add_event: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calendar/remove_event', methods=['POST'])
def api_calendar_remove_event():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'success': False, 'error': 'Missing JSON'}), 400
        date_str = data.get('date')
        idx = data.get('idx')
        if date_str is None or idx is None:
            return jsonify({'success': False, 'error': 'Missing date or idx'}), 400
        billcalendar.remove_event(date_str, idx)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error in /api/calendar/remove_event: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --------- OCR Receipt Routes and Logic ---------
def extract_text_from_pdf_filelike(filelike):
    """Extract text from a PDF file-like object using PyMuPDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        filelike.seek(0)
        tmp.write(filelike.read())
        tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        filelike.seek(0)
        tmp.write(filelike.read())
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    finally:
        os.remove(tmp_path)

def parse_expenses_from_openai(text):
    """
    Use OpenAI to extract expenses from receipt text.
    Returns a list of dicts: [{name, cost, date_of_purchase, other_possible_dates, urgency=0}, ...]
    """
    prompt = f"""
You are an expert at reading receipts and extracting expense data for accounting.
Given the following receipt text, extract all expenses as a JSON array.
Each expense should have:
- name (string, e.g. 'Staples', 'Walmart', or the item/service)
- cost (number, in dollars)
- purchase_date (string, YYYY-MM-DD, use today's date if not found)
- other_possible_dates (array of strings, can be empty)
- urgency (integer 0-5, guess 0 if not clear)

Receipt text:
{text}

Return ONLY the JSON array, no explanation.
Example:
[
  {{"name": "Staples", "cost": 42.50, "due_date": "2024-06-01", "other_possible_dates": [], "urgency": 1}}
]
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0
    )
    # Try to extract JSON from the response
    import json
    import re
    content = response.choices[0].message.content
    # Find the first JSON array in the response
    match = re.search(r'(\[\s*{.*}\s*\])', content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = content
    try:
        expenses = json.loads(json_str)
        if isinstance(expenses, dict):
            expenses = [expenses]
        return expenses
    except Exception:
        return []

@app.route('/ocr_receipt', methods=['POST'])
def ocr_receipt():
    try:
        file = request.files.get('receipt_pdf')
        if not file or file.filename == "":
            return jsonify({"message": "No PDF uploaded."}), 400
        text = extract_text_from_pdf_filelike(file)
        expenses = parse_expenses_from_openai(text)
        # Optionally, save to user's expenses if user_id provided
        user_id = request.form.get('user_id')
        if user_id and expenses:
            existing = load_user_expenses(user_id)
            # Only add if not already present (by name+cost+due_date)
            for exp in expenses:
                if not any(
                    e.get('name') == exp.get('name') and
                    float(e.get('cost', 0)) == float(exp.get('cost', 0)) and
                    e.get('due_date') == exp.get('due_date')
                    for e in existing
                ):
                    existing.append(exp)
            save_user_expenses(user_id, existing)
        return jsonify({"expenses": expenses})
    except Exception as e:
        logger.error(f"Error in /ocr_receipt: {e}\n{traceback.format_exc()}")
        return jsonify({"message": "Failed to process receipt."}), 500

# --------- Employee Class ---------
class Expence:
    def __init__(self, cost: float, due_date: date, other_possible_dates: List[date], urgency: int):
        self.cost = cost
        self.due_date = due_date
        self.other_possible_dates = other_possible_dates
        self.urgency = urgency

class Employee:
    def __init__(self, pay: float, time_at_company: float, position: str, name: str, pay_frequency: str = "semimonthly"):
        self.pay = pay
        self.time_at_company = time_at_company
        self.position = position
        self.name = name
        self.pay_frequency = pay_frequency or "semimonthly"

    def get_paid(self, start_date: date, end_date: date) -> List[Expence]:
        pay_dates = []
        current = start_date

        # Determine pay interval and number of periods per year
        freq = self.pay_frequency.lower()
        if freq == "weekly":
            interval = 7
            periods = 52
        elif freq == "biweekly":
            interval = 14
            periods = 26
        elif freq == "monthly":
            interval = None  # Special handling
            periods = 12
        else:  # semimonthly (default)
            interval = None  # Special handling
            periods = 24

        # Generate pay dates for a year
        if freq in ("weekly", "biweekly"):
            while current <= end_date:
                pay_dates.append(current)
                current += timedelta(days=interval)
        elif freq == "monthly":
            # Pay on the same day each month as start_date
            for i in range(12):
                month = (start_date.month + i - 1) % 12 + 1
                year = start_date.year + ((start_date.month + i - 1) // 12)
                day = min(start_date.day, 28)  # Avoid invalid dates
                pay_dates.append(date(year, month, day))
        else:  # semimonthly: 1st and 15th of each month
            d = start_date
            while d <= end_date:
                if d.day == 1 or d.day == 15:
                    pay_dates.append(d)
                d += timedelta(days=1)

        # Remove duplicates and sort
        pay_dates = sorted(set(pay_dates))

        # Calculate pay per period
        pay_per_cycle = self.pay / periods if periods else self.pay / 24
        expenses = []
        for d in pay_dates:
            expenses.append(Expence(cost=pay_per_cycle, due_date=d, other_possible_dates=[], urgency=1))
        return expenses

    def __repr__(self):
        return (f"{self.name} is an employee at your company who gets paid {self.pay} "
                f"a year for being a {self.position} for {self.time_at_company} years, paid {self.pay_frequency}")

# --------- Run ---------
if __name__ == "__main__":
    try:
        initialize_app()
        # Create upload folder if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        logger.info("Starting server...")
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        logger.info("Starting server...")
        logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")

        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

        logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")

        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
