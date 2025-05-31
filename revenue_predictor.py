import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_file, send_from_directory, make_response, render_template, redirect, url_for, flash
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

# Load environment variables
load_dotenv()

app = Flask(__name__)
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

# --------- User Management ---------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# --------- Dataset Class ---------
class RevenueDataset(Dataset):
    def __init__(self, df, seq_length):
        self.seq_length = seq_length

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

        try:
            df['DayOfWeek'] = df['DayOfWeek'].astype('category').cat.codes
        except KeyError:
            print("Error: 'DayOfWeek' column not found in DataFrame.")
            raise

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
    return send_from_directory('static/pages', 'index.html')

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

        # Run HackMHSofficial.py
        process = subprocess.run(
            ['python', 'HackMHSofficial.py'],
            check=False,
            capture_output=True,
            text=True,
            env=env  # Pass the environment variables
        )

        if process.returncode != 0:
            logger.error(f"HackMHSofficial.py failed with return code {process.returncode}")
            logger.error(f"stdout: {process.stdout}")
            logger.error(f"stderr: {process.stderr}")
            return jsonify({'message': f'Error running HackMHSofficial.py: {process.stderr}'})

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
        logger.error(f"HackMHSofficial.py not found: {e}")
        return jsonify({'message': f'Error: HackMHSofficial.py not found.'})
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
        return jsonify({'message': f'Error: {str(e)}'})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    revenue_file = request.files.get('revenue_csv')
    past_data_file = request.files.get('past_data_csv')
    if not revenue_file or not past_data_file:
        return jsonify({'message': 'Both revenue and past data CSV files are required.'}), 400
    try:
        revenue_file.save(CSV_PATH)
        past_data_file.save('pricingdata.csv')
        return jsonify({'message': 'Files uploaded successfully!'})
    except Exception as e:
        logger.error(f"Error saving files: {e}")
        return jsonify({'message': 'Failed to save files.'}), 500

@app.route('/predict_revenue', methods=['POST'])
def predict_revenue():
    try:
        df = pd.read_csv(CSV_PATH)
        dataset = RevenueDataset(df, SEQ_LENGTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        input_size = len(dataset.feature_cols)
        model = RevenueRNN(input_size=input_size, hidden_size=HIDDEN_SIZE)
        train_model(model, dataloader, EPOCHS, LEARNING_RATE)

        torch.save(model.state_dict(), "revenue_rnn.pth")
        forecast_next_month(model, dataset)
        # Run HackMHSofficial.py to generate df_revised.csv
        process = subprocess.run(
            ['python', 'HackMHSofficial.py'],
            check=False,
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            logger.error(f"HackMHSofficial.py failed: {process.stderr}")
            return jsonify({'message': f'Error running HackMHSofficial.py: {process.stderr}'}), 500
        # Return the df_revised.csv file
        return send_file("df_revised.csv", as_attachment=True)
    except Exception as e:
        logger.error(f"Error in /predict_revenue: {e}\n{traceback.format_exc()}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        if not username or not password:
            return jsonify({'message': 'Username and password required.'}), 400
        users = load_users()
        if username in users:
            return jsonify({'message': 'Username already exists.'}), 409
        users[username] = password
        save_users(users)
        return jsonify({'message': 'Registration successful.'}), 200
    except Exception:
        return jsonify({'message': 'Server error.'}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        users = load_users()
        if username in users and users[username] == password:
            return jsonify({'message': 'Login successful.'}), 200
        else:
            return jsonify({'message': 'Invalid username or password.'}), 401
    except Exception:
        return jsonify({'message': 'Server error.'}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/pages/<path:filename>')
def pages_files(filename):
    return send_from_directory('static/pages', filename)

@app.route('/get_advice', methods=['POST'])
def get_advice():
    try:
        # Assumes predicted_revenue.csv has already been generated by /predict_revenue
        # Get user parameters from request
        params = request.get_json(force=True, silent=True) or {}
        env = os.environ.copy()
        # Set environment variables for HackMHSofficial.py
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

        # Only run HackMHSofficial.py and pricingoffical.py, do NOT re-run prediction
        process1 = subprocess.run(
            ['python', 'HackMHSofficial.py'],
            check=False,
            capture_output=True,
            text=True,
            env=env
        )
        if process1.returncode != 0:
            logger.error(f"HackMHSofficial.py failed: {process1.stderr}")
            return jsonify({'message': f'Error running HackMHSofficial.py: {process1.stderr}'}), 500

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
    return send_from_directory('static/pages', f'{page}.html')

# --------- Job Application Analyzer Routes and Logic ---------
@app.route("/jobapp")
def jobapp_index():
    return render_template("pages/jobapp.html")

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

def extract_section(text, section_name):
    pattern = re.compile(f"{section_name}:(.*?)(\n[A-Z][a-z]+:|\Z)", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else "Not found"

def extract_score(text):
    match = re.search(r"fit score.*?(\d+(\.\d+)?)", text, re.IGNORECASE)
    return float(match.group(1)) if match else 0

def annotate_pdf_with_pros_cons(original_pdf_path, pros, cons, output_path):
    doc = fitz.open(original_pdf_path)
    page = doc[0]
    note_text = f"Pros:\n{pros}\n\nCons:\n{cons}"
    page.insert_text((50, 50), note_text, fontsize=10, color=(0, 0, 1))
    doc.save(output_path)

def process_all_applications(job_description, folder_path):
    results = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            text = extract_text_from_pdf(file_path)
            analysis = analyze_application(text, job_description)

            pros = extract_section(analysis, "Pros")
            cons = extract_section(analysis, "Cons")
            summary = extract_section(analysis, "Summary")
            score = extract_score(analysis)

            output_path = os.path.join(folder_path, "annotated_" + file)
            annotate_pdf_with_pros_cons(file_path, pros, cons, output_path)

            results.append({
                "file": file,
                "summary": summary,
                "score": score,
                "pros": pros,
                "cons": cons
            })

    best = max(results, key=lambda x: x['score'])
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
    saved_files = []
    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            saved_files.append(filename)

    results, best = process_all_applications(job_description, upload_dir)

    return render_template(
        "pages/jobapp_results.html",  # Corrected path
        results=results,
        best=best,
        upload_dir=upload_dir
    )

@app.route("/jobapp/download/<path:filename>")
def jobapp_download_file(filename):
    dir_name = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    return send_from_directory(dir_name, base_name, as_attachment=True)

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

# --------- Run ---------
if __name__ == "__main__":
    # NOTE: Run this file with `python app.py` to start the backend server.
    # Do NOT use Live Server for this file; Live Server is for static HTML/JS only.
    app.run(host='0.0.0.0', port=5000, debug=True)
