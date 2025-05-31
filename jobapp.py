import os
import fitz  # PyMuPDF
import re
import openai
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import tempfile
import shutil
api_key = os.getenv("OPENAI_API_KEY")  # Ensure you have set this environment variable
openai.api_key = api_key  # Set the API key for OpenAI

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
        model="gpt-4-",
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()  # Temporary folder for uploads

from flask import send_file

@app.route("/")
def index():
    return render_template("pages/jobapp.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    job_description = request.form.get("job_description", "")
    files = request.files.getlist("applications")
    if not job_description or not files or files[0].filename == "":
        flash("Please provide a job description and at least one PDF.")
        return redirect(url_for("index"))

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
        "pages/jobapp_results.html",
        results=results,
        best=best,
        upload_dir=upload_dir
    )

@app.route("/download/<path:filename>")
def download_file(filename):
    dir_name = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    return send_from_directory(dir_name, base_name, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)