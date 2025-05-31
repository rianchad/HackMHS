import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Optional: load from .env file
from dotenv import load_dotenv
load_dotenv()

  # Or paste key directly

def read_contract(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.pdf'):
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        raise ValueError("Unsupported file type. Use .txt or .pdf")

def simplify_contract(text, max_words=1000):
    prompt = (
        "Simplify the following legal or contract text into plain English, preserving the important terms."
        " Give advice on whether there are hidden risks or obligations. "
        "Be concise, accurate, and readable: \n\n"
        f"{text[:max_words*5]}"  # limit input size
    )

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.5,
    max_tokens=1000)
    return response.choices[0].message.content

# ---------- Example Usage ----------
if __name__ == "__main__":
    file_path = "contract.pdf"  # or "contract.txt"
    original_text = read_contract(file_path)
    simplified = simplify_contract(original_text)

    print("📝 Simplified Version:\n")
    print(simplified)
