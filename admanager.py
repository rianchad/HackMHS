import requests
import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)
api_key = os.getenv("GOOGLE_API_KEY")
cx = os.getenv("GOOGLE_CX")

@app.route("/")
def index():
    return "Flask backend is running."

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
        result = find_ads_google_custom_search(business_type, county, state, api_key, cx)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Exception in api_find_ads: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
