from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime
import base64
import requests

app = Flask(__name__)

# REMOVED GITHUB TOKEN FIX THIS BYU JSUT MAKING IT PUBLIC AND TAILOR CODE TO PUBLIC GITHUB UPLAOD

@app.route('/receive_symbols', methods=['POST'])
def receive_symbols():
    data = request.get_json()
    if not data or 'symbols' not in data:
        return jsonify({'message': 'No symbols provided'}), 400

    symbols = data['symbols']
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"symbols_{now}.csv"

    # Save symbols into a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol'])  # Write header
        for symbol in symbols:
            writer.writerow([symbol])  # Write each symbol as a new row

    print(f"Received symbols: {symbols}")
    
    # Upload to GitHub
    upload_to_github(filename)

    return jsonify({'message': f'Symbols received and saved successfully in {filename}'}), 200


def upload_to_github(filename):
    # Read the CSV content
    with open(filename, "rb") as file:
        content = base64.b64encode(file.read()).decode()

    # GitHub API URL for uploading files
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"

    # Prepare the request payload
    payload = {
        "message": f"Add {filename}",
        "content": content,
        "branch": GITHUB_BRANCH,
    }


    # PROBABLY WONT NEED THIS NO MORE
    # Prepare the headers with your GitHub token
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }

    # Send a PUT request to GitHub API
    response = requests.put(url, json=payload, headers=headers)

    if response.status_code == 201:
        print(f"File {filename} successfully uploaded to GitHub.")
    else:
        print(f"Failed to upload {filename} to GitHub: {response.status_code} - {response.text}")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
