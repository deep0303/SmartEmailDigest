from flask import Flask, render_template, request, jsonify
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import base64
import os
import pickle
import google.generativeai as genai
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
import threading

app = Flask(__name__)

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
TOKEN_FILE = 'token.pickle'

# Gemini API setup
GEMINI_API_KEY = ''  # Replace with your real API key from Google AI Studio
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Lock for rate limiting
rate_limit_lock = threading.Lock()
last_call_time = 0
MIN_DELAY = 0.5  # 500ms to stay within ~15 requests/minute

def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if not os.path.exists('credentials.json'):
            return None, "Error: 'credentials.json' not found. Download it from Google Cloud Console."
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds), None

def enforce_rate_limit():
    with rate_limit_lock:
        global last_call_time
        current_time = time.time()
        time_to_wait = max(0, MIN_DELAY - (current_time - last_call_time))
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        last_call_time = time.time()

def classify_and_summarize_email(message_id, service, combined_prompt):
    enforce_rate_limit()
    try:
        msg = service.users().messages().get(userId='me', id=message_id, format='full').execute()
        payload = msg['payload']
        headers = payload['headers']
        
        try:
            subject = next(h['value'] for h in headers if h['name'] == 'Subject')
        except StopIteration:
            subject = "No Subject"
        
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown Sender")
        
        text = ""
        if 'data' in payload.get('body', {}):
            text = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
        elif 'parts' in payload and payload['parts']:
            for part in payload['parts']:
                if 'data' in part.get('body', {}):
                    text = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                    break
        if not text:
            text = f"Subject suggests {subject.lower()}; no body content available."
        
        text = re.sub(r'\s+', ' ', text).strip()
        full_content = f"Subject: {subject}\nFrom: {sender}\n{text[:2000]}"
        
        # Enhanced prompt with strict instructions
        combined_query = (
            f"{combined_prompt}. Classify strictly as 'Urgent' if the email requires immediate action (e.g., deadlines, errors) or 'Non-Urgent' otherwise. "
            f"Return only the classification label (e.g., 'Urgent'), followed by a pipe '|', then a concise summary (30-50 words) focusing on main intent or actions. "
            f"Examples: 'Urgent|Fix server error by EOD' or 'Non-Urgent|Read article on AI trends'.\n\n"
            f"{full_content}"
        )
        response = model.generate_content(combined_query)
        result = response.text.strip().split('|', 1)
        classification = result[0].strip() if result and result[0] in ['Urgent', 'Non-Urgent'] else "Non-Urgent"
        summary = result[1].strip() if len(result) > 1 else "Unable to generate summary due to insufficient content."
        
        return subject, sender, classification, summary
    except Exception as e:
        text = text[:200] if 'text' in locals() else "No content available"
        if "429" in str(e):
            return subject if 'subject' in locals() else "Unknown", "Unknown Sender", "Quota Exceeded", f"{text[:100]}..."
        return subject if 'subject' in locals() else "Unknown", "Unknown Sender", "Error", f"{text[:200]} (Error: {str(e)})"

def send_digest(summaries, service):
    formatted_digest = "Daily Email Digest\n" + "="*50 + "\n"
    for i, (subject, sender, classification, summary) in enumerate(summaries, 1):
        wrapped_subject = textwrap.fill(subject, width=80)
        wrapped_sender = textwrap.fill(f"From: {sender}", width=80)
        wrapped_classification = textwrap.fill(f"Classification: {classification}", width=80)
        wrapped_summary = textwrap.fill(f"Summary: {summary}", width=80)
        formatted_digest += (
            f"Email {i}:\n"
            f"{wrapped_subject}\n"
            f"{wrapped_sender}\n"
            f"{wrapped_classification}\n"
            f"{wrapped_summary}\n"
            f"{'-'*50}\n"
        )
    message = MIMEText(formatted_digest)
    message['to'] = 'sdeepika6232@gmail.com'  # Replace with your email
    message['from'] = 'me'
    message['subject'] = 'Your Daily Email Digest'
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId='me', body={'raw': raw}).execute()
    return formatted_digest

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_digest', methods=['POST'])
def generate_digest():
    service, error = get_gmail_service()
    if error:
        return render_template('result.html', error=error)
    
    combined_prompt = request.form.get('combined_prompt', 'Classify email as Urgent or Non-Urgent and summarize in 30-50 words focusing on actions').strip()
    if not combined_prompt:
        return render_template('result.html', error="Error: Please provide a combined classification and summarization prompt.<br>Try 'Classify as Urgent or Non-Urgent and summarize in 30-50 words focusing on actions'.")
    
    results = service.users().messages().list(userId='me', q='is:unread', maxResults=5).execute()
    messages = results.get('messages', [])
    if not messages:
        return render_template('result.html', error="No unread emails found.")
    
    summaries = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_msg = {executor.submit(classify_and_summarize_email, msg['id'], service, combined_prompt): msg for msg in messages}
        for future in future_to_msg:
            subject, sender, classification, summary = future.result()
            summaries.append((subject, sender, classification, summary))
    
    digest = send_digest(summaries, service)
    for msg in messages:
        service.users().messages().modify(userId='me', id=msg['id'], body={'removeLabelIds': ['UNREAD']}).execute()
    
    return render_template('result.html', digest=digest)



if __name__ == '__main__':
    app.run(debug=True)
