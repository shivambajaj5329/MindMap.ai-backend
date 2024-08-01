from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
import json
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'file' not in request.files or 'context' not in request.form or 'model' not in request.form:
        return jsonify({'error': 'No file part, context, or model specified'}), 400
    
    file = request.files['file']
    context = request.form['context']
    model = request.form['model']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process text file
        if file.filename.endswith('.txt'):
            with open(file_path, 'r') as f:
                transcription = f.read()
        else:
            # Placeholder for audio transcription
            transcription = transcribe_audio(file_path)
        
        # Analyze the transcription using the chosen model
        if model == 'openai':
            summary, sentiment, insights = analyze_transcript_openai(transcription, context)
        elif model == 'gemini':
            summary, sentiment, insights = analyze_transcript_gemini(transcription, context)
        else:
            return jsonify({'error': 'Invalid model specified'}), 400

        # Return JSON response
        return jsonify({
            "summary": summary,
            "sentiment": sentiment,
            "insights": insights
        }), 200

def transcribe_audio(file_path):
    # Simulated transcription
    return "Simulated transcription of the audio."

def analyze_transcript_openai(transcript, context):
    try:

        prompt = f"""
            You are a highly efficient and intelligent bot.

            Task: Summarize the following meeting transcript, provide sentiment analysis, and key insights.

            Requirements:
            1. **Summary**:
            - Provide a concise summary of the main points discussed in the meeting.

            2. **Sentiment Analysis**:
            - Analyze the overall sentiment of the meeting, indicating whether it was positive, negative, or neutral.

            3. **Key Insights**:
            - Extract and list key insights from the meeting. Insights should be an array of strings, each string representing a distinct insight.

            Formatting Rules:
            - Do not include the words "JSON" or "string" anywhere in the response.
            - Enclose property names in double quotes.
            - Enclose the entire output in curly brackets.
            - Ensure insights are formatted as an array of strings.

            Context:
            - Additional context: {context}

            Transcript:
            - Following is the meeting transcript:

            \n\n{transcript}"""        
        result = prompt_openai(prompt=prompt)
        
        # Parse the JSON response
        analysis = json.loads(result)
        summary = analysis.get("summary", "No summary provided.")
        sentiment = analysis.get("sentiment", {})
        insights = analysis.get("insights", [])
        
        return summary, sentiment, insights
    except Exception as e:
        print(f"Error in analyzing transcript: {e}")
        return "Error: Could not analyze transcript.", {}, []
    


def prompt_gemini(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    return response.text


def prompt_openai(prompt):

    messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}"}
                ]
    response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages= messages,
            max_tokens=10000
            )
    result = response.choices[0].message['content'].strip()
    return result

@app.route('/deidentify', methods=['POST'])
def deidentify():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'No file or model specified'}), 400
    
    file = request.files['file']
    model = request.form['model']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        if file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            # Placeholder for audio transcription
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            text = transcribe_audio(file_path)
        
        deidentified_text = deidentify_text(text, model)

        return jsonify({
            "deidentifiedText": deidentified_text
        }), 200

def deidentify_text(text, model):
    try:
        prompt = f"""You are a good bot.

Task: Please anonymize the following clinical note.

Task Specific Rules:
1. **Names**:
   - Replace any strings that might be a name, acronym, or initials, including patients' names, doctors' names, M.D., Dr., and medical staff names, with "[REDACTED]".
   - Replace the therapist and client names specifically with 'therapist' and 'client' respectively.

2. **Locations**:
   - Replace any strings that might be a location or address, such as "3970 Longview Drive", clinic, and hospital names, with "[REDACTED]".

3. **Ages**:
   - Replace any strings that indicate age, such as "something years old" or "age 37", with "[REDACTED]".

4. **Dates and IDs**:
   - Replace any dates, IDs, or ID-like strings, and record dates with "[REDACTED]".

5. **Professions**:
   - Replace any mentions of professions, such as "manager", with "[REDACTED]".

6. **Contact Information**:
   - Replace any contact information, including phone numbers, email addresses, and other contact details, with "[REDACTED]".

Formatting Rules:
- Ensure the text has a double newline charecters ("\n\n") after every dialog spoken by either the therapist or the client.

Instructions:
- Please do not remove any information that is not specified to be redacted.
- Ensure the integrity of the clinical note is maintained, with only the specified information redacted.

Following is the text. Please ensure all specified information is redacted according to the rules provided:

\n\n{text}"""
        if model.lower() == "gemini":
            resp = prompt_gemini(prompt)
        elif model.lower() == "openai":
            resp = prompt_openai(prompt)
        else:
            return "Error: Invalid model specified"
        
        return resp
    except Exception as e:
        print(f"Error in deidentifying text: {e}")
        return "Error: Could not deidentify text."

def analyze_transcript_gemini(transcript, context):
    try:
        prompt = f"""
            You are a highly efficient and intelligent bot.

            Task: Summarize the following meeting transcript, provide sentiment analysis, and key insights.

            Requirements:
            1. **Summary**:
            - Provide a concise summary of the main points discussed in the meeting.

            2. **Sentiment Analysis**:
            - Analyze the overall sentiment of the meeting, indicating whether it was positive, negative, or neutral.

            3. **Key Insights**:
            - Extract and list key insights from the meeting. Insights should be an array of strings, each string representing a distinct insight.

            Formatting Rules:
            - Do not include the words "JSON" or "string" anywhere in the response.
            - Enclose property names in double quotes.
            - Enclose the entire output in curly brackets.
            - Ensure insights are formatted as an array of strings.

            Context:
            - Additional context: {context}

            Transcript:
            - Following is the meeting transcript:

            \n\n{transcript}"""
        response = prompt_gemini(prompt)

        # print(response)
        
        # Parse the response assuming it's a structured JSON
        analysis = json.loads(response)
        summary = analysis.get("summary", "No summary provided.")
        sentiment = analysis.get("sentiment", {})
        insights = analysis.get("insights", [])

        return summary, sentiment, insights
    except Exception as e:
        print(f"Error in analyzing transcript: {e}")
        return "Error: Could not analyze transcript.", {}, []

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=8080)
