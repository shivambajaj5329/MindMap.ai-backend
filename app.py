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
        # Using OpenAI to analyze the transcript
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following meeting transcript, provide sentiment analysis, and key insights in string format with keys summary, sentiment, and insights:\n\n{transcript}\nAdditional context: {context} \n dont include the word JSON or string anywhere. property name enclosed in double quotes enclude the output in curly brackets. Insights should be an array of strings of the insights"}
            ],
            max_tokens=10000
        )
        result = response.choices[0].message['content'].strip()
        
        # Parse the JSON response
        analysis = json.loads(result)
        summary = analysis.get("summary", "No summary provided.")
        sentiment = analysis.get("sentiment", {})
        insights = analysis.get("insights", [])
        
        return summary, sentiment, insights
    except Exception as e:
        print(f"Error in analyzing transcript: {e}")
        return "Error: Could not analyze transcript.", {}, []

def analyze_transcript_gemini(transcript, context):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Summarize the following meeting transcript, provide sentiment analysis (Positive, Negative, Neutral one of these), and key informative insights:\n\n{transcript}\nAdditional context: {context} \n dont include the word JSON or string anywhere. property name enclosed in double quotes enclude the output in curly brackets. Insights should be an array of strings of the insights. Please make sure insights a list of strings"
        response = model.generate_content(prompt)

        # print(response)
        
        # Parse the response assuming it's a structured JSON
        analysis = json.loads(response.text)
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
