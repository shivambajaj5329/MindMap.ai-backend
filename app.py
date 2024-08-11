from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
import openai
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
import json
import google.generativeai as genai
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.units import inch
import matplotlib
import logging
import re
from reportlab.lib.utils import ImageReader
matplotlib.use('Agg')


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
cors_origins = os.environ.get('CORS_ORIGINS', '').split(',')
CORS(app, resources={r"/*": {
    "origins": ["http://localhost:3000", "https://mind-map-ai-frontend.vercel.app"],
    "methods": ["GET", "POST", "PUT", "DELETE"],
    "allow_headers": ["Content-Type", "Authorization"]
}})
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

        prompt = f"""
            You are a highly efficient and intelligent bot.

            Task: Summarize the following meeting transcript, provide sentiment analysis, and key insights.

            Requirements:
            1. **summary**:
            - Provide a concise summary of the main points discussed in the meeting.

            2. **sentiment**:
            - Analyze the overall sentiment of the meeting, indicating whether it was Very Negative, Negative, Neutral, Positive, Very Positive.

            3. **insights**:
            - Extract and list key valuable insights from the meeting. Insights should be an array of strings, each string representing a distinct insight.

            Formatting Rules:
            - Do not include the words "JSON" or "string" anywhere in the response.
            - Enclose property names in double quotes.
            - Enclose the entire output in curly brackets.
            - Ensure insights are formatted as an array of strings.

            Context:
            - Additional context: {context}

            Transcript:
            - Following is the meeting transcript:

            \n\n{transcription}"""        
        # Analyze the transcription using the chosen model
        if model == 'openai':
            summary, sentiment, insights = analyze_transcript_openai(prompt)
        elif model == 'gemini':
            summary, sentiment, insights = analyze_transcript_gemini(prompt)
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

def analyze_transcript_openai(prompt):
    try:

        
        result = prompt_openai(prompt=prompt)

        # print(result)
        
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

def analyze_transcript_gemini(prompt):
    try:
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

def quantify_sentiment(sentiment):
    sentiment = sentiment.lower()
    if 'positive' in sentiment:
        return 1
    elif 'negative' in sentiment:
        return -1
    else:  # neutral
        return 0

def generate_sentiment_graph(summaries):
    dates = [datetime.strptime(summary['therapyDate'], '%Y-%m-%d') for summary in summaries]
    sentiments = [summary['sentiment'] for summary in summaries]

    # Define a mapping of sentiment to numeric values
    sentiment_map = {'Very Negative': -2, 'Negative': -1, 'Neutral': 0, 'Positive': 1, 'Very Positive': 2}
    
    # Convert sentiments to numeric values
    numeric_sentiments = [sentiment_map.get(s, 0) for s in sentiments]

    plt.figure(figsize=(10, 6))
    plt.plot(dates, numeric_sentiments, marker='o')
    plt.title('Sentiment Analysis Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.yticks(list(sentiment_map.values()), list(sentiment_map.keys()))
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph = base64.b64encode(buffer.getvalue()).decode()
    plt.close()  # Close the figure to free up memory

    return graph

def gpt_analysis(summaries):
    prompt = f'''

Analyze the following set of therapy summaries for Patient over the period. Please provide a comprehensive analysis addressing the following aspects:\n

Rule to follow - if you are answering in points like 1,2,3 DO NOT ADD THE NUMBER TO THE OUTPUT, ABSOLUTELY NO NUMBERS THAT INDICATE THE START OF THE POINT. BUT YOU CAN ADD HEADINGS, AND OTHER IMPORTANT THINGS TO MAKE THE DISTINCTION

Overall Progress:\n

Identify key trends in the patient's emotional state and behavior.\n
Highlight significant improvements or setbacks observed during this period.\n


Recurring Themes:\n

List and briefly explain the main issues or topics that consistently appear in the summaries.\n
Note any patterns in how these themes evolve over time.\n


Coping Strategies:\n

Identify coping mechanisms or strategies the patient has developed or improved upon.\n
Assess the effectiveness of these strategies based on the summaries.\n


Treatment Efficacy:\n

Evaluate the apparent effectiveness of the current treatment approach.\n
Suggest any potential adjustments or additional interventions that might be beneficial.\n


Risk Assessment:\n

Flag any potential risk factors or concerning behaviors mentioned in the summaries.\n
Recommend appropriate actions or precautions if necessary.\n


Interpersonal Relationships:\n

Analyze how the patient's relationships (family, friends, work) have been impacted or have evolved.\n
Identify any recurring relationship patterns or issues.\n


Goals and Aspirations:\n

Summarize the patient's stated goals or aspirations, if any.\n
Assess progress towards these goals based on the information provided.\n


Recommendations:\n

Provide 2-3 key recommendations for the next steps in the patient's treatment plan.\n
Suggest any additional assessments or interventions that might be beneficial.\n


Please synthesize this information into a coherent analysis, highlighting the most crucial insights and providing a balanced view of the patient's therapeutic journey. Your analysis should be both comprehensive and concise, suitable for review by healthcare professionals.\n\n
'''
    for summary in summaries:
        prompt += f"Date: {summary['therapyDate']}\nSummary: {summary['summary']}\nSentiment: {summary['sentiment']}\n\n"
    
   
    response = prompt_gemini(prompt)
    return response

def create_pdf_report(patient_name, start_date, end_date, summaries, sentiment_graph, gpt_analysis, png_logo_path=None):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()

        def update_or_add_style(name, parent, **kwargs):
            if name in styles:
                for k, v in kwargs.items():
                    setattr(styles[name], k, v)
            else:
                styles.add(ParagraphStyle(name=name, parent=styles[parent], **kwargs))

        update_or_add_style('Heading1', 'Heading1', fontSize=18, spaceAfter=12)
        update_or_add_style('Heading2', 'Heading2', fontSize=14, spaceAfter=6)
        update_or_add_style('Heading3', 'Heading3', fontSize=12, spaceAfter=6, textColor=colors.blue)
        update_or_add_style('BodyText', 'BodyText', fontSize=10, leading=14)
        update_or_add_style('Bold', 'BodyText', fontSize=10, leading=14, fontName='Helvetica-Bold')
        update_or_add_style('ListItem', 'BodyText', fontSize=10, leading=14, leftIndent=20, firstLineIndent=-20)

        story = []

        # Add the heading and date range
        story.append(Paragraph(f"Therapy Report for {patient_name}", styles['Heading1']))
        story.append(Paragraph(f"Date Range: {start_date} to {end_date}", styles['BodyText']))
        story.append(Spacer(1, 12))

        # Add the sentiment graph
        img = Image(io.BytesIO(base64.b64decode(sentiment_graph)))
        img.drawHeight = 4 * inch
        img.drawWidth = 6 * inch
        story.append(img)
        story.append(Spacer(1, 12))

        # Add the AI-generated analysis
        story.append(Paragraph("AI-Generated Analysis:", styles['Heading2']))
        story.append(Spacer(1, 6))

        # Process the GPT analysis text
        lines = gpt_analysis.split('\n')
        list_items = []
        in_list = False

        for line in lines:
            number_match = re.match(r'^(\d+)\.\s', line)
            if number_match:
                if not in_list:
                    if list_items:
                        story.append(ListFlowable(list_items, bulletType='1'))
                        list_items = []
                in_list = True
                number = number_match.group(1)
                item_text = line[len(number_match.group(0)):].strip()
                list_items.append(Paragraph(f"{number}. {item_text}", styles['ListItem']))
            elif in_list and line.strip():
                list_items[-1] = Paragraph(list_items[-1].text + ' ' + line.strip(), styles['ListItem'])
            else:
                if in_list:
                    story.append(ListFlowable(list_items, bulletType='1'))
                    list_items = []
                    in_list = False
                
                if line.startswith('##'):
                    story.append(Paragraph(line.strip('#'), styles['Heading3']))
                else:
                    parts = re.split(r'\*\*(.*?)\*\*', line)
                    for i, part in enumerate(parts):
                        if i % 2 == 0 and part.strip():
                            story.append(Paragraph(part.strip(), styles['BodyText']))
                        elif i % 2 != 0:
                            story.append(Paragraph(part.strip(), styles['Bold']))
                story.append(Spacer(1, 3))

        if list_items:
            story.append(ListFlowable(list_items, bulletType='1'))

        story.append(Spacer(1, 12))

        # Add session summaries
        story.append(Paragraph("Session Summaries:", styles['Heading2']))
        story.append(Spacer(1, 6))

        for summary in summaries:
            data = [
                ['Date', Paragraph(str(summary.get('therapyDate', '')), styles['BodyText'])],
                ['Summary', Paragraph(str(summary.get('summary', '')), styles['BodyText'])],
                ['Sentiment', Paragraph(str(summary.get('sentiment', '')), styles['BodyText'])]
            ]
            table = Table(data, colWidths=[1.2*inch, 5.3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (1, 0), (-1, -1), colors.beige),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(KeepTogether([table, Spacer(1, 12)]))

        def add_logo(canvas, doc):
            if png_logo_path:
                logo = ImageReader(png_logo_path)
                canvas.drawImage(logo, doc.width + doc.leftMargin - 50, doc.height + doc.topMargin - 40, width=50, height=50)

        doc.build(story, onFirstPage=add_logo, onLaterPages=add_logo)
        buffer.seek(0)
        return buffer

    except Exception as e:
        logging.error(f"Error in PDF generation: {str(e)}")
        return None

    
# Updated generate_report function
@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.form
        start_date = data['startDate']
        end_date = data['endDate']
        patient_name = data['patientName']
        summaries = [json.loads(data[f'summaryFile{i}']) for i in range(len(data) - 3)]  # -3 for startDate, endDate, and patientName

        sentiment_graph = generate_sentiment_graph(summaries)
        analysis = gpt_analysis(summaries)
        
        pdf_buffer = create_pdf_report(patient_name, start_date, end_date, summaries, sentiment_graph, analysis,png_logo_path='MMLOGO7.png')
        
        if pdf_buffer is None:
            return jsonify({"error": "Failed to generate PDF"}), 500

        return send_file(
            io.BytesIO(pdf_buffer.getvalue()),
            as_attachment=True,
            download_name=f"{patient_name}_report_{start_date}_to_{end_date}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        logging.error(f"Error in generate_report: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=8080)