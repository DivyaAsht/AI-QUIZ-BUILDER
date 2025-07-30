import streamlit as st
import base64
import time
import re
import os
import sqlite3
import joblib
from pathlib import Path
from PIL import Image
from groq import Groq
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import pandas as pd
import streamlit.components.v1 as components
import plotly.express as px
from io import BytesIO
from auth_helper import init_user_db, signup_user, authenticate_user
from datetime import datetime

# =============================================
# CONFIGURATION
# =============================================
st.set_page_config(page_title="AI Quiz Generator", page_icon="🎯", layout="wide")

# Load pre-trained models for difficulty prediction
try:
    clf = joblib.load('model/difficulty_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
except FileNotFoundError:
    st.error("Error: Model files (difficulty_model.pkl, vectorizer.pkl) not found. Please ensure they are in the 'model/' directory.")
    st.stop()

# Initialize database if it doesn't exist
if not os.path.exists("quiz_data.db"):
    try:
        df = pd.read_csv('data/training_data.csv')
        conn = sqlite3.connect('quiz_data.db')
        df.to_sql('training_questions', conn, if_exists='replace', index=False)
        conn.close()
    except FileNotFoundError:
        st.warning("Warning: 'training_data.csv' not found. Quiz difficulty prediction might be limited without training data.")
    except Exception as e:
        st.error(f"Error initializing database: {e}")

# Added: Initialize quiz_results table
def init_quiz_results_db():
    """Creates the quiz_results table if it doesn't exist, matching save_quiz_result_to_db schema."""
    conn = sqlite3.connect('quiz_data.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            score INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            percentage REAL NOT NULL,
            difficulty_level TEXT,
            topic TEXT,
            taken_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Initialize session state variables for quiz playing
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = []
if 'quiz_played' not in st.session_state:
    st.session_state.quiz_played = False
if 'show_quiz_player' not in st.session_state:
    st.session_state.show_quiz_player = False
if 'popup_state' not in st.session_state:
    st.session_state.popup_state = 'visible'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'current_quiz_topic' not in st.session_state:
    st.session_state.current_quiz_topic = "General"
if 'quiz_result_saved' not in st.session_state:
    st.session_state.quiz_result_saved = False
if 'trigger_results_display' not in st.session_state:
    st.session_state.trigger_results_display = False


# =============================================
# HELPER FUNCTIONS
# =============================================
def img_to_bytes(img_path):
    """Converts an image file to base64 bytes for embedding in HTML/CSS."""
    try:
        return base64.b64encode(Path(img_path).read_bytes()).decode()
    except FileNotFoundError:
        st.warning(f"Warning: Asset '{img_path}' not found. Welcome popup image might not display.")
        return ""

def show_popup(state):
    """Displays a welcome popup with a fade-out effect."""
    fade_class = "fade-out" if state == "fading" else ""
    st.markdown(f"""
    <style>
    header {{visibility: hidden;}}
    .st-emotion-cache-1avcm0n, footer, [data-testid="stToolbar"] {{display: none !important;}}
    html, body, .stApp {{
        margin: 0; padding: 0; height: 100%; overflow-x: hidden; background-color: #f0f2f6;
    }}
    .welcome-popup {{
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.7); display: flex; justify-content: center; align-items: center;
        z-index: 9999; animation: fadeIn 0.5s;
    }}
    .welcome-content {{
        background: white; padding: 2rem; border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center;
        max-width: 400px; animation: scaleUp 0.4s ease-out forwards;
    }}
    .welcome-logo {{ max-width: 100%; height: auto; margin-bottom: 1.5rem; }}
    .welcome-text {{ font-size: 2.5rem; font-weight: 700; color: #2c3e50; margin-top: 1rem; }}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @keyframes scaleUp {{ from {{ transform: scale(0.9); }} to {{ transform: scale(1); }} }}
    @keyframes fadeOut {{ from {{ opacity: 1; }} to {{ opacity: 0; }} }}
    .fade-out {{ animation: fadeOut 0.5s forwards; }}
    </style>
    <div class="welcome-popup {fade_class}">
        <div class="welcome-content">
            <img class="welcome-logo" src="data:image/png;base64,{img_to_bytes('assets/splash.gif')}">
            <div class="welcome-text">Welcome</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# AI/ML FUNCTIONS
# =============================================
@st.cache_resource
def load_blip_model():
    """Loads the BLIP image captioning model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def caption_image_with_blip(image):
    """Generates a caption for an image using the BLIP model."""
    image = image.resize((512, 512))
    processor, model = load_blip_model()
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def extract_main_subject(caption):
    """Extracts the main subject from an image caption."""
    keywords = ["dog", "cat", "tiger", "lion", "car", "plane", "tree", "flower", "apple", "banana", "person"]
    for word in keywords:
        if word in caption.lower():
            return word
    match = re.search(r"\b[a-zA-Z]{3,}\b", caption)
    return match.group(0) if match else "this image"

def generate_quiz_from_text(text, num_questions=5):
    """Generates multiple-choice questions from text using Groq API."""
    client = Groq(api_key=st.secrets["groq_api_key"])
    prompt = f"""Generate exactly {num_questions} multiple-choice questions based on the provided text.
Each question MUST follow this precise format, with no additional text, introductions, or conclusions:

Question 1: [Question text]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Correct answer: [Letter]

Question 2: [Question text]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Correct answer: [Letter]

... and so on for all {num_questions} questions.
Ensure there is exactly one correct answer per question, indicated by the letter (A, B, C, or D).

Text to generate questions from: {text}"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful quiz generator. STRICTLY adhere to the requested output format without any deviations, extra text, or preamble. Ensure each part (Question, A, B, C, D, Correct answer) is present and correctly formatted."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Added: New function to infer topic from text
def infer_topic_from_text(text):
    """Infers a concise topic from a given text using Groq API."""
    client = Groq(api_key=st.secrets["groq_api_key"])
    prompt = f"""Given the following text, identify and return a single, concise topic name (1-3 words).
    Do NOT include any other text, explanations, or punctuation other than the topic itself.
    Examples:
    Text: "The history of ancient Rome, including its emperors and architecture."
    Topic: Ancient Rome
    Text: "Basic concepts of Python programming, loops, and data structures."
    Topic: Python Programming
    Text: "The life cycle of a butterfly and its metamorphosis."
    Topic: Butterfly Life Cycle
    Text: "Different types of renewable energy sources like solar and wind power."
    Topic: Renewable Energy

    Text: {text}
    Topic:"""
    response = client.chat.completions.create(
        model="llama3-8b-8192", # Using a smaller model for faster topic inference
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts concise topic names from text. Respond ONLY with the topic name."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=20 # Ensure concise response
    )
    return response.choices[0].message.content.strip()


def parse_quiz_text(quiz_text):
    """
    Parses the raw quiz text into a structured list of dictionaries.
    Improved regex to robustly extract question, options, and correct answer.
    """
    parsed = []
    # Split the entire text into potential question blocks
    # This regex looks for "Question X:" or "Q X:" or "Q S:" followed by anything, up to the next "Question" or end of string.
    # It also handles the "Here are X multiplechoice questions..." preamble.
    # The (?:...) makes the groups non-capturing.
    question_blocks = re.split(r'(?:Question\s*\d*[:\.]?|Q\s*\d*[:\.]?|Q\s*S\s*:\s*)', quiz_text, flags=re.IGNORECASE)
    
    # Filter out empty strings and the initial preamble block if it exists
    # Also filter out any block that is clearly just an introduction like "Here are the X questions..."
    question_blocks = [
        block.strip() for block in question_blocks 
        if block.strip() and not (
            block.strip().lower().startswith("here are") or 
            block.strip().lower().startswith("the following") or
            block.strip().lower().startswith("based on the text")
        )
    ]

    for block in question_blocks:
        # Regex to extract components from a single question block
        # It assumes the block starts directly with the question text after any prefix handled by split.
        # It then looks for A), B), C), D) and the Correct answer line.
        # Added (?:Question\s*\d*\s*)? to optionally remove "Question X" if it appears *within* the question text after split.
        match = re.search(
            r"^(?:Question\s*\d*\s*)?(.*?)\n\s*A[\)\.]\s*(.*?)\s*B[\)\.]\s*(.*?)\s*C[\)\.]\s*(.*?)\s*D[\)\.]\s*(.*?)\s*Correct answer:\s*([A-D])",
            block,
            re.DOTALL | re.IGNORECASE
        )
        if match:
            q_text, a, b, c, d, correct = match.groups()
            
            if not q_text.strip() or not a.strip() or not b.strip() or not c.strip() or not d.strip() or not correct.strip():
                st.warning(f"Skipping malformed question due to missing parts: '{q_text}'")
                continue

            parsed.append({
                'question': q_text.strip(),
                'options': {
                    'A': a.strip(),
                    'B': b.strip(),
                    'C': c.strip(),
                    'D': d.strip()
                },
                'correctAnswer': correct.strip().upper(),
            })
        else:
            st.warning(f"Could not parse question block: \n{block[:200]}...") # Show first 200 chars for debugging
    return parsed


def clean_question(text):
    """Cleans question text for consistent processing."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?\!]', '', text)
    return text

def extract_clean_questions(raw_text):
    """Extracts and cleans individual question lines from raw quiz text."""
    lines = raw_text.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("here are"):
            continue
        if "multiple-choice" in line.lower():
            continue
        if len(line) < 10:
            continue
        if re.match(r"Question\s*\d*[:\.]?|([A-D][)\.])", line, re.IGNORECASE):
             questions.append(line)
    return "\n".join(questions)

def predict_difficulty(question_text):
    """Predicts the difficulty of a given question text."""
    if 'vectorizer' in globals() and 'clf' in globals():
        vec = vectorizer.transform([question_text])
        return clf.predict(vec)[0]
    return "Unknown"

def save_to_db(parsed_questions):
    """Saves new quiz questions to CSV and SQLite database with predicted difficulty."""
    cleaned_data = []
    for q in parsed_questions:
        q_text = clean_question(q['question'])
        difficulty = predict_difficulty(q_text)
        cleaned_data.append({
            'question': q_text,
            'option_1': q['options']['A'],
            'option_2': q['options']['B'],
            'option_3': q['options']['C'],
            'option_4': q['options']['D'],
            'correct_option': q['correctAnswer'],
            'difficulty': difficulty
        })

    df_new = pd.DataFrame(cleaned_data)

    csv_path = "data/training_data.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df_existing_csv = pd.DataFrame(columns=df_new.columns)
    if os.path.exists(csv_path):
        try:
            df_existing_csv = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            st.warning("training_data.csv is empty. Starting with new data.")
        except Exception as e:
            st.error(f"Error reading existing CSV: {e}")

    df_combined = pd.concat([df_existing_csv, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset='question', inplace=True)
    df_combined.to_csv(csv_path, index=False)

    conn = sqlite3.connect('quiz_data.db')
    df_existing_db = pd.read_sql_query("SELECT question FROM training_questions", conn)
    existing_questions_set = set(df_existing_db['question'].tolist())
    unique_rows = df_new[~df_new['question'].isin(existing_questions_set)]

    if not unique_rows.empty:
        unique_rows.to_sql('training_questions', conn, if_exists='append', index=False)
    conn.close()

def load_quiz_from_db(num_questions):
    """
    Loads a specified number of quiz questions from the SQLite database,
    including their predicted difficulty.
    """
    conn = sqlite3.connect('quiz_data.db')
    try:
        query = f"SELECT question, option_1, option_2, option_3, option_4, correct_option, difficulty FROM training_questions ORDER BY ROWID DESC LIMIT {num_questions}"
        df_quiz = pd.read_sql_query(query, conn)
        
        quiz_data = []
        for index, row in df_quiz.iterrows():
            quiz_data.append({
                'question': row['question'],
                'options': {
                    'A': row['option_1'],
                    'B': row['option_2'],
                    'C': row['option_3'],
                    'D': row['option_4']
                },
                'correctAnswer': row['correct_option'],
                'difficulty': row['difficulty']
            })
        return quiz_data
    except Exception as e:
        st.error(f"Error loading quiz from database: {e}")
        return []
    finally:
        conn.close()


# =============================================
# QUIZ PLAYING UI FUNCTIONS
# =============================================

def _next_question():
    """Callback for 'Next' button."""
    if st.session_state.current_question_index < len(st.session_state.quiz_questions) - 1:
        st.session_state.current_question_index += 1
    else:
        # Quiz finished logic
        # Calculate score and save result here, only once
        score = 0
        total = len(st.session_state.quiz_questions)
        for i, q in enumerate(st.session_state.quiz_questions):
            user_ans = st.session_state.user_answers[i]
            correct_ans = q['correctAnswer']
            if (user_ans == correct_ans) and (user_ans is not None):
                score += 1

        if st.session_state.user_name and not st.session_state.quiz_result_saved:
            difficulty = st.session_state.quiz_questions[0].get("difficulty", "Mixed")
            save_quiz_result_to_db(st.session_state.user_name, score, total, difficulty, st.session_state.current_quiz_topic)
            st.session_state.quiz_result_saved = True # Set flag to prevent duplicate saves
        
        st.session_state.quiz_played = True
        st.session_state.show_quiz_player = False
        st.session_state.trigger_results_display = True # Set flag to trigger display in main script


def _prev_question():
    """Callback for 'Previous' button."""
    if st.session_state.current_question_index > 0:
        st.session_state.current_question_index -= 1

def update_user_answer(question_index, radio_key):
    """Callback for st.radio to update the user's answer."""
    selected_value_full_string = st.session_state[radio_key]
    if selected_value_full_string:
        st.session_state.user_answers[question_index] = selected_value_full_string[0]


def display_quiz_question():
    if not st.session_state.quiz_questions:
        st.warning("No quiz questions available. Please generate a quiz first.")
        return

    q_index = st.session_state.current_question_index
    question = st.session_state.quiz_questions[q_index]
    
    if len(st.session_state.user_answers) != len(st.session_state.quiz_questions):
        st.session_state.user_answers = [None] * len(st.session_state.quiz_questions)

    current_user_selection_letter = st.session_state.user_answers[q_index]

    st.markdown(f"""
    <style>
        .quiz-card {{
            background: #fff;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 800px;
            margin: auto;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 2rem;
        }}
        .question-text {{
            font-size: 1.25rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1.5rem;
        }}
        div[data-testid="stRadio"] > label {{
            margin: 10px 0;
            padding: 12px 15px;
            background: #f1f5f9;
            border-radius: 8px;
            transition: background 0.2s, border-left 0.2s;
            cursor: pointer;
            border-left: 5px solid transparent;
        }}
        div[data-testid="stRadio"] > label:hover {{
            background: #e2e8f0;
        }}
        div[data-testid="stRadio"] label.st-dg {{
            background-color: #d4edda !important;
            border-left: 5px solid #38a169;
        }}
        .navigation-buttons {{
            display: flex;
            justify-content: space-between;
            margin-top: 2rem;
        }}
        .stButton button {{
            background-color: #4c51bf;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            font-size: 1rem;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .stButton button:hover {{
            background-color: #5a67d8;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
    </style>
    <div class='quiz-card'>
        <div class='question-text'>
            Q{q_index + 1}: {question['question']}
        </div>
    """, unsafe_allow_html=True)

    options_dict = question['options']
    options_list = [f"{key}) {value}" for key, value in options_dict.items()]

    initial_index = None
    if current_user_selection_letter and current_user_selection_letter in options_dict:
        selected_full_option = f"{current_user_selection_letter}) {options_dict[current_user_selection_letter]}"
        if selected_full_option in options_list:
            initial_index = options_list.index(selected_full_option)

    radio_widget_key = f"question_{q_index}"

    st.radio(
        "Choose an option:",
        options_list,
        index=initial_index,
        key=radio_widget_key,
        label_visibility="collapsed",
        on_change=update_user_answer,
        args=(q_index, radio_widget_key)
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if q_index > 0:
            st.button("Previous", on_click=_prev_question)
    with col2:
        if q_index < len(st.session_state.quiz_questions) - 1:
            st.button("Next", on_click=_next_question)
        else:
            st.button("Submit Quiz", on_click=_next_question)

    st.markdown("</div>", unsafe_allow_html=True)


def display_quiz_results():
    score = 0
    total = len(st.session_state.quiz_questions)
    detailed_results = []

    # Recalculate score for display purposes only, as saving is done in _next_question
    for i, q in enumerate(st.session_state.quiz_questions):
        user_ans = st.session_state.user_answers[i]
        correct_ans = q['correctAnswer']
        if (user_ans == correct_ans) and (user_ans is not None):
            score += 1

    st.markdown(f"""
    <style>
        .result-box {{
            background: #edf2f7;
            padding: 2rem;
            border-radius: 15px;
            max-width: 700px;
            margin: auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            font-family: 'Segoe UI', sans-serif;
            margin-top: 2rem;
        }}
        .result-box h2 {{
            color: #2d3748;
            text-align: center;
            margin-bottom: 1.5rem;
        }}
        .answer-block {{
            margin-top: 1rem;
            padding: 1rem;
            background: #fff;
            border-left: 5px solid;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        .answer-line {{
            display: flex;
            align-items: center;
        }}
        .answer-label {{
            font-weight: bold;
            margin-right: 0.5rem;
        }}
        .correct {{
            border-color: #38a169;
            background-color: #f0fff4;
        }}
        .incorrect {{
            border-color: #e53e3e;
            background-color: #fff5f5;
        }}
        .final-score {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-top: 2rem;
        }}
    </style>
    <div class='result-box'>
        <h2>Quiz Results</h2>
    """, unsafe_allow_html=True)

    for i, q in enumerate(st.session_state.quiz_questions):
        user_ans = st.session_state.user_answers[i]
        correct_ans = q['correctAnswer']
        
        correct = (user_ans == correct_ans) and (user_ans is not None)
        
        status_class = "correct" if correct else "incorrect"
        result_icon = "✅ Correct" if correct else "❌ Incorrect"

        user_answer_text = f"{user_ans}) {q['options'].get(user_ans, 'Not answered')}" if user_ans else "Not answered"
        correct_answer_text = f"{correct_ans}) {q['options'].get(correct_ans, 'Error: No correct option found')}"

        st.markdown(f"""
        <div class='answer-block {status_class}'>
            <div class='answer-line'><span class='answer-label'>Q{i+1}:</span> {q['question']}</div>
            <div class='answer-line'><span class='answer-label'>Your answer:</span> {user_answer_text}</div>
            <div class='answer-line'><span class='answer-label'>Correct answer:</span> {correct_answer_text}</div>
            <div class='answer-line'><span class='answer-label'>Result:</span> {result_icon}</div>
        </div>
        """, unsafe_allow_html=True)

        detailed_results.append({
            "Q#": i + 1,
            "Question": q['question'],
            "Your Answer": user_answer_text,
            "Correct Answer": correct_answer_text,
            "Result": "Correct" if correct else "Incorrect",
            "Difficulty": q.get("difficulty", "N/A")
        })

    st.markdown(f"""
        <div class='final-score'>Final Score: {score} / {total}</div>
    </div>
    """, unsafe_allow_html=True)

    incorrect = total - score
    st.subheader("📈 Score Analysis")
    fig = px.pie(
        names=["Correct", "Incorrect"],
        values=[score, incorrect],
        title="Your Score Distribution",
        color_discrete_sequence=["#38a169", "#e53e3e"]
    )
    st.plotly_chart(fig, use_container_width=True)

    df_results = pd.DataFrame(detailed_results)

    score_summary = pd.DataFrame({
        "Total Questions": [total],
        "Correct": [score],
        "Incorrect": [incorrect],
        "Score (%)": [f"{(score/total)*100:.2f}%"]
    })

    st.subheader("⬇️ Export to Excel")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, index=False, sheet_name="Quiz Results")
        score_summary.to_excel(writer, index=False, sheet_name="Score Summary")
        writer.close()

    st.download_button(
        label="📥 Download Excel Report",
        data=output.getvalue(),
        file_name="quiz_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if st.button("🔁 Restart Quiz"):
        st.session_state.quiz_questions = []
        st.session_state.user_answers = []
        st.session_state.quiz_played = False
        st.session_state.show_quiz_player = False
        st.session_state.current_question_index = 0
        st.session_state.quiz_result_saved = False # Reset flag for new quiz
        st.session_state.trigger_results_display = False # Reset trigger
        st.rerun()

# -------------------------------
# Helper function (already exists in your provided code)
# -------------------------------
def save_quiz_result_to_db(username, score, total, difficulty, topic):
    conn = sqlite3.connect("quiz_data.db")
    cursor = conn.cursor()
    percentage = (score / total) * 100
    cursor.execute('''
        INSERT INTO quiz_results (username, score, total_questions, percentage, difficulty_level, topic, taken_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    ''', (username, score, total, percentage, difficulty, topic))
    conn.commit()
    conn.close()

def show_user_stats(username):
    st.subheader("📊 Your Quiz Stats")
    conn = sqlite3.connect("quiz_data.db")
    df = pd.read_sql_query("SELECT topic, score, total_questions, percentage, difficulty_level, taken_at FROM quiz_results WHERE username = ? ORDER BY taken_at DESC", conn, params=(username,))
    
    if not df.empty:
        st.markdown("#### Detailed Quiz History")
        st.dataframe(df, use_container_width=True)

        st.subheader("Your Scores Over Time")
        fig_user_time = px.line(
            df,
            x='taken_at',
            y='percentage',
            title='Your Quiz Scores Over Time',
            markers=True,
            labels={'taken_at': 'Date', 'percentage': 'Score (%)'}
        )
        st.plotly_chart(fig_user_time, use_container_width=True)

        st.subheader("Your Performance by Difficulty")
        avg_score_user_difficulty = df.groupby('difficulty_level')['percentage'].mean().reset_index()
        avg_score_user_difficulty.columns = ['Difficulty', 'Average Score (%)']
        st.dataframe(avg_score_user_difficulty)

        fig_user_difficulty = px.bar(
            avg_score_user_difficulty,
            x='Difficulty',
            y='Average Score (%)',
            title='Your Average Score by Difficulty',
            color='Average Score (%)',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_user_difficulty, use_container_width=True)

        # Added: Summarized view by topic
        st.subheader("Your Performance by Topic (Summarized)")
        avg_score_user_topic = df.groupby('topic')['percentage'].mean().reset_index()
        avg_score_user_topic.columns = ['Topic', 'Average Score (%)']
        st.dataframe(avg_score_user_topic)

        fig_user_topic = px.bar(
            avg_score_user_topic,
            x='Topic',
            y='Average Score (%)',
            title='Your Average Score by Topic',
            color='Average Score (%)',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_user_topic, use_container_width=True)

    else:
        st.info("You haven't played any quizzes yet. Generate and play a quiz to see your history!")
    conn.close()


def show_admin_dashboard():
    st.title("📈 Admin Dashboard")
    st.markdown("---")
    
    # Establish a separate connection for users.db
    users_conn = None
    try:
        users_conn = sqlite3.connect('users.db') # Connect to users.db
        # Total users
        total_users = users_conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        st.metric("👥 Total Registered Users", total_users)
    except sqlite3.OperationalError as e:
        st.error(f"Error loading user count from users.db: {e}. Please ensure 'users.db' exists and 'auth_helper.py' correctly initializes a table named 'users' within it.")
    finally:
        if users_conn:
            users_conn.close()

    conn = sqlite3.connect("quiz_data.db")

    # Total quizzes taken
    try:
        total_quizzes = conn.execute("SELECT COUNT(*) FROM quiz_results").fetchone()[0]
        st.metric("📝 Total Quizzes Taken", total_quizzes)
    except sqlite3.OperationalError as e:
        st.error(f"Error loading quiz count from quiz_data.db: {e}. Please ensure the 'quiz_results' table exists and is accessible.")


    # Average by difficulty
    st.subheader("📊 Average Score by Difficulty")
    try:
        avg_scores = pd.read_sql_query('''
            SELECT difficulty_level, ROUND(AVG(percentage), 2) as avg_score
            FROM quiz_results
            GROUP BY difficulty_level
        ''', conn)
        if not avg_scores.empty:
            st.dataframe(avg_scores)
            fig_difficulty = px.bar(
                avg_scores,
                x='difficulty_level',
                y='avg_score',
                title='Average Score by Quiz Difficulty',
                color='avg_score',
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={'difficulty_level': 'Difficulty', 'avg_score': 'Average Score (%)'}
            )
            st.plotly_chart(fig_difficulty, use_container_width=True)
        else:
            st.info("No quiz results yet to show difficulty breakdown.")
    except Exception as e:
        st.error(f"Error loading average scores by difficulty: {e}")

    # All users' scores
    st.subheader("🧑‍🎓 All Users' Scores")
    try:
        # This query correctly selects all data from quiz_results
        all_scores_df = pd.read_sql_query("SELECT username, topic, score, percentage, difficulty_level, taken_at FROM quiz_results ORDER BY taken_at DESC", conn)
        
        if not all_scores_df.empty:
            # Attempt to join with users.db to get full names if possible
            users_conn_for_join = None
            try:
                users_conn_for_join = sqlite3.connect('users.db')
                users_df = pd.read_sql_query("SELECT username, name FROM users", users_conn_for_join)
                # Merge quiz results with user names
                all_scores_df = pd.merge(all_scores_df, users_df, on='username', how='left')
                # Reorder columns for better display: username, name, then other details
                all_scores_df = all_scores_df[['username', 'name', 'topic', 'score', 'percentage', 'difficulty_level', 'taken_at']]
            except Exception as e:
                st.warning(f"Could not load full names for all users: {e}. Displaying only usernames.")
                # If join fails, ensure 'name' column exists, but with NaNs
                if 'name' not in all_scores_df.columns:
                    all_scores_df['name'] = None
            finally:
                if users_conn_for_join:
                    users_conn_for_join.close()

            st.dataframe(all_scores_df, use_container_width=True)
            # Diagnostic print: Print the DataFrame to console for debugging
            print("Admin Dashboard - All Users' Scores DataFrame:")
            print(all_scores_df)
        else:
            st.info("No quiz results available from any user yet.")
    except Exception as e:
        st.error(f"Error loading all users' scores: {e}")

    conn.close()


def login_signup_page():
    st.markdown("### 🔐 Login or Signup to Continue")

    action = st.radio("Choose:", ["Login", "Sign Up"], horizontal=True)

    username = st.text_input("Username")
    name = st.text_input("Full Name (only for signup)") if action == "Sign Up" else ""
    password = st.text_input("Password", type="password")

    if st.button("Continue"):
        if action == "Login":
            valid, user_full_name = authenticate_user(username, password) # Renamed user_name to user_full_name for clarity
            if valid:
                st.success(f"Welcome back, {user_full_name}!")
                st.session_state.logged_in = True
                st.session_state.user_name = username # Changed: Set user_name to the actual username input
                st.rerun()
            else:
                st.error("Invalid username or password.")
        else:
            if not (username and name and password):
                st.warning("Please fill all fields.")
            else:
                success, msg = signup_user(username, name, password)
                if success:
                    st.success(msg + " Please log in now.")
                else:
                    st.error(msg)


# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main function to run the Streamlit application."""
    if st.session_state.popup_state == 'visible':
        show_popup('visible')
        time.sleep(2)
        st.session_state.popup_state = 'fading'
        st.rerun()
    elif st.session_state.popup_state == 'fading':
        show_popup('fading')
        time.sleep(0.5)
        st.session_state.popup_state = 'hidden'
        st.rerun()

    if st.session_state.popup_state != 'hidden':
        return

    # ✅ Login first
    if not st.session_state.logged_in:
        login_signup_page()
        return

    # Sidebar for navigation and logout
    st.sidebar.title(f"Welcome, {st.session_state.user_name}!")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_name = None
        # Reset quiz state upon logout
        st.session_state.quiz_questions = []
        st.session_state.user_answers = []
        st.session_state.quiz_played = False
        st.session_state.show_quiz_player = False
        st.session_state.current_question_index = 0
        st.session_state.current_quiz_topic = "General"
        st.session_state.quiz_result_saved = False # Reset flag for new quiz
        st.session_state.trigger_results_display = False # Reset trigger
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    
    # Conditional navigation for admin
    if st.session_state.user_name == "admin":
        page_selection = st.sidebar.radio("Go to:", ["Generate Quiz", "Your Quiz History", "Admin Dashboard"])
    else:
        page_selection = st.sidebar.radio("Go to:", ["Generate Quiz", "Your Quiz History"])


    st.markdown("""<style>
    .main-container {
        background-color: white; padding: 2rem; border-radius: 15px;
        max-width: 800px; margin: auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    </style>""", unsafe_allow_html=True)

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Display content based on page selection
    if page_selection == "Generate Quiz":
        st.title("AI Quiz Generator 🎯")
        # Check if results display is triggered
        if st.session_state.trigger_results_display:
            display_quiz_results()
            st.session_state.trigger_results_display = False # Reset trigger after display
        elif st.session_state.show_quiz_player:
            display_quiz_question()
        else:
            input_type = st.radio("Choose your input type:", ["Text", "Image"], horizontal=True)
            num_questions = st.slider("How many MCQs do you want?", min_value=1, max_value=50, value=5)

            user_input = None
            if input_type == "Text":
                user_input = st.text_area("Enter your content here:", height=200)
            else:
                user_input = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

            if st.button("Generate Quiz"):
                if user_input:
                    st.session_state.quiz_questions = []
                    st.session_state.current_question_index = 0
                    st.session_state.user_answers = [None] * num_questions
                    st.session_state.quiz_played = False
                    st.session_state.quiz_result_saved = False # Reset flag for new quiz
                    st.session_state.trigger_results_display = False # Ensure this is false for new quiz generation

                    with st.spinner("Generating quiz..."):
                        parsed_quiz_data = []
                        quiz_topic = "General" # Default topic
                        if input_type == "Text":
                            # Infer topic from text
                            try:
                                inferred_topic = infer_topic_from_text(user_input)
                                quiz_topic = inferred_topic
                                st.info(f"Inferred Topic: {quiz_topic}")
                            except Exception as e:
                                st.warning(f"Could not infer topic from text: {e}. Using 'Text Input'.")
                                quiz_topic = "Text Input"
                            
                            quiz_raw_text = generate_quiz_from_text(user_input, num_questions)
                            parsed_quiz_data = parse_quiz_text(quiz_raw_text)
                        else:
                            try:
                                image = Image.open(user_input)
                                caption = caption_image_with_blip(image)
                                subject = extract_main_subject(caption)
                                quiz_prompt = f"Make {num_questions} MCQ questions about {subject} for a school quiz."
                                quiz_raw_text = generate_quiz_from_text(quiz_prompt, num_questions)
                                parsed_quiz_data = parse_quiz_text(quiz_raw_text)
                                quiz_topic = subject.capitalize() 
                                st.success(f"Recognized Subject: {subject.capitalize()}.")
                            except Exception as e:
                                st.error(f"Error processing image or generating quiz: {e}")
                                parsed_quiz_data = []

                        if parsed_quiz_data:
                            # Clean the topic string before saving to session state
                            # Ensure it's not empty after cleaning
                            cleaned_topic = quiz_topic.strip().replace(" ", "_").replace("-", "_").lower()
                            if not cleaned_topic:
                                cleaned_topic = "general_quiz" # Fallback if topic extraction fails
                            st.session_state.current_quiz_topic = cleaned_topic

                            save_to_db(parsed_quiz_data) # This saves questions to training_questions table
                            st.info("✅ Quiz generated and saved to database with predicted difficulty.")

                            st.session_state.quiz_questions = load_quiz_from_db(num_questions)
                            if st.session_state.quiz_questions:
                                st.session_state.user_answers = [None] * len(st.session_state.quiz_questions)
                                st.success("Quiz loaded from database, ready to play!")
                                st.session_state.show_quiz_player = True
                                st.rerun()
                            else:
                                st.warning("❗ Could not load quiz from database. Please try generating again.")
                        else:
                            st.warning("❗ Could not generate or parse quiz. Please try again.")
                else:
                    st.warning("⚠️ Please provide some input first.")
    
    elif page_selection == "Your Quiz History":
        show_user_stats(st.session_state.user_name)
    
    elif page_selection == "Admin Dashboard":
        show_admin_dashboard()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    init_user_db()  # Creates users.db if not present
    init_quiz_results_db() # Added: Creates quiz_results table if not present
    main()
