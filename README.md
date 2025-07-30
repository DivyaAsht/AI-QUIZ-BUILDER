# AI-QUIZ-BUILDER

AI Quiz Generator allows users to create multiple-choice quizzes from either text input or uploaded images. It features user authentication (login/signup), tracks individual quiz history and performance, and includes an admin dashboard to view overall user statistics and quiz data. The quizzes are generated using a large language model, and quiz difficulty is predicted using a pre-trained machine learning model, designed to facilitate interactive learning and assessment by leveraging advanced AI models for content creation and robust database management for tracking user performance.

**Here's a detailed breakdown of its functionalities:**

**1. Core Quiz Generation Capabilities:**
The application offers two primary methods for generating multiple-choice quizzes (MCQs):

**Text-Based Quiz Generation:** Users can paste or type any textual content into a dedicated text area. The application then utilizes the Groq API (specifically the llama3-70b-8192 model) to generate a specified number of MCQs directly from the provided text. A key feature here is the intelligent inference of a concise topic for the quiz from the input text, replacing generic labels like "Text Input" with more descriptive terms (e.g., "Ancient Rome," "Python Programming").

**Image-Based Quiz Generation:** Alternatively, users can upload an image file (JPG, JPEG, PNG). The system employs the BLIP (Bootstrapping Language-Image Pre-training) model from the transformers library to generate a descriptive caption for the image. From this caption, a main subject is extracted (e.g., "Kite," "Dog," "Flower"). This subject is then used as a prompt for the Groq API to generate relevant MCQs.
All generated questions adhere to a strict, consistent format, ensuring clarity and ease of parsing for the quiz-playing interface. Furthermore, each generated question is analyzed, and its difficulty level is predicted using pre-trained machine learning models (difficulty_model.pkl and vectorizer.pkl), providing valuable metadata for each quiz.

**2. Interactive User Experience:**
The application prioritizes a smooth and engaging user journey:
**User Authentication:** A secure login and signup system allows users to create individual accounts and access personalized features.
**Dynamic Quiz Player:** Once a quiz is generated, users can navigate through questions one by one using "Next" and "Previous" buttons. They can select their answers via radio buttons, and their choices are saved in real-time within the session.
**Comprehensive Quiz Results:** Upon completion, users are presented with a detailed results page. This includes their overall score, a breakdown of correct and incorrect answers for each question, and a visual score analysis (pie chart) powered by plotly.express. Users also have the option to download a full Excel report (.xlsx) containing detailed quiz results and a score summary.
**Personalized Quiz History:** Each logged-in user can access their "Your Quiz History" section. This displays a table of all quizzes they have played, including the topic, score, total questions, percentage, difficulty level, and the date/time taken. Visualizations (line charts for scores over time, bar charts for performance by difficulty and by topic) offer insights into their learning progress. A "Summarized" view by topic provides average scores for unique topics, preventing redundant display of the same topic from multiple quiz attempts.

**3. Robust Admin Dashboard:**
**For administrators (specifically, the user with the username "admin"), a dedicated dashboard provides overarching insights into the application's usage and user performance:
Overall Metrics:** Displays key statistics such as the "Total Registered Users" (fetched from users.db) and "Total Quizzes Taken" across all users (from quiz_data.db).

**Aggregated Performance:** Shows the "Average Score by Difficulty" across all quizzes played by all users, presented both as a table and a bar chart.

**All Users' Scores:** A comprehensive table lists every quiz played by every user, including their username, full name (if available from the users database), topic, score, percentage, difficulty, and timestamp. This provides a complete overview of all quiz activity within the system.

**4. Technical Architecture:**
The application is built using:
**Streamlit:** For rapid development and deployment of the interactive web-based user interface.
**SQLite:** As the backend database for persistent storage. It uses two separate database files: quiz_data.db to store all quiz questions, their options, correct answers, difficulty, and user quiz results; and users.db (managed by auth_helper.py) for user authentication data.
**Groq API: **For powerful large language model capabilities, enabling dynamic quiz question generation and intelligent topic inference.
Hugging Face Transformers (BLIP): For state-of-the-art image captioning, facilitating quiz creation from visual content.
Pandas: For efficient data manipulation and integration with SQLite databases.
**Plotly Express: **For generating interactive and visually appealing data visualizations in the user history and admin dashboard.
**Joblib: **For loading pre-trained scikit-learn models used in difficulty prediction.


In essence, your AI Quiz Generator is a dynamic and intelligent platform that streamlines the process of creating and taking quizzes, while simultaneously providing valuable analytical insights for both individual users and administrators.
