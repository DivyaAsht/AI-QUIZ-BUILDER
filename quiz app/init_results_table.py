import sqlite3

conn = sqlite3.connect("quiz_data.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS quiz_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    score INTEGER,
    total_questions INTEGER,
    percentage REAL,
    difficulty_level TEXT,
    topic TEXT,
    taken_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()
print("✅ Table 'quiz_results' created successfully.")
