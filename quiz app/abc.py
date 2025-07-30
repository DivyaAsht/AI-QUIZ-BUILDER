'''import sqlite3
import pandas as pd

try:
    # Connect to the DB
    conn = sqlite3.connect("quiz_data.db")
    print("✅ Connected to quiz_data.db")

    # Check table names
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("📦 Available tables:")
    print(tables)

    # Read the training_questions table
    df = pd.read_sql_query("SELECT difficulty, question FROM training_questions", conn)

    # Show last few rows
    print("📝 Last 10 entries in 'training_questions':")
    print(df.tail(10))

except Exception as e:
    print("❌ Error:", e)

finally:
    conn.close()'''
import pandas as pd

try:
    df = pd.read_csv("data/training_data.csv")

    if len(df) >= 5:
        df = df[:-5]  # Drop last 5
        df.to_csv("data/training_data.csv", index=False)
        print("✅ Last 5 rows deleted from training_data.csv.")
    else:
        print("⚠️ Not enough rows to delete 5.")

except Exception as e:
    print("❌ CSV Error:", e)
    