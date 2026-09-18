from pathlib import Path
import sqlite3


DATABASE_NAME = str(Path(__file__).resolve().parents[1] / "stroke_history.db")


# ==========================
# CREATE DATABASE
# ==========================

def create_database():

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            glucose REAL,
            bmi REAL,
            result TEXT
        )
        """
    )

    conn.commit()
    conn.close()



# ==========================
# SAVE PREDICTION
# ==========================

def save_prediction(
    age,
    glucose,
    bmi,
    result
):

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (age, glucose, bmi, result)
        VALUES (?, ?, ?, ?)
        """,
        (
            age,
            glucose,
            bmi,
            result
        )
    )

    conn.commit()
    conn.close()



# ==========================
# GET HISTORY
# ==========================

def get_history():

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT age, glucose, bmi, result
        FROM predictions
        """
    )

    data = cursor.fetchall()

    conn.close()

    return data