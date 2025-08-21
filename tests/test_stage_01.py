import sqlite3
from pathlib import Path
from utils.config import CONFIG


CHINOOK_TABLE_SUMMARIES_DB_PATH: Path = Path(CONFIG["CHINOOK_TABLE_SUMMARIES_DB_PATH"])


def check_summaries(db_path: str = CHINOOK_TABLE_SUMMARIES_DB_PATH):
    """Check if summaries exist in table_summaries.db and print them."""
    if not Path(db_path).exists():
        print(f"❌ DB not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Count how many rows have a non-empty summary
    cur.execute("SELECT COUNT(*) FROM table_info WHERE summary_text IS NOT NULL AND summary_text != ''")
    count = cur.fetchone()[0]

    if count == 0:
        print("⚠️ No summaries found in table_info.summary_text")
    else:
        print(f"✅ Found {count} summaries in table_info.summary_text:\n")
        cur.execute("SELECT table_name, summary_text FROM table_info WHERE summary_text IS NOT NULL AND summary_text != ''")
        rows = cur.fetchall()
        for tname, summary in rows:
            print(f"- {tname}: {summary}")

    conn.close()


if __name__ == "__main__":
    check_summaries()


# uv run tests/test_stage_01.py