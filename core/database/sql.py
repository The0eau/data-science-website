import sqlite3
import os

"""
DOCS & COMPARISON:
-------------------------------------------------------------------------------
2. SQLite (The "Featherweight")
- How it works: A self-contained, serverless database engine in a single file (.db).
- Architecture: The engine is integrated directly into the application.
- Best for: Mobile apps (iOS/Android), desktop software, IoT, or prototyping.
- Pros: Zero configuration, ultra-fast reads, highly portable.

3. PostgreSQL (The "Heavyweight")
- How it works: A robust Client-Server database management system (RDBMS).
- Architecture: Runs as a separate service on a server. Multiple users connect via network.
- Best for: High-traffic websites (Instagram-scale), complex enterprise apps, GIS data.
- Pros: Handles massive data, advanced security, strict data integrity.
-------------------------------------------------------------------------------
"""

def run_database_demo():
    db_name = 'demo_database.db'
    
    # 1. CONNECTION (SQLite)
    # Note: For PostgreSQL, you would use 'psycopg2' and provide host/user/pass
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    print("--- SQL MASTER CLASS: DEMO START ---")

    # 2. DDL (Data Definition Language) - Creating the structure
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS developers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            specialty TEXT,
            experience_years INTEGER
        )
    ''')
    print("[DDL] Table 'developers' created/verified.")

    # 3. DML (Data Manipulation Language) - Inserting data
    devs = [
        ('Alice', 'PostgreSQL Expert', 8),
        ('Bob', 'SQLite Specialist', 3),
        ('Charlie', 'Fullstack Dev', 5)
    ]
    cursor.executemany("INSERT INTO developers (name, specialty, experience_years) VALUES (?, ?, ?)", devs)
    conn.commit()
    print(f"[DML] {len(devs)} developers inserted.")

    # 4. DQL (Data Query Language) - Selecting and Filtering
    print("\n--- QUERY RESULTS (Experience > 4 years) ---")
    cursor.execute("SELECT name, specialty FROM developers WHERE experience_years > 4 ORDER BY experience_years DESC")
    
    for row in cursor.fetchall():
        print(f"-> DEV: {row[0]} | Skills: {row[1]}")

    # 5. AGGREGATE FUNCTIONS - Calculations
    cursor.execute("SELECT AVG(experience_years) FROM developers")
    avg_exp = cursor.fetchone()[0]
    print(f"\n[STATS] Average experience: {avg_exp:.1f} years")

    # 6. UPDATE & DELETE - Maintenance
    cursor.execute("UPDATE developers SET experience_years = 4 WHERE name = 'Bob'")
    cursor.execute("DELETE FROM developers WHERE name = 'Charlie'")
    conn.commit()
    print("[DML] Bob updated and Charlie deleted.")

    # Close connection
    conn.close()
    print("\n--- DEMO FINISHED: Connection closed ---")

if __name__ == "__main__":
    run_database_demo()