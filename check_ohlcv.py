import sqlite3

db_path = r"D:RAG_FamilyPSamertrady2db.sqlite3"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# استعلام للحصول على كل أسماء الجداول
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tabels are:")
for table in tables:
    print(table[0])

cursor.close()
conn.close()
