import psycopg2

try:
    connection = psycopg2.connect(
        host="localhost",
        database="machine_entries",
        user="postgres",
        password="7028",
        port="5432"
    )

    print("Connected to PostgreSQL successfully!")
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM entries") #use to interact with sql (cursor.execute)
    rows = cursor.fetchall() #
    for row in rows:
        print(row)

    connection.close()

except Exception as e:
    print("Connection failed:", e)
    




    