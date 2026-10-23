import psycopg2 #mainly use to connect python with sql 
import re   #use to check weather the use string follows the required format



connection = psycopg2.connect(
    host="localhost",
    database="machine_entries",
    user="postgres",
    password="7028",
    port="5432"
)

cursor = connection.cursor()


# Receive new data


new_data = input("Enter new entry code: ")


# Check required format


if not re.fullmatch(r"ABC\d{5}", new_data):

    print("Invalid format")
    print("Required format: ABC12345")

else:

    
    # Check if data already exists
    

    cursor.execute(
        "SELECT * FROM entries WHERE entry_code = %s",
        (new_data,)
    )

    result = cursor.fetchone()

    if result:

        print("Data already exists")

    else:

        print("New entry")

        
        # Add new data
        

        cursor.execute(
            "INSERT INTO entries (entry_code) VALUES (%s)",
            (new_data,)
        )

        
        # Remove oldest entry
        

        cursor.execute("""
            DELETE FROM entries
            WHERE id = (
                SELECT id
                FROM entries
                ORDER BY created_at ASC, id ASC
                LIMIT 1
            )
        """)

        connection.commit()

        print("New data added successfully")
        print("Oldest data removed")


# Display current database


cursor.execute("""
    SELECT * FROM entries
    ORDER BY created_at ASC, id ASC
""")

rows = cursor.fetchall()

print("\nCurrent entries:")

for row in rows:
    print(row)


# Close connection


cursor.close()
connection.close()