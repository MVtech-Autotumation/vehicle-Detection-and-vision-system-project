import psycopg2
import re
import time



# Connect to PostgreSQL

connection = psycopg2.connect(
    host="localhost",
    database="plc_machine_entries",
    user="postgres",
    password="7028",
    port="5432"
) 

cursor = connection.cursor()



# Receive data

new_data = input("Enter new entry code: ").strip()



# Check required format
# 
# Example: ABCD261332a001


if not re.fullmatch(r"[A-Z]{4}\d{6}[a-z]\d{3}", new_data):
    print("Invalid format")
    print("Required format: ABCD260812a001")

else:

    
    # Check whether entry already exists
    

    cursor.execute(
        "SELECT id FROM entries WHERE entry_code = %s",
        (new_data,)
    )

    result = cursor.fetchone()
   #------------- print(result)


    
    # Existing entry
    

    if result:

        print("Data already exists")
        print("PLC Signal: STOP")
        print("Machine should NOT start.")


    
    # New entry
    

    else:

        print("New entry")

        
        # Check current number of records
        

        cursor.execute("SELECT COUNT(*) FROM entries")

        count = cursor.fetchone()[0]

        print("Current database entries:", count)


        
        # If database has reached 100,000
        # delete oldest entry
        

        if count >= 1000:

            cursor.execute("""
                DELETE FROM entries
                WHERE id = (
                    SELECT id
                    FROM entries
                    ORDER BY created_at ASC, id ASC
                    LIMIT 1
                )
            """)

            print("Oldest entry removed")


        
        # Insert new entry
        

        cursor.execute(
            "INSERT INTO entries (entry_code) VALUES (%s)",
            (new_data,)
        )

        connection.commit()

        print("New data added successfully")

        
        # PLC signal
        

        print("PLC Signal: START")
        print("Machine can start.")



# Display current number of records


cursor.execute("SELECT COUNT(*) FROM entries")

total_entries = cursor.fetchone()[0]

print("\nTotal entries in database:", total_entries)


# time code to find time


start_time = time.perf_counter()

cursor.execute(
    "SELECT id FROM entries WHERE entry_code = %s",
    (new_data,)
)

result = cursor.fetchone()

end_time = time.perf_counter()

search_time = (end_time - start_time) * 1000

print(f"Database search time: {search_time:.3f} ms")



# Close connection


cursor.close()
connection.close()