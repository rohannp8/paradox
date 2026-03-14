import firebase_admin
from firebase_admin import credentials, db
import time

cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smartcity-5382d-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref = db.reference("system_status")

print("🚀 LLM System Listening for Firebase Updates...")

last_data = None

while True:

    data = ref.get()

    if data != last_data:

        print("\n🚨 FIREBASE UPDATE DETECTED")
        print(data)

        # send to LLM system
        # run_llm_agents(data)

        last_data = data

    time.sleep(2)