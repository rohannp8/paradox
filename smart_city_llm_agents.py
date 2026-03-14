import os
import time
from crewai import Agent, Task, Crew, LLM
import firebase_admin
from firebase_admin import credentials, db

# -----------------------------------
# FIREBASE INITIALIZATION
# -----------------------------------

cred = credentials.Certificate("firebase_key.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(
        cred,
        {
            "databaseURL": "https://smartcity-5382d-default-rtdb.asia-southeast1.firebasedatabase.app/"
        },
    )

ref = db.reference("system_status")

# -----------------------------------
# GEMINI API
# -----------------------------------

os.environ["GOOGLE_API_KEY"] = "AIzaSyBE6oHznc3eWVDMkyhQ5wfYAvkRxhWGtJo"

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0
)

# -----------------------------------
# AGENT 1 : INCIDENT ANALYSIS
# -----------------------------------

incident_agent = Agent(
    role="Incident Analysis Agent",
    goal="Analyze surveillance events from CCTV cameras and determine incident type and priority.",
    backstory="""
You are an expert surveillance analyst working in a Smart City Command Center.

Possible incident types:
- hit_and_run
- women_distress
- crowd_panic
- physical_assault

Priority rules:
hit_and_run → CRITICAL
women_distress → CRITICAL
physical_assault → HIGH
crowd_panic → MEDIUM
""",
    llm=llm,
    verbose=True
)

# -----------------------------------
# AGENT 2 : EMERGENCY RESPONSE
# -----------------------------------

response_agent = Agent(
    role="Emergency Response Coordinator",
    goal="Determine which emergency services should respond.",
    backstory="""
You coordinate emergency responses in a Smart City.

Rules:

hit_and_run
- Ambulance
- Hospital
- Police

women_distress
- Women Helpline 1091
- Police Patrol

crowd_panic
- Crowd Control Authority
- Police

physical_assault
- Police Station
- Police Patrol
""",
    llm=llm,
    verbose=True
)

# -----------------------------------
# AGENT 3 : POLICE DASHBOARD
# -----------------------------------

dashboard_agent = Agent(
    role="Police Dashboard Alert Generator",
    goal="Generate structured emergency alerts for the police command center dashboard.",
    backstory="""
You generate structured alerts displayed in a police command center dashboard.
Police officers use your alerts to quickly understand incidents and take action.
""",
    llm=llm,
    verbose=True
)

# -----------------------------------
# AGENT 4 : AI EXPLANATION
# -----------------------------------

explanation_agent = Agent(
    role="AI Decision Explanation Agent",
    goal="Explain why the emergency response actions were triggered.",
    backstory="""
You explain the reasoning behind emergency alerts so that police officers
can understand why the system triggered the response.
""",
    llm=llm,
    verbose=True
)

print("🚀 Smart City LLM Listening to Firebase...\n")

last_data = None

# -----------------------------------
# LISTENER LOOP
# -----------------------------------

while True:

    data = ref.get()

    if data and data != last_data:

        print("\n🚨 FIREBASE UPDATE DETECTED")
        print(data)

        incident_type = data.get("Incident Type", "unknown")
        location = data.get("location", "Unknown")
        camera = data.get("camera", "Unknown")
        time_event = data.get("started_at", "Unknown")

        event_text = f"""
Incident Type: {incident_type}
Location: {location}
Camera: {camera}
Time: {time_event}
"""

        # -----------------------------------
        # TASK 1 : INCIDENT ANALYSIS
        # -----------------------------------

        task1 = Task(
            description=f"""
Analyze the following surveillance event.

{event_text}

Determine:
1. Incident Type
2. Priority Level
""",
            expected_output="Incident type and priority level",
            agent=incident_agent
        )

        # -----------------------------------
        # TASK 2 : EMERGENCY RESPONSE
        # -----------------------------------

        task2 = Task(
            description="""
Based on the incident analysis determine which emergency services must respond.
""",
            expected_output="Emergency services list",
            agent=response_agent
        )

        # -----------------------------------
        # TASK 3 : DASHBOARD ALERT
        # -----------------------------------

        task3 = Task(
            description=f"""
Create a structured police dashboard alert.

{event_text}

Format:

🚨 INCIDENT ALERT

Incident Type:
Location:
Camera:
Time:
Priority Level:

Emergency Services Notified:
✔ Service
""",
            expected_output="Formatted alert",
            agent=dashboard_agent
        )

        # -----------------------------------
        # TASK 4 : AI EXPLANATION
        # -----------------------------------

        task4 = Task(
            description="Explain why the emergency response was triggered.",
            expected_output="Short explanation",
            agent=explanation_agent
        )

        crew = Crew(
            agents=[
                incident_agent,
                response_agent,
                dashboard_agent,
                explanation_agent
            ],
            tasks=[task1, task2, task3, task4],
            verbose=True
        )

        crew.kickoff()

        print("\n==============================")
        print("🚓 POLICE DASHBOARD ALERT")
        print("==============================")
        print(task3.output)

        print("\n==============================")
        print("🧠 AI DECISION EXPLANATION")
        print("==============================")
        print(task4.output)

        last_data = data

    time.sleep(2)