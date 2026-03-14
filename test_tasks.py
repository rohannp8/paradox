import mediapipe as mp
try:
    print(f"Tasks: {dir(mp.tasks)}")
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions
    print("Tasks Vision imported")
except Exception as e:
    print(f"Error: {e}")
