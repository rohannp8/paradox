import sys
try:
    import mediapipe as mp
    print(f"MediaPipe path: {mp.__file__}")
    print(f"Dir(mp): {dir(mp)}")
    
    try:
        import mediapipe.solutions
        print("Successfully imported mediapipe.solutions")
    except ImportError as e:
        print(f"Failed to import mediapipe.solutions: {e}")

    try:
        from mediapipe import solutions
        print("Successfully imported solutions from mediapipe")
    except ImportError as e:
        print(f"Failed to import solutions from mediapipe: {e}")

except ImportError as e:
    print(f"Critical: {e}")
