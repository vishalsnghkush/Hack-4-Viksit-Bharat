import mediapipe
try:
    import mediapipe.python.solutions as solutions
    print("Imported mediapipe.python.solutions")
except ImportError as e:
    print(f"Failed direct import: {e}")

try:
    from mediapipe.python import solutions
    print("From mediapipe.python import solutions success")
except ImportError as e:
    print(f"Failed from import: {e}")
