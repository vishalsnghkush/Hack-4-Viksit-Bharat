import mediapipe
print(dir(mediapipe))
try:
    print(mediapipe.solutions)
except AttributeError as e:
    print(f"Error: {e}")
