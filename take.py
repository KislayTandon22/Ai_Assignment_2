import tensorflow as tf

try:
    from tensorflow.keras.models import Sequential
    print("Import successful!")
except Exception as e:
    print("Import failed:", e)