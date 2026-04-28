import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Dataset Configuration [cite: 29, 31, 42]
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 9  # Aphids, Armyworm, Beetle, Bollworm, Grasshopper, Mites, Mosquito, Sawfly, Stem Borer

# 2. CNN Architecture [cite: 36, 38, 40]
def build_model():
    model = models.Sequential([
        # Block 1 [cite: 38]
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2 [cite: 38]
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3 [cite: 38]
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification Head [cite: 38, 40]
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Prevent overfitting [cite: 40]
        layers.Dense(NUM_CLASSES, activation='softmax') # Multi-class output [cite: 39]
    ])
    
    # 3. Compile Model [cite: 41]
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize the system
smart_pest_detector = build_model()
smart_pest_detector.summary()

print("Smart Pest Detection System initialized successfully.")