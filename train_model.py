import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set paths
metadata_path = os.path.join("HAM10000", "HAM10000_metadata.csv")
image_dir = os.path.join("HAM10000", "image_folder")

# Load metadata
df = pd.read_csv(metadata_path)

# Debug: Show column names
print("Columns in dataset:", df.columns)

# Append .jpg to image_id to match actual image filenames
df['image_id'] = df['image_id'] + ".jpg"

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

# Image data generators
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col='image_id',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_dir,
    x_col='image_id',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Get number of classes
num_classes = len(train_gen.class_indices)

# Build CNN model
model = Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save model
model.save("skin_disease_model.h5")

print("✅ Model trained and saved as 'skin_disease_model.h5'")
