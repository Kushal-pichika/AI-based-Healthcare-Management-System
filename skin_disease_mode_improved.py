import os
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Load Dataset
dataset_dir = r"E:\Mini Project\ham10000"
csv_path = r"E:\Mini Project\HAM10000_metadata.csv"
df = pd.read_csv(csv_path)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(dataset_dir, x + ".jpg"))
df['dx'] = df['dx'].astype(str)

# Advanced Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_dataframe(
    dataframe=df,
    x_col="image_path",
    y_col="dx",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_dataframe(
    dataframe=df,
    x_col="image_path",
    y_col="dx",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

num_classes = len(train_data.class_indices)

# Use EfficientNetB3 (Better Accuracy)
base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

# Unfreeze last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=x)

# Use Focal Loss for better performance
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,  # Increase epochs for better learning
    verbose=1
)

# Save the model
model.save("skin_disease_model_improved.h5")
print("✅ Model training complete. Saved as 'skin_disease_model_improved.h5'")
