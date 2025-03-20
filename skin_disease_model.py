import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# ✅ 1. Define Dataset Paths
dataset_dir = r"E:\Mini Project\ham10000"  # Update if necessary
csv_path = r"E:\Mini Project\HAM10000_metadata.csv"

# ✅ 2. Load CSV File
df = pd.read_csv(csv_path)

# ✅ 3. Verify Dataset Directory
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

# ✅ 4. Check If Image Paths Are Correct
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(dataset_dir, x + ".jpg"))

# Print first 5 paths to verify
for path in df['image_path'].values[:5]:
    print(f"Checking {path}: Exists -> {os.path.exists(path)}")

# Convert labels to string
df['dx'] = df['dx'].astype(str)

# ✅ 5. Image Data Generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

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

# ✅ 6. Fix `num_classes` Calculation
num_classes = len(train_data.class_indices)  # ✅ FIXED

# ✅ 7. Load Pretrained Model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  

# ✅ 8. Build Custom Model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=x)

# ✅ 9. Compile & Train Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    verbose=1
)

model.save("skin_disease_model.h5")
print("✅ Model training complete. Saved as 'skin_disease_model.h5'")
