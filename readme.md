# Skin Disease Classification using EfficientNetB3

## Overview
This project is a deep learning model for classifying skin diseases using images. It utilizes the **EfficientNetB3** architecture for feature extraction and is trained on a dataset of skin disease images. The model is trained with **data augmentation**, **focal loss**, and **transfer learning** to improve accuracy.

## Features
- Uses **EfficientNetB3** for feature extraction.
- Data augmentation for better generalization.
- Supports multi-class classification.
- Utilizes **Focal Loss** to handle class imbalances.
- Saves trained models for future testing.

## Dataset
The dataset used in this project is not included in the repository. It consists of images of various skin diseases with labels. If you wish to use the same dataset, you can download the **HAM10000 dataset** from Kaggle or other medical imaging sources and place the images in the appropriate directory.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/skin-disease-classification.git
   cd skin-disease-classification
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure you have a dataset with images and corresponding labels.

## Training the Model
Run the following command to train the model:
```sh
python skin_disease_model.py
```
This will train the model and save it as `skin_disease_model_improved.h5`.

## Testing the Model
To test the model on new images, run:
```sh
python test_skin_disease_model.py --image test_images/sample.jpg
```
Make sure to provide the correct path to the test image.

## Git Ignore
The repository includes a `.gitignore` file to exclude sensitive and unnecessary files:
- `.env`
- `venv/`
- `__pycache__/`
- `*.h5` (model files)
- `logs/`
- `datasets/`

## Contributing
Feel free to contribute by creating issues or submitting pull requests.

## License
This project is open-source and available under the [MIT License](LICENSE).

