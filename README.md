# AI Outfit Planner 👗

An image-based clothing classification project that suggests outfit combinations based on weather, occasion, and personal style.

## Features

- Upload a clothing image.
- Classify clothing into six categories: tops, pants, dresses, skirts, shoes, and outerwear.
- Generate outfit suggestions based on selected preferences.
- Interactive interface built using Streamlit.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Scikit-image
- Pillow
- Streamlit
- HOG feature extraction
- Linear Support Vector Classifier (Linear SVC)

## How It Works

1. Load clothing images from the dataset.
2. Extract HOG features and RGB color histograms.
3. Train a machine learning classifier.
4. Evaluate the classifier on validation images.
5. Display predictions and rule-based outfit suggestions in the web application.

## Project Structure

```text
AI-Outfit-Planner/
├── data/
├── src/
├── models/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Train the Model

Ensure the dataset and CSV manifests are available and correctly configured.

```bash
python train_model.py
```

## Run the Application

```bash
python -m streamlit run app.py
```

## Current Limitations

The classifier predicts clothing categories. Outfit suggestions are generated using predefined rules rather than a separately trained outfit compatibility model.

## Future Improvements

- Improve classification accuracy.
- Develop a dedicated outfit compatibility model.
- Integrate live weather information.
- Add wardrobe management and personalized recommendations.
- Deploy the application online.

## Author

**Tanushree Padole**

GitHub: [tanushreepadole](https://github.com/tanushreepadole)

*This project is a work in progress. Model performance and features will be updated as development continues.*
