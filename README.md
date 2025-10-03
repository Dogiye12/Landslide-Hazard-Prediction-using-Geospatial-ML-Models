# Landslide Hazard Prediction using Geospatial ML Models

This repository provides a **synthetic geospatial dataset** and **machine learning pipeline** for predicting landslide hazards.  
The project demonstrates how **GIS-based features** (e.g., slope, elevation, rainfall, lithology, landcover, distance to rivers/faults/roads, NDVI, soil clay content) can be integrated with **ML algorithms** to assess landslide susceptibility in a reproducible manner.  

---

## 📂 Project Structure
.
├── synthetic_landslide_dataset.xlsx # Generated synthetic dataset
├── landslide_hazard_geospatial_ml.py # Python script to build dataset & ML models
├── roc_curve.png # ROC curve of the best model
├── pr_curve.png # Precision-Recall curve
├── confusion_matrix.png # Confusion matrix of predictions
├── feature_importance.png # Top predictors contributing to landslide hazard
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 Features
- Generates a **synthetic dataset** of 5,000 geospatial points.  
- Incorporates multiple features: **elevation, slope, rainfall, lithology, landcover, NDVI, distances to rivers/faults/roads, soil clay content**.  
- Trains and compares **Logistic Regression**, **Random Forest**, and **Gradient Boosting** classifiers.  
- Evaluates models using **ROC-AUC, Precision-Recall AUC, confusion matrix, and feature importance**.  
- Saves results to `.csv`, `.xlsx`, and `.png` outputs for analysis.  

---

## 📊 Example Outputs
- **ROC Curve**: Evaluates model discrimination ability.  
- **Precision-Recall Curve**: Shows balance of positive predictions.  
- **Confusion Matrix**: Summarizes classification accuracy.  
- **Feature Importance**: Highlights key drivers of landslide hazard.  

---

## 🛠️ Requirements
Install dependencies via `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn openpyxl
▶️ Usage
Clone the repository:

bash
Copy code
git clone https://github.com/username/landslide-hazard-ml.git
cd landslide-hazard-ml
Run the Python script to generate dataset and train models:

bash
Copy code
python landslide_hazard_geospatial_ml.py
View outputs:

synthetic_landslide_dataset.xlsx → dataset

roc_curve.png, pr_curve.png, confusion_matrix.png, feature_importance.png → model results

📖 Background
Landslides are among the most frequent and damaging natural hazards in mountainous regions. This project demonstrates how machine learning and geospatial analysis can be combined to:

Identify areas at risk of landslides

Support disaster preparedness and risk reduction

Provide a reproducible framework for hazard mapping

Although this dataset is synthetic, the workflow can be applied to real-world geospatial datasets.

👨‍💻 Author
Amos Meremu Dogiye
Geospatial & Machine Learning Enthusiast

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

⭐ Acknowledgements
Inspired by research in GIS-based landslide susceptibility mapping

Uses open-source geospatial & ML libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

