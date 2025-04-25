# Counterfeit RFID Tag Detection Using Machine Learning

This project presents a lightweight, data-driven approach for detecting counterfeit RFID tags in smart environments using machine learning techniques. By analyzing RSSI (Received Signal Strength Indicator) patterns from passive RFID tags, the system identifies counterfeit behavior without requiring cryptographic protocols or specialized hardware.

## Project Structure

- `datasets/`
  - `RFID - Activity, Location and PID Labels.csv`: Raw original RFID dataset used in the project.
  - `Processed_RFID_Features_Windowed.csv`: Dataset after preprocessing and feature extraction.
- `code/`
  - `preprocessing.py`: Script for data cleaning, feature extraction, and counterfeit tag simulation.
  - `ML1.py`: Script for model training, evaluation, and visualization (Random Forest, SVM, XGBoost).

## Approach

1. **Data Preprocessing**: 
   - Time-windowed segmentation (30s bins).
   - Extraction of temporal, spatial, and RSSI-based statistical features.

2. **Counterfeit Simulation**:
   - Injected subtle signal anomalies to mimic cloned RFID tags.

3. **Machine Learning Models**:
   - Trained Random Forest, SVM, and XGBoost classifiers.
   - Evaluated using 5-fold cross-validation.
   - Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve.

## How to Run

1. Place the original dataset in the `datasets/` folder.
2. Run `code/preprocessing.py` to generate the processed dataset.
3. Run `code/ML1.py` to train models and evaluate performance.

## Results Summary

- Random Forest and XGBoost achieved F1-scores above 95%.
- Demonstrated that lightweight, signal-based analysis can effectively detect counterfeit RFID tags.

## Applications

- Smart homes
- Hospitals and healthcare systems
- Retail inventory management
- Secure access control systems

## Authors

- Charishma Annamaneedi
- Hemanth Cheela Ramu

## License

This project is made available for academic and research purposes.

