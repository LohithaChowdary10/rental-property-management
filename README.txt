# Rental Price Prediction and Category Classification

## What this app does
- Predicts an estimated monthly rental price using a Random Forest regression model.
- Classifies the rental into Affordable, Moderate, or Premium using a Random Forest classifier.
- Shows a dataset preview and basic model evaluation.

## Run in VS Code
1. Extract this ZIP file.
2. Open VS Code -> File -> Open Folder -> select `rental_price_prediction_app`.
3. Open Terminal -> New Terminal.
4. Run these commands one by one:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

If PowerShell blocks activation, use:
```bash
.venv\Scripts\Activate.ps1
```
If activation is blocked, you can still continue by running `python -m pip install -r requirements.txt`.

5. Install packages:
```bash
python -m pip install -r requirements.txt
```

6. Start the website:
```bash
python -m streamlit run app.py
```

7. A browser tab should open. If it does not, copy the Local URL shown in the terminal (usually http://localhost:8501) into Chrome.

## Important
`rental_data.csv` is generated demo data so the project runs immediately. It is not real market data. For a proper academic result, replace it with a real rental dataset containing these columns:
`area_sqft, bedrooms, bathrooms, location, furnishing, property_type, rent`
The app creates the category labels from rent: below ₹20,000 = Affordable; ₹20,000–₹40,000 = Moderate; above ₹40,000 = Premium. Adjust these thresholds to match your project requirements and local market.
