# Bias Heist — Visa Approval Model

## Run the chatbot interview (recommended)
```
pip install -r requirements.txt
streamlit run chat_app.py
```

## Run the plain form version
```
streamlit run app.py
```

## Regenerate from scratch
```
python generate_dataset.py   # writes visa_dataset.csv
python train_model.py        # writes visa_model.joblib
```

See ANSWER_KEY_LEADS_ONLY.md for the hidden bias explanation — do not
share that file with participants.
