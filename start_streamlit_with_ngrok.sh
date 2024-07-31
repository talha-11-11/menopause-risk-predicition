#!/bin/bash

# Start Streamlit
streamlit run app/app.py &

# Start ngrok
./ngrok http 8501
