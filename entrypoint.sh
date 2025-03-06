#!/bin/bash
kaggle datasets download 'mlg-ulb/creditcardfraud'
unzip creditcardfraud.zip
python xgboost_sol.py

