# DNN-streamlit

## Project Overview

The `dnn-streamlit` repository contains an interactive Streamlit web application designed to demonstrate time-series forecasting. Despite the "dnn" (Deep Neural Network) in the project name, the current application uses a **LightGBM** (Gradient Boosting Machine) model to predict future events based on simulated historical patterns.

## The Core Concept

The application models a specific scenario: **Hour-long selections initiated every 5 minutes.**

**The Rule:** Every initiation opens a 60-minute selection window. The user is expected to return *within that specific hour-long window* that they selected.

The machine learning model's goal is to learn the relationship between when these selections are initiated and when the corresponding returns actually happen within those active 60-minute windows.

## Application Features (`src/app.py`)

The Streamlit dashboard walks through the entire Machine Learning lifecycle:

1. **Data Simulation:** Automatically generates 14 days of mock data. It simulates "initiation counts" (incorporating daily cycles and weekend spikes) and corresponding "return times" (simulated to happen strictly within the 1-hour active window after initiation).
2. **Feature Engineering:** Calculates features required for the LightGBM model:
   - Raw initiation counts per 5-minute slot.
   - Rolling/Active counts (how many 1-hour windows are currently open).
   - Time-based features (normalized hour of day, day of week).
3. **Data Preparation:** Formats the sequential data into lookback windows and prediction windows, flattening it so it can be ingested by the LightGBM model.
4. **Model Training & Evaluation:** Provides a user interface to train the LightGBM Regressor. After training, it makes predictions on a test dataset and displays a chart comparing actual vs. predicted return counts, along with the Mean Absolute Error (MAE).

## Running the App

The project uses `uv` and `pyproject.toml` for dependency management.

To run the Streamlit application:
```bash
# Ensure dependencies are installed
uv sync

# Run the app
streamlit run src/app.py
```
