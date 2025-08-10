# TDS-Project-2-2025

A FastAPI-based web service that analyzes questions using LLM and returns structured responses.

## Features

- Accepts questions via multipart form data
- Uses AIPipe API for LLM analysis
- Returns structured JSON responses
- Handles correlation analysis with numeric outputs
- Generates scatterplot visualizations as base64 PNG images

## Usage

1. Start the server: `python3 app.py`
2. Send POST request to `/api/` with `questions.txt` file
3. Receive JSON array with analysis results

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Requests
- Python-multipart