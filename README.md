# Spotify Playlist Success Analysis

## Overview
This repository contains a data science notebook project that analyzes Spotify playlist-level data to identify factors associated with playlist success and to predict whether a playlist will be successful.

The full analysis lives in:

- `Spotify_Successful_Playlist_AmanJain.ipynb`

## Project Objective
Identify the behavioral and content signals that contribute to a playlist being successful, and use those signals to build predictive machine learning models.

## What the Notebook Does
The notebook runs an end-to-end workflow:

1. Imports data and libraries in Google Colab.
2. Performs exploratory data analysis (EDA) on playlist features.
3. Engineers additional engagement, growth, and structure features.
4. Builds a `music_cluster` feature using K-Means clustering over genre/mood metrics.
5. Compares Spotify-owned and personal playlist behavior.
6. Defines success labels based on heuristic categories:
   - Viral
   - Evergreen
   - Niche
   - Combined flag: `successful_playlist`
7. Trains and evaluates multiple classifiers:
   - Logistic Regression
   - LightGBM
   - Random Forest
8. Optionally enriches playlists with Spotify API metadata using Spotipy.

## Key Outputs Reported in the Notebook
- Initial dataset shape: `403,366 x 25`
- Filtered user-base shape: `40,032 x 49`
- Successful playlists share: `6.57%` of filtered user-base
- Spotify-owned playlists in user-base: `0.99%`
- Spotipy analysis subset: `2,976 x 56`

## Model Results (as shown in executed notebook cells)
- Logistic Regression accuracy: `0.8072`
- LightGBM:
  - Accuracy: `0.7645`
  - F1: `0.2757`
  - Precision: `0.1745`
  - Recall: `0.6563`
- Random Forest:
  - Accuracy: `0.8123`
  - F1: `0.2778`
  - Precision: `0.1884`
  - Recall: `0.5283`

## Tech Stack
- Python
- pandas, numpy, scipy
- scikit-learn
- LightGBM
- seaborn, matplotlib
- Spotipy

## Data and Environment Notes
This notebook is set up for Google Colab and references files in Google Drive (for example `/content/drive/My Drive/Spotify/...`).

To run successfully, you need:

- Access to the source data files referenced in the notebook
- Spotify API credentials for Spotipy-based sections
- A compatible Python environment with required packages installed

## Current Repository Scope
This repository currently contains the notebook only and does not yet include:

- A separate dataset in-repo
- A reproducible local environment file (`requirements.txt`, `environment.yml`, etc.)
- A script-based pipeline outside the notebook

