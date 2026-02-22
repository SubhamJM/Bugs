# Clash Royale Win Predictor: AI-Powered Deck Evaluation

![Clash Royale Analytics](https://img.shields.io/badge/Track-Machine_Learning-blueviolet)
![Status](https://img.shields.io/badge/Hackathon-NOOBATHON-orange)
![Category](https://img.shields.io/badge/Sector-Game_Analytics-red)

## 🏰 Project Overview
 In high-level *Clash Royale* play, victory is often determined by deck composition and meta-awareness before the first elixir drops. This project involves building an intelligent deck evaluation system for Supercell's competitive analytics team. The goal is to design a Machine Learning model that predicts match outcomes based on two opposing decks, simulating the intuition of a meta-aware competitive player.

## 🎯 Objectives
* **Outcome Prediction:** Predict which of two decks is more likely to win a matchup.
* **Meta Adaptation:** Ensure the model adapts to evolving gameplay, balance changes, and shifting win conditions.
* **Strategic Insights:** Identify card interactions, archetype strengths (Beatdown, Cycle, Control), and hard counters.


---

## 🛠️ Technical Deliverables

### 1. Feature Engineering
We encode decks using meaningful gameplay features:
* **Card Presence:** Binary vectors for deck composition.
* **Elixir Metrics:** Average elixir cost calculations.
* **Archetype Indicators:** Identification of playstyles like Beatdown, Cycle, or Control.
* **Interaction Signals:** Synergy and counter-relationship features.

### 2. Model Architecture
The system utilizes advanced predictive modeling to output win probabilities:
* **Algorithms:** Implementation of Logistic Regression, XGBoost, or Neural Networks.
* **Symmetry:** Ensuring prediction consistency even when the order of decks is swapped.
* **Generalization:** Robust performance on unseen matchups.

### 3. Data Strategy
* **Meta-Focus:** Prioritizing recent match data to reflect the current competitive landscape.
* **Cleanup:** Discarding data from outdated balance patches and strategies.
* **Integrity:** Strict train-test splitting to prevent data leakage.

---

## ✨ Advanced Features (Brownie Points)
To elevate the solution, the following components are integrated:

* **Counter-Deck Recommender:** Suggests a deck that maximizes winning chances against a specific opponent.
* **Card-Level Analytics:** Visualizations of individual card win rates, hard counters, and synergy graphs.
* **End-to-End UI:** A modern interface (built with Streamlit/Gradio) for real-time predictions and visual insights.

---

## 📊 Evaluation Metrics
The model is evaluated based on:
*  **Primary Stats:** Accuracy, Precision, Recall, and F1-score.
* **Probability:** ROC-AUC and win probability calibration.
* **Qualitative:** Quality of feature engineering and clarity of strategic reasoning.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Scikit-learn / XGBoost / TensorFlow or PyTorch
* Pandas/NumPy for data manipulation

### Dataset
The model is trained using historical competitive match data:
[Kaggle: Clash Royale Games](https://www.kaggle.com/datasets/s1m0n38/clash-royale-games).

---
*Developed for NOOBATHON Machine Learning Track.* 