# Project Documentation

Welcome! This project provides a forward-looking portfolio analysis engine offering user optionality between historical analysis, or user defined valuation. It allows investors, students, analysts, and competition teams to test portfolio outcomes using custom expected returns, volatilities, and correlations and explore probabilistic results via Monte Carlo simulation.

---

## Overview

Instead of only relying on historical returns, this simulator allows  you to:

Define your own annual return and risk assumptions for each asset

Upload or construct a correlation matrix (Excel, manual entry, CSV, or estimated from prices)

Run thousands of Monte Carlo paths over custom time horizons

Optimize portfolio weights for a target final value

Evaluate portfolio risk and performance metrics

This tool was originally built for high-quality investment strategy analysis and forward-looking portfolio design.

---

## Features

- Forward-looking Monte Carlo simulation
- Uses user-supplied return and risk parameters

- Correlation input flexibility, upload Excel triangle matrices, paste CSV matrices, estimate from recent prices via yfinance

- or Manual N×N entry

- Weight optimizer for target outcomes
Searches over feasible allocations and estimates the probability of reaching a target final value

-Risk analytics such as: 

Sharpe ratio

Sortino ratio

Max drawdown

Probability of exceeding a target wealth level

- Interactive Streamlit interface
No coding required to run simulations or visualize results

---

## Getting Started

Follow these steps to get started with this project:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/anyastha/Public-Wharton-Monte-Carlo.git

   ```

2. **Navigate to the project directory**  
   ```bash
   cd Public-Wharton-Monte-Carlo
   ```

3. **Read this README for further instructions**

- install dependencies (pip install -r requirements.txt)

---

## File Structure

| File/Folder   | Purpose/Description              |
|:--------------|:---------------------------------|
| `README.md`  | Project documentation|
| `app.py`        | runner of code  |
| `data/`      |different metrics of investment value  |


---

## Usage

- launch app using streamlit run app.py
- once the app starts enter tickers and portfolio weights
- provide correlation matrices using one of the available modes
- run the Monte carlo simulation to generate portfolio paths
- Review final value distribution, performance metrics, drawdowns etc..
