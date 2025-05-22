Trading Bot README
Overview
This project is a Python-based trading bot for trading BTC/USDT perpetual futures on the Phemex exchange. The bot uses a LightGBM machine learning model for price prediction, integrates with the Groq API for trade decision-making, and includes a Flask web server for monitoring and controlling the bot. It features technical indicators (e.g., RSI, MACD, Bollinger Bands), trailing stop-loss, and a retraining mechanism for the ML model based on trade outcomes.
Key Features:
Fetches historical and real-time market data using the CCXT library.

Computes technical indicators for ML model input.

Uses LightGBM for price movement prediction.

Consults Groq API for final trade decisions (long/short).

Implements a trailing stop-loss strategy.

Provides a Flask API for starting, stopping, and checking bot status.

Logs actions and errors for debugging.

Disclaimer: This bot is for educational purposes only. Trading involves significant financial risk, especially with leverage. Test thoroughly with small amounts or in a demo environment before using real funds.
Prerequisites
Python 3.8+

Phemex account with API key and secret

Groq API key

Render account for deployment

Git installed

Basic knowledge of Python, Flask, and machine learning

Installation
Clone the Repository
bash

git clone <your-repository-url>
cd <repository-directory>

Set Up a Virtual Environment
bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies
Create a requirements.txt file with the following:

ccxt>=4.0.0
pandas>=2.0.0
numpy>=1.26.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
groq>=0.4.0
flask>=2.0.0

Then run:
bash

pip install -r requirements.txt

Set Environment Variables
Create a .env file in the project root or set environment variables:

API_KEY=<your-phemex-api-key>
SECRET=<your-phemex-secret>
GROQ_API_KEY=<your-groq-api-key>
PORT=10000

Use a tool like python-dotenv to load these variables locally, or set them directly in your environment.

Local Usage
Run the Bot
Ensure environment variables are set, then run:
bash

python trading_bot.py

The bot will start trading and the Flask server will run on http://0.0.0.0:10000.

API Endpoints
GET /status: Check bot status (e.g., running, position, last action).

POST /start: Start the bot.

POST /stop: Stop the bot.

Example using curl:
bash

curl http://localhost:10000/status
curl -X POST http://localhost:10000/start
curl -X POST http://localhost:10000/stop

Logs
Logs are output to the console and can be redirected to a file by modifying the logging configuration in the code.

Deployment on Render
Render is a cloud platform for deploying web applications. Follow these steps to deploy the trading bot.
Step 1: Prepare the Code
Ensure requirements.txt Exists
Verify that the requirements.txt file (listed above) is in the project root.

Add a Procfile
Create a Procfile in the project root with:

web: python trading_bot.py

Modify the Flask Port
Ensure the Flask app uses the PORT environment variable:
python

app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))

This is already included in the provided code.

Commit Changes
bash

git add .
git commit -m "Prepare for Render deployment"

Step 2: Set Up Render
Create a Render Account
Sign up at render.com and log in.

Create a New Web Service
Go to the Render dashboard and click "New" > "Web Service."

Connect your Git repository (push your code to GitHub, GitLab, or Bitbucket first).

Select the repository containing your trading bot.

Configure the Web Service
Runtime: Python

Build Command: pip install -r requirements.txt

Start Command: python trading_bot.py

Instance Type: Choose a free or paid tier based on your needs (free tier is sufficient for testing).

Environment Variables:
API_KEY: Your Phemex API key

SECRET: Your Phemex secret

GROQ_API_KEY: Your Groq API key

PORT: Set to 10000 (or another port if needed)

Add these in the "Environment" section of the Render dashboard.

Deploy
Click "Create Web Service" to deploy. Render will build and start the application. Monitor the build logs for errors.

Step 3: Access the Deployed Bot
Once deployed, Render provides a URL (e.g., https://your-bot.onrender.com).

Access the API endpoints:
GET https://your-bot.onrender.com/status

POST https://your-bot.onrender.com/start

POST https://your-bot.onrender.com/stop

Use tools like curl or Postman to interact with the API.

Step 4: Monitor and Debug
Logs: Check logs in the Render dashboard under the "Logs" tab.

Scaling: If the bot is slow or times out, consider upgrading to a paid Render instance.

Environment Variables: Ensure sensitive keys (API_KEY, SECRET, GROQ_API_KEY) are secure and not hardcoded.

Security Notes
API Keys: Never commit API keys or secrets to the repository. Use environment variables or Render's secure environment variable storage.

Phemex API Restrictions: Restrict your Phemex API key to trading only (disable withdrawals) for security.

Rate Limits: The bot uses enableRateLimit: True in CCXT to avoid hitting exchange rate limits. Monitor for rate limit errors in logs.

Groq API: Ensure your Groq API key has sufficient quota for continuous operation.

Troubleshooting
Bot Not Starting: Check Render logs for missing environment variables or dependency errors.

Exchange Errors: Verify Phemex API key/secret and ensure the BTC/USDT:USDT market is available.

Groq API Errors: Confirm your Groq API key is valid and has quota.

Data Issues: If fetch_historical_data returns empty, check the exchange connection or try reducing the limit parameter.

Flask Errors: Ensure the PORT environment variable matches Render's expectations.

Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs, features, or improvements.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Notes for Deployment
Render Free Tier: The free tier may sleep after inactivity, causing delays in bot startup. Consider a paid tier for continuous operation.

Cost Considerations: Monitor Phemex trading fees and Render costs (if using a paid tier).

Testing: Deploy with a Phemex testnet account first (if available) to avoid financial risk.

This README.md provides a complete guide for setting up, running, and deploying the trading bot on Render. Let me know if you need further assistance or modifications!

