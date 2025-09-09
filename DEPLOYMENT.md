# Deployment Guide - Stock Analyzer

**Author: Vikas Ramaswamy**

## Local Development

### Run Streamlit App Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py

# Or use the runner script
python run_streamlit.py
```

The app will be available at: http://localhost:8501

## Streamlit Cloud Deployment

### Option 1: Direct GitHub Deployment
1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and `streamlit_app.py` as the main file
5. Deploy!

### Option 2: Manual Upload
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Create a new app
3. Upload your project files
4. Set `streamlit_app.py` as the main file
5. Deploy!

## Other Hosting Options

### Heroku
1. Create a `Procfile`:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy to Heroku

### Railway
1. Connect your GitHub repository
2. Railway will auto-detect Streamlit
3. Deploy automatically

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Environment Variables
No environment variables required - the app uses simulated sentiment data for demo purposes.

## Features Available
- Single stock analysis with sentiment
- Popular stocks dashboard
- Custom portfolio analysis
- Real-time ML predictions
- Interactive charts and metrics
- Mobile-responsive design

## Performance Notes
- First analysis may take 5-10 seconds (model training)
- Subsequent analyses are faster due to caching
- Dashboard updates can be resource-intensive for multiple stocks

---