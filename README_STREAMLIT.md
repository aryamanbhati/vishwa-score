# VishwaScore Explorer - Streamlit Dashboard

## 🎯 Overview
Interactive Streamlit dashboard for VishwaScore credit scoring system with real-time analytics.

## 📊 Features
- **6 Key Metrics**: Total users, average score, risk analysis, digital adoption
- **Interactive Filters**: Persona, score category, EMI, insurance, digital usage
- **5 Visualizations**: Score distribution, persona analysis, income correlation, component comparison
- **Top Performers**: Highest scoring users with detailed metrics
- **Improvement Potential**: Identifies users ready for score improvement

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export DATABRICKS_SERVER_HOSTNAME="your-workspace.cloud.databricks.com"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/your-warehouse-id"
export DATABRICKS_TOKEN="your-personal-access-token"
```

**Get these values from Databricks:**
- **Server Hostname**: Workspace URL (e.g., adb-7474646039518887.7.azuredatabricks.net)
- **HTTP Path**: SQL Warehouse → Connection Details → HTTP Path
- **Token**: User Settings → Developer → Access Tokens → Generate New Token

### 3. Run the App
```bash
streamlit run vishwascore_streamlit_app.py
```

The app will open at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

### Option 1: Streamlit Community Cloud (Free)

1. **Push to GitHub**
   ```bash
   git init
   git add vishwascore_streamlit_app.py requirements.txt
   git commit -m "VishwaScore Streamlit Dashboard"
   git remote add origin https://github.com/your-username/vishwascore-dashboard
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `vishwascore_streamlit_app.py` as main file
   - Add Secrets in Settings:
     ```toml
     DATABRICKS_SERVER_HOSTNAME = "your-workspace.cloud.databricks.com"
     DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/xxxxx"
     DATABRICKS_TOKEN = "your-token"
     ```
   - Click Deploy!

### Option 2: Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY vishwascore_streamlit_app.py .
   EXPOSE 8501
   CMD ["streamlit", "run", "vishwascore_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   docker build -t vishwascore-dashboard .
   docker run -p 8501:8501      -e DATABRICKS_SERVER_HOSTNAME="your-hostname"      -e DATABRICKS_HTTP_PATH="your-path"      -e DATABRICKS_TOKEN="your-token"      vishwascore-dashboard
   ```

## 📊 Dashboard Components

### Key Metrics
- Total Users (100K)
- Average VishwaScore (~439)
- Risk Users (bounce_count > 0)
- Digital Adoption Rate (37.2%)

### Visualizations
1. **Score Distribution**: Histogram showing score spread
2. **Persona Analysis**: Average scores by user type
3. **Score Categories**: Pie chart breakdown
4. **Income vs Score**: Scatter plot correlation
5. **Component Comparison**: Payment/Digital/Income scores by persona

### Data Tables
- **Top 10 Performers**: Highest scoring users
- **Improvement Potential**: Users ready for coaching (Very Poor + High Digital + No Bounces)
- **Detailed User Table**: Full dataset with 16 columns

## 🔒 Security Notes
- **Never commit tokens** to version control
- Use environment variables or secrets management
- Rotate tokens regularly
- Restrict SQL Warehouse permissions

## 📈 Architecture
```
Databricks Tables (workspace.default.vishwascore_dashboard)
          ↓
Databricks SQL Connector
          ↓
Pandas DataFrame (cached 10 min)
          ↓
Plotly Visualizations
          ↓
Streamlit Frontend
```

## 🎓 For Hackathon Judges
This dashboard demonstrates:
- **Real-time Analytics**: Live connection to 100K user records
- **Interactive Exploration**: 6 filter dimensions
- **Production Architecture**: Cached queries, error handling
- **Business Value**: Clear insights for lenders and users
- **Scalability**: Handles 27.3M transactions

## 📞 Support
For issues or questions, check:
- Databricks connection settings
- SQL Warehouse is running
- Token has correct permissions
