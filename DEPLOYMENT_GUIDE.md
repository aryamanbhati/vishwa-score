# 🚀 STREAMLIT CLOUD DEPLOYMENT GUIDE

## Step 1: Get Your Databricks Credentials

### A. Server Hostname
- Already identified: **adb-7474646039518887.7.azuredatabricks.net**

### B. SQL Warehouse HTTP Path
1. In Databricks, go to **SQL Warehouses** (left sidebar)
2. Click on any running warehouse (or start one)
3. Click **Connection details** tab
4. Copy the **HTTP Path** (looks like: /sql/1.0/warehouses/90835408decf3669)

### C. Personal Access Token
1. In Databricks, click your profile (top right) → **User Settings**
2. Go to **Developer** → **Access Tokens**
3. Click **Generate New Token**
4. Set lifetime: 90 days (or longer)
5. Copy the token immediately (you won't see it again!)

---

## Step 2: Push to GitHub

### A. Initialize Git Repository
```bash
# Navigate to your project folder (after downloading files)
cd /path/to/vishwascore-dashboard

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "VishwaScore Streamlit Dashboard - Initial commit"
```

### B. Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: **vishwascore-dashboard**
3. Description: "VishwaScore Alternative Credit Scoring Dashboard"
4. Make it **Public** (required for free Streamlit Cloud)
5. DON'T initialize with README (we already have files)
6. Click **Create repository**

### C. Push Code
```bash
# Connect to your new GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/vishwascore-dashboard.git

# Push code
git branch -M main
git push -u origin main
```

---

## Step 3: Deploy on Streamlit Cloud

### A. Sign Up for Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click **Sign up** (use your GitHub account)
3. Authorize Streamlit to access your repositories

### B. Create New App
1. Click **New app** button
2. Fill in the form:
   - **Repository**: YOUR_USERNAME/vishwascore-dashboard
   - **Branch**: main
   - **Main file path**: vishwascore_streamlit_app.py
   - **App URL**: vishwascore-explorer (or your choice)

### C. Add Secrets
1. Click **Advanced settings**
2. In the **Secrets** section, paste this TOML format:

```toml
DATABRICKS_SERVER_HOSTNAME = "adb-7474646039518887.7.azuredatabricks.net"
DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/YOUR_WAREHOUSE_ID"
DATABRICKS_TOKEN = "YOUR_PERSONAL_ACCESS_TOKEN"
```

3. Replace YOUR_WAREHOUSE_ID and YOUR_PERSONAL_ACCESS_TOKEN with real values
4. Click **Deploy!**

---

## Step 4: Wait for Deployment

- Streamlit Cloud will install dependencies (~2-3 minutes)
- Watch the logs for any errors
- Once complete, you'll get a public URL like:
  **https://vishwascore-explorer.streamlit.app**

---

## Step 5: Test Your Dashboard

1. Open the public URL
2. Check all filters work
3. Verify data loads (should see 100K users)
4. Test visualizations render correctly
5. Share the URL with your team/judges! 🎉

---

## 🔧 Troubleshooting

### Error: "Connection failed"
- Check your SQL Warehouse is running in Databricks
- Verify HTTP Path is correct (including /sql/1.0/warehouses/ prefix)
- Ensure token has SQL Warehouse access permissions

### Error: "Module not found"
- Check requirements.txt has all dependencies
- Verify file names are correct (case-sensitive)

### Error: "Table not found"
- Verify table exists: workspace.default.vishwascore_dashboard
- Check token has READ permission on the table

### Slow Loading
- Normal for first load (fetching 100K rows)
- Subsequent loads use 10-minute cache
- Consider adding LIMIT for testing

---

## 📊 What Judges Will See

✅ **Professional UI**: Clean, modern Streamlit interface
✅ **Real-time Data**: Live connection to 100K users
✅ **Interactive Filters**: 6 filter dimensions
✅ **Rich Visualizations**: 5 Plotly charts
✅ **Business Insights**: Top performers, improvement candidates
✅ **Production Quality**: Error handling, caching, documentation

---

## 🎓 Demo Tips for Hackathon

1. **Start with Overview**: Show key metrics (100K users, 37% digital adoption)
2. **Filter by Persona**: Demonstrate Farmer vs Salaried differences
3. **Highlight Insights**: Point out Top 10 performers (all Salaried, 532-542 scores)
4. **Show Opportunity**: 1,784 improvement candidates ready for coaching
5. **Explain Architecture**: Bronze→Silver→Gold pipeline, 73 features, R²=89%

---

## 📞 Need Help?

- Streamlit Docs: https://docs.streamlit.io
- Community Forum: https://discuss.streamlit.io
- Databricks SQL Connector: https://docs.databricks.com/sql/api/sql-connector.html

🚀 Good luck with your hackathon deployment!
