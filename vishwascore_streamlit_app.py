import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from databricks import sql
import os

# Page configuration
st.set_page_config(
    page_title="VishwaScore Explorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏦 VishwaScore Explorer Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Alternative Credit Scoring for India's Credit-Invisible Population")
st.markdown("---")

# Database connection function
@st.cache_resource
def get_connection():
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )

# Load data function
@st.cache_data(ttl=600)
def load_data():
    conn = get_connection()
    query = "SELECT * FROM workspace.default.vishwascore_dashboard"
    with conn.cursor() as cursor:
        cursor.execute(query)
        df = cursor.fetchall_arrow().to_pandas()
    return df

# Load data
try:
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Persona filter
    personas = st.sidebar.multiselect(
        "Select Persona",
        options=sorted(df['persona'].unique()),
        default=sorted(df['persona'].unique())
    )
    
    # Score category filter
    score_categories = st.sidebar.multiselect(
        "Select Score Category",
        options=sorted(df['score_category'].unique()),
        default=sorted(df['score_category'].unique())
    )
    
    # EMI filter
    emi_filter = st.sidebar.selectbox(
        "Has EMI",
        options=["All", "Yes", "No"]
    )
    
    # Insurance filter
    insurance_filter = st.sidebar.selectbox(
        "Has Insurance",
        options=["All", "Yes", "No"]
    )
    
    # Digital user filter
    digital_filter = st.sidebar.selectbox(
        "High Digital User",
        options=["All", "Yes", "No"]
    )
    
    # User ID search
    st.sidebar.markdown("---")
    st.sidebar.header("🔎 Search User")
    user_search = st.sidebar.text_input("Enter User ID")
    
    # Apply filters
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['persona'].isin(personas)]
    filtered_df = filtered_df[filtered_df['score_category'].isin(score_categories)]
    
    if emi_filter != "All":
        filtered_df = filtered_df[filtered_df['has_emi'] == emi_filter]
    if insurance_filter != "All":
        filtered_df = filtered_df[filtered_df['has_insurance'] == insurance_filter]
    if digital_filter != "All":
        filtered_df = filtered_df[filtered_df['high_digital_user'] == digital_filter]
    if user_search:
        filtered_df = filtered_df[filtered_df['user_id'].str.contains(user_search, case=False)]
    
    # Key Metrics Row
    st.header("📊 Key Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Users", f"{len(filtered_df):,}")
    
    with col2:
        avg_score = filtered_df['predicted_vishwascore'].mean()
        st.metric("Average VishwaScore", f"{avg_score:.0f}")
    
    with col3:
        poor_count = len(filtered_df[filtered_df['score_category'] == 'Poor'])
        st.metric("Poor Score Users", f"{poor_count:,}")
    
    with col4:
        very_poor_count = len(filtered_df[filtered_df['score_category'] == 'Very Poor'])
        st.metric("Very Poor Score Users", f"{very_poor_count:,}")
    
    with col5:
        high_risk = len(filtered_df[filtered_df['bounce_count'] > 0])
        st.metric("High Risk Users", f"{high_risk:,}", delta=None if high_risk == 0 else f"-{high_risk}")
    
    with col6:
        digital_adoption = filtered_df['digital_adoption_rate'].mean() * 100
        st.metric("Digital Adoption", f"{digital_adoption:.1f}%")
    
    st.markdown("---")
    
    # Visualizations Row 1
    st.header("📈 Score Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Score Distribution
        fig_dist = px.histogram(
            filtered_df, 
            x='predicted_vishwascore',
            nbins=30,
            title="VishwaScore Distribution",
            labels={'predicted_vishwascore': 'VishwaScore', 'count': 'Number of Users'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_dist.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Average Score by Persona
        persona_avg = filtered_df.groupby('persona')['predicted_vishwascore'].mean().sort_values()
        fig_persona = px.bar(
            x=persona_avg.values,
            y=persona_avg.index,
            orientation='h',
            title="Average Score by Persona",
            labels={'x': 'Average VishwaScore', 'y': 'Persona'},
            color=persona_avg.values,
            color_continuous_scale='viridis'
        )
        fig_persona.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_persona, use_container_width=True)
    
    with col3:
        # Score Category Pie Chart
        category_counts = filtered_df['score_category'].value_counts()
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Score Category Distribution",
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations Row 2
    st.header("🎯 Deep Dive Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Income vs Score Scatter
        fig_scatter = px.scatter(
            filtered_df,
            x='avg_monthly_income',
            y='predicted_vishwascore',
            color='persona',
            title="Income vs VishwaScore Correlation",
            labels={
                'avg_monthly_income': 'Monthly Income (₹)',
                'predicted_vishwascore': 'VishwaScore',
                'persona': 'Persona'
            },
            hover_data=['user_id', 'score_category'],
            opacity=0.6
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Component Score Comparison
        component_data = filtered_df.groupby('persona').agg({
            'payment_behaviour_score': 'mean',
            'digital_flow_score': 'mean',
            'income_stability_component': 'mean'
        }).reset_index()
        
        fig_component = go.Figure()
        fig_component.add_trace(go.Bar(
            name='Payment Behaviour',
            x=component_data['persona'],
            y=component_data['payment_behaviour_score'],
            marker_color='#636EFA'
        ))
        fig_component.add_trace(go.Bar(
            name='Digital Flow',
            x=component_data['persona'],
            y=component_data['digital_flow_score'],
            marker_color='#EF553B'
        ))
        fig_component.add_trace(go.Bar(
            name='Income Stability',
            x=component_data['persona'],
            y=component_data['income_stability_component'],
            marker_color='#00CC96'
        ))
        
        fig_component.update_layout(
            title="Component Score Comparison by Persona",
            barmode='group',
            xaxis_title="Persona",
            yaxis_title="Average Score",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_component, use_container_width=True)
    
    st.markdown("---")
    
    # Top Performers
    st.header("🏆 Top 10 High Performers")
    top_performers = filtered_df.nlargest(10, 'predicted_vishwascore')[
        ['user_id', 'persona', 'predicted_vishwascore', 'avg_monthly_income', 
         'emi_regularity_score', 'digital_adoption_rate', 'savings_ratio']
    ].copy()
    
    # Format columns
    top_performers['avg_monthly_income'] = top_performers['avg_monthly_income'].apply(lambda x: f"₹{x:,.0f}")
    top_performers['emi_regularity_score'] = top_performers['emi_regularity_score'].apply(lambda x: f"{x:.2%}")
    top_performers['digital_adoption_rate'] = top_performers['digital_adoption_rate'].apply(lambda x: f"{x:.2%}")
    top_performers['savings_ratio'] = top_performers['savings_ratio'].apply(lambda x: f"{x:.3f}")
    
    top_performers.columns = ['User ID', 'Persona', 'VishwaScore', 'Monthly Income', 
                               'EMI Regularity', 'Digital Adoption', 'Savings Ratio']
    
    st.dataframe(
        top_performers,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Score Improvement Potential
    st.header("💡 Score Improvement Potential")
    st.markdown("**Users in 'Very Poor' category with high digital adoption (>50%) and no bounces**")
    
    improvement_df = filtered_df[
        (filtered_df['score_category'] == 'Very Poor') &
        (filtered_df['digital_adoption_rate'] > 0.5) &
        (filtered_df['bounce_count'] == 0)
    ][['user_id', 'persona', 'predicted_vishwascore', 'digital_adoption_rate', 
       'avg_monthly_income', 'savings_ratio']].head(20).copy()
    
    if len(improvement_df) > 0:
        improvement_df['digital_adoption_rate'] = improvement_df['digital_adoption_rate'].apply(lambda x: f"{x:.2%}")
        improvement_df['avg_monthly_income'] = improvement_df['avg_monthly_income'].apply(lambda x: f"₹{x:,.0f}")
        improvement_df['savings_ratio'] = improvement_df['savings_ratio'].apply(lambda x: f"{x:.3f}")
        
        improvement_df.columns = ['User ID', 'Persona', 'Current Score', 'Digital Adoption', 
                                   'Monthly Income', 'Savings Ratio']
        
        st.dataframe(
            improvement_df,
            use_container_width=True,
            hide_index=True
        )
        st.info(f"🎯 **{len(filtered_df[(filtered_df['score_category'] == 'Very Poor') & (filtered_df['digital_adoption_rate'] > 0.5) & (filtered_df['bounce_count'] == 0)])} users** are prime candidates for score improvement coaching!")
    else:
        st.info("No users match the improvement potential criteria in the current filter selection.")
    
    st.markdown("---")
    
    # Detailed User Table
    st.header("📋 Detailed User Data")
    
    display_df = filtered_df[[
        'user_id', 'persona', 'predicted_vishwascore', 'score_category',
        'payment_behaviour_score', 'digital_flow_score', 'income_stability_component',
        'avg_monthly_income', 'emi_regularity_score', 'digital_adoption_rate',
        'savings_ratio', 'total_transactions', 'active_months',
        'has_emi', 'has_insurance', 'bounce_count'
    ]].copy()
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Footer
    st.markdown("---")
    footer_text = """
    ### 🚀 VishwaScore Architecture Highlights
    * **Training Data**: 27.3M transactions, 100K users
    * **Model Performance**: R²=0.8938, RMSE=14.5
    * **Features**: 73 engineered features across 4 categories
    * **Pipeline**: Bronze → Silver → Gold (Medallion architecture with DLT)
    * **Personas**: 6 India-specific (Farmer, Salaried, Gig Worker, Casual User, Diverse Spender, SHG Woman)
    """
    st.markdown(footer_text)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure you have configured the Databricks connection environment variables:")
    st.code("""
export DATABRICKS_SERVER_HOSTNAME="your-workspace.cloud.databricks.com"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/xxxxx"
export DATABRICKS_TOKEN="your-access-token"
    """)
