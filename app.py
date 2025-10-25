import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Diabetes Prediction App", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(254,202,87,0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(72,219,251,0.3);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
    h1, h2, h3 {
        color: #2d3436;
        font-weight: 800;
    }
    .success-box {
        background: linear-gradient(135deg, #00b894, #00cec9);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-10px);}
        to {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Advanced Diabetes Prediction System")
st.markdown("### 🔬 AI-Powered Medical Risk Assessment with Individual Patient Analysis")

def create_sample_data():
    """Create a sample diabetes dataset"""
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.randint(60, 200, n_samples),
        'BloodPressure': np.random.randint(40, 122, n_samples),
        'SkinThickness': np.random.randint(10, 99, n_samples),
        'Insulin': np.random.randint(15, 846, n_samples),
        'BMI': np.random.uniform(15, 67, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
        'Outcome': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    mask = np.random.rand(n_samples, len(df.columns)) < 0.1
    for col in df.columns[:-1]:
        df.loc[mask[:, df.columns.get_loc(col)], col] = np.nan
    
    return df

def impute_missing_values(df, method='mean', target_col=None):
    """Impute missing values - works with or without target column"""
    df_imputed = df.copy()
    
    if target_col and target_col in df_imputed.columns:
        features = [col for col in df_imputed.columns if col != target_col]
    else:
        features = df_imputed.columns.tolist()
    
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy='mean')
    
    df_imputed[features] = imputer.fit_transform(df_imputed[features])
    return df_imputed

def save_model(model_name, model, scaler, feature_names, metrics, imputer=None):
    """Save trained model with all components"""
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"saved_models/{model_name.replace(' ', '_')}_{timestamp}.pkl"
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'timestamp': timestamp,
        'model_name': model_name,
        'imputer': imputer
    }
    
    joblib.dump(model_package, model_filename)
    return model_filename

def load_model(model_filename):
    """Load a saved model"""
    return joblib.load(model_filename)

def get_saved_models():
    """Get list of saved models"""
    if not os.path.exists('saved_models'):
        return []
    
    models = []
    for filename in os.listdir('saved_models'):
        if filename.endswith('.pkl'):
            filepath = os.path.join('saved_models', filename)
            try:
                model_pkg = joblib.load(filepath)
                models.append({
                    'filename': filepath,
                    'name': model_pkg['model_name'],
                    'timestamp': model_pkg['timestamp'],
                    'accuracy': model_pkg['metrics'].get('accuracy', 0)
                })
            except:
                pass
    
    return sorted(models, key=lambda x: x['timestamp'], reverse=True)

def explain_individual_prediction(model, X, feature_names, patient_data, prediction, prediction_proba):
    """Explain individual patient prediction with feature contributions - preserves correct risk direction"""
    
    patient_values = patient_data.flatten()
    X_mean = X.mean(axis=0)
    difference = patient_values - X_mean
    
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        contributions = difference * coefficients
        
    elif hasattr(model, 'feature_importances_'):
        contributions = np.zeros(len(feature_names))
        
        pred_original = model.predict_proba(patient_data)[0, 1]
        
        for i in range(len(feature_names)):
            X_modified = patient_data.copy()
            X_modified[0, i] = X_mean[i]
            
            pred_modified = model.predict_proba(X_modified)[0, 1]
            
            marginal_effect = pred_original - pred_modified
            
            contributions[i] = marginal_effect
    else:
        contributions = difference
    
    contributions_df = pd.DataFrame({
        'Feature': feature_names,
        'Patient Value': patient_values,
        'Average Value': X_mean,
        'Difference': difference,
        'Contribution': contributions,
        'Abs Contribution': np.abs(contributions)
    }).sort_values('Abs Contribution', ascending=False)
    
    return contributions_df

def plot_individual_explanation(contributions_df, patient_name, prediction, risk_score):
    """Create waterfall-style plot for individual patient"""
    
    top_features = contributions_df.head(8).copy()
    
    colors = ['#ff6b6b' if c > 0 else '#48dbfb' for c in top_features['Contribution']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_features['Feature'],
        x=top_features['Contribution'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f"{c:.3f}" for c in top_features['Contribution']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.3f}<extra></extra>'
    ))
    
    risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
    risk_color = "#ff6b6b" if prediction == 1 else "#48dbfb"
    
    fig.update_layout(
        title=dict(
            text=f"🔬 Individual Risk Analysis - {patient_name}<br><sub>Diabetes Risk: {risk_level} ({risk_score:.1%} probability)</sub>",
            font=dict(size=18, color=risk_color, family="Arial Black")
        ),
        xaxis_title="Contribution to Diabetes Risk",
        yaxis_title="Medical Features",
        template='plotly_white',
        height=500,
        showlegend=False,
        font=dict(size=12),
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
    
    return fig

def get_medical_explanation(feature, value, avg_value, contribution):
    """Provide medical explanation for each feature"""
    
    explanations = {
        'Glucose': {
            'high': f"🔴 Glucose level ({value:.1f}) is higher than average ({avg_value:.1f}). High blood sugar is the PRIMARY indicator of diabetes - the body cannot properly process sugar.",
            'low': f"🟢 Glucose level ({value:.1f}) is lower than average ({avg_value:.1f}). Normal glucose levels indicate good blood sugar control.",
            'normal': f"🟡 Glucose level ({value:.1f}) is close to average ({avg_value:.1f})."
        },
        'BMI': {
            'high': f"🔴 BMI ({value:.1f}) is higher than average ({avg_value:.1f}). Excess body fat causes insulin resistance, making it harder for cells to absorb glucose.",
            'low': f"🟢 BMI ({value:.1f}) is lower than average ({avg_value:.1f}). Healthy weight reduces diabetes risk.",
            'normal': f"🟡 BMI ({value:.1f}) is close to average ({avg_value:.1f})."
        },
        'Age': {
            'high': f"🔴 Age ({value:.0f}) is higher than average ({avg_value:.1f}). Aging reduces insulin production and increases resistance naturally.",
            'low': f"🟢 Age ({value:.0f}) is younger than average ({avg_value:.1f}). Younger age typically has better metabolic function.",
            'normal': f"🟡 Age ({value:.0f}) is close to average ({avg_value:.1f})."
        },
        'DiabetesPedigreeFunction': {
            'high': f"🔴 Genetic risk ({value:.3f}) is higher than average ({avg_value:.3f}). Family history significantly increases diabetes likelihood.",
            'low': f"🟢 Genetic risk ({value:.3f}) is lower than average ({avg_value:.3f}). Lower genetic predisposition is protective.",
            'normal': f"🟡 Genetic risk ({value:.3f}) is close to average ({avg_value:.3f})."
        },
        'Pregnancies': {
            'high': f"🔴 Number of pregnancies ({value:.0f}) is higher than average ({avg_value:.1f}). Multiple pregnancies stress insulin-producing cells.",
            'low': f"🟢 Number of pregnancies ({value:.0f}) is lower than average ({avg_value:.1f}).",
            'normal': f"🟡 Number of pregnancies ({value:.0f}) is close to average ({avg_value:.1f})."
        },
        'Insulin': {
            'high': f"🔴 Insulin level ({value:.1f}) is higher than average ({avg_value:.1f}). High insulin may indicate insulin resistance.",
            'low': f"🟢 Insulin level ({value:.1f}) is lower than average ({avg_value:.1f}).",
            'normal': f"🟡 Insulin level ({value:.1f}) is close to average ({avg_value:.1f})."
        },
        'BloodPressure': {
            'high': f"🔴 Blood pressure ({value:.1f}) is higher than average ({avg_value:.1f}). Hypertension often coexists with diabetes (metabolic syndrome).",
            'low': f"🟢 Blood pressure ({value:.1f}) is lower than average ({avg_value:.1f}). Healthy blood pressure is beneficial.",
            'normal': f"🟡 Blood pressure ({value:.1f}) is close to average ({avg_value:.1f})."
        },
        'SkinThickness': {
            'high': f"🔴 Skin thickness ({value:.1f}) is higher than average ({avg_value:.1f}). May indicate higher body fat percentage.",
            'low': f"🟢 Skin thickness ({value:.1f}) is lower than average ({avg_value:.1f}).",
            'normal': f"🟡 Skin thickness ({value:.1f}) is close to average ({avg_value:.1f})."
        }
    }
    
    if feature not in explanations:
        return f"Value: {value:.2f} (Average: {avg_value:.2f})"
    
    diff_pct = abs((value - avg_value) / avg_value) if avg_value != 0 else 0
    
    if diff_pct > 0.15:
        if value > avg_value:
            return explanations[feature]['high']
        else:
            return explanations[feature]['low']
    else:
        return explanations[feature]['normal']

sidebar_selection = st.sidebar.selectbox(
    "📋 Navigation",
    ["🏠 Home", "📚 Tutorial - How to Use", "📊 Data Upload & Analysis", "🤖 Train Models", "🔮 Make Predictions", "👤 Individual Analysis"]
)

if sidebar_selection == "🏠 Home":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; color: white; margin-bottom: 20px;'>
        <h2>🎯 Welcome to Advanced Diabetes Prediction System</h2>
        <p style='font-size: 18px;'>Medical-grade AI system for diabetes risk assessment with individual patient explanations</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🔬 Advanced AI</h3>
            <p>Multiple machine learning models including Random Forest, XGBoost, and Logistic Regression for accurate predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>👥 Individual Analysis</h3>
            <p>Detailed explanations for each patient showing exactly why they may be at risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>💾 Save & Reuse</h3>
            <p>Train once, save your model, and use it repeatedly for new patient predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🌟 Key Features")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        ✅ **Works for Both Males and Females**
        - Pregnancy column is optional
        - Gender-inclusive analysis
        
        ✅ **Handles Missing Data Automatically**
        - Multiple imputation methods (Mean, Median, KNN)
        - Visual comparison of methods
        
        ✅ **Works With or Without Outcome Column**
        - Training mode: Use data with outcomes
        - Prediction mode: Predict on new patients without labels
        
        ✅ **Beautiful Visualizations**
        - Animated charts and graphs
        - Color-coded risk levels (Red/Yellow/Blue)
        - Interactive plots
        """)
    
    with features_col2:
        st.markdown("""
        ✅ **Individual Patient Explanations**
        - See exactly why each person may have diabetes
        - Feature contribution analysis
        - Medical interpretations
        
        ✅ **Model Saving & Loading**
        - Save trained models for future use
        - No need to retrain every time
        - Track model performance
        
        ✅ **Professional Medical Reports**
        - Downloadable predictions
        - Detailed risk assessments
        - Ready for clinical review
        
        ✅ **Interactive Tutorial**
        - Step-by-step guide
        - Learn as you go
        - Example datasets included
        """)
    
    st.markdown("---")
    st.info("👈 **Get Started:** Use the navigation menu on the left to begin. Start with the Tutorial if this is your first time!")

elif sidebar_selection == "📚 Tutorial - How to Use":
    st.header("📚 Interactive Tutorial - How to Use This Application")
    
    st.markdown("""
    <div class='success-box'>
        <h3>🎓 Welcome to the Tutorial!</h3>
        <p>This guide will walk you through every step of using the diabetes prediction system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Step 1: Data Upload", "🔧 Step 2: Data Preparation", "🤖 Step 3: Train Models", "🔮 Step 4: Make Predictions", "👤 Step 5: Understand Results"])
    
    with tab1:
        st.subheader("📁 Step 1: Upload Your Data")
        
        st.markdown("""
        ### What You Need:
        
        A CSV file with patient health data. The system works in two modes:
        
        **🎯 Training Mode (with Outcome column):**
        - Use this when you have historical data with known diabetes outcomes
        - Required columns: Glucose, BMI, Age, etc.
        - Optional: Outcome column (0 = No Diabetes, 1 = Has Diabetes)
        
        **🔮 Prediction Mode (without Outcome column):**
        - Use this for new patients where you want to predict diabetes risk
        - Same columns except no Outcome needed
        
        ### Expected Columns:
        
        | Column | Description | Example Values | Required |
        |--------|-------------|----------------|----------|
        | Pregnancies | Number of times pregnant | 0-17 | No (0 for males) |
        | Glucose | Blood glucose level | 60-200 mg/dL | Yes |
        | BloodPressure | Diastolic pressure | 40-122 mm Hg | Yes |
        | SkinThickness | Triceps skin fold | 10-99 mm | Yes |
        | Insulin | 2-hour serum insulin | 15-846 μU/ml | Yes |
        | BMI | Body mass index | 15-67 kg/m² | Yes |
        | DiabetesPedigreeFunction | Genetic risk score | 0.08-2.42 | Yes |
        | Age | Age in years | 21-81 | Yes |
        | Outcome | Has diabetes? | 0 or 1 | No (for prediction) |
        
        ### 👨👩 Gender-Inclusive:
        - **For males:** Set Pregnancies = 0
        - **For females:** Use actual pregnancy count
        - The system works perfectly for both!
        
        ### ⚡ Don't have data?
        No problem! Use the "Use Sample Data" option to explore with example data.
        """)
        
        st.markdown("""
        <div style='background: #48dbfb; padding: 15px; border-radius: 10px; color: white;'>
            <h4>✨ Tip: Missing Values are OK!</h4>
            <p>If your data has missing values, the system will handle it automatically in Step 2.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("🔧 Step 2: Prepare Your Data")
        
        st.markdown("""
        ### Understanding Missing Data:
        
        Missing values are common in medical data. This system offers multiple ways to handle them:
        
        **📊 Imputation Methods:**
        
        1. **Mean Imputation** 
           - Replaces missing values with the column average
           - ✅ Fast and simple
           - ⚠️ May not preserve relationships
        
        2. **Median Imputation**
           - Replaces with the middle value
           - ✅ Robust to outliers
           - ✅ Good for skewed data
        
        3. **KNN Imputation** (Recommended)
           - Uses similar patients to predict missing values
           - ✅ Preserves data patterns
           - ✅ Most accurate
           - ⚠️ Slower for large datasets
        
        ### How to Choose:
        
        The system will automatically test all methods and show you which works best for your data!
        
        ### Data Visualization:
        
        You'll see:
        - 📊 Distribution charts (how values are spread)
        - 🔥 Correlation heatmap (which features are related)
        - 📈 Diabetes vs Non-diabetes comparisons
        - ✨ Missing data patterns
        """)
    
    with tab3:
        st.subheader("🤖 Step 3: Train Your Model")
        
        st.markdown("""
        ### What is Model Training?
        
        Training means teaching the AI to recognize patterns in your data that indicate diabetes risk.
        
        ### Available Models:
        
        1. **🌲 Random Forest**
           - Uses multiple decision trees
           - ✅ Very accurate
           - ✅ Handles complex patterns
           - ✅ Shows feature importance
        
        2. **⚡ XGBoost**
           - Advanced gradient boosting
           - ✅ State-of-the-art performance
           - ✅ Excellent for medical data
           - ✅ Fast predictions
        
        3. **📊 Logistic Regression**
           - Classic statistical model
           - ✅ Easy to interpret
           - ✅ Fast training
           - ✅ Good baseline
        
        ### What You'll See:
        
        - 📊 **Accuracy:** How often the model is correct
        - 🎯 **Precision:** Of predicted diabetes cases, how many are true
        - 🔍 **Recall:** Of actual diabetes cases, how many were found
        - 📈 **ROC Curve:** Overall model performance visualization
        - 🗺️ **Confusion Matrix:** Detailed breakdown of predictions
        - ⭐ **Feature Importance:** Which factors matter most
        
        ### 💾 Save Your Model:
        
        Once trained, you can save the model to reuse it later without retraining!
        """)
    
    with tab4:
        st.subheader("🔮 Step 4: Make Predictions")
        
        st.markdown("""
        ### Two Ways to Predict:
        
        **Option A: Load Saved Model**
        1. Select a previously trained model
        2. Upload new patient data (without Outcome column)
        3. Get instant predictions!
        
        **Option B: Use Current Model**
        1. After training in Step 3
        2. Upload new patient data
        3. See predictions immediately
        
        ### What You Get:
        
        For each patient, you'll see:
        - 🎯 **Prediction:** Diabetes or No Diabetes
        - 📊 **Risk Score:** Probability (0-100%)
        - 🎨 **Color Coding:**
          - 🔴 Red = High Risk (>70%)
          - 🟡 Yellow = Medium Risk (30-70%)
          - 🟢 Blue/Green = Low Risk (<30%)
        
        ### 📥 Download Results:
        
        Export predictions as CSV for:
        - Patient records
        - Further analysis
        - Clinical review
        - Research purposes
        """)
    
    with tab5:
        st.subheader("👤 Step 5: Understand Individual Results")
        
        st.markdown("""
        ### Individual Patient Analysis
        
        This is where the magic happens! For EACH patient, you get:
        
        ### 🔬 Feature Contribution Chart
        
        Shows which medical factors are increasing or decreasing diabetes risk:
        
        - **Red Bars (→):** Factors INCREASING risk
        - **Blue Bars (←):** Factors DECREASING risk
        - **Length:** How much each factor contributes
        
        ### 🏥 Medical Explanations
        
        For each important factor, you get a plain-language explanation:
        
        **Example:**
        > *"🔴 Glucose level (156.2) is higher than average (120.5). High blood sugar is the PRIMARY indicator of diabetes - the body cannot properly process sugar."*
        
        ### 📋 What Each Feature Means:
        
        - **Glucose:** Blood sugar level - most important!
        - **BMI:** Body fat - affects insulin resistance
        - **Age:** Older age increases risk
        - **Genetic Risk:** Family history matters
        - **Pregnancies:** For females only
        - **Insulin:** Hormone regulating blood sugar
        - **Blood Pressure:** Often linked with diabetes
        - **Skin Thickness:** Indirect measure of body fat
        
        ### 🎯 Making Decisions:
        
        Use this information to:
        1. Understand why a patient is at risk
        2. Identify which factors to focus on
        3. Plan interventions (diet, exercise, medication)
        4. Educate patients about their specific risks
        5. Monitor high-risk factors over time
        """)
    
    st.markdown("---")
    st.success("✅ **Ready to start?** Use the navigation menu to go to 'Data Upload & Analysis' and begin!")

elif sidebar_selection == "📊 Data Upload & Analysis":
    st.header("📊 Data Upload & Analysis")
    
    data_option = st.radio("📁 Choose data source:", ["Upload CSV File", "Use Sample Data"])
    
    if data_option == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload your diabetes dataset (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_original = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df_original.shape}")
                st.session_state['df_original'] = df_original
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                st.stop()
        elif 'df_original' not in st.session_state:
            st.info("👆 Please upload a CSV file to continue")
            st.stop()
        else:
            df_original = st.session_state['df_original']
    else:
        df_original = create_sample_data()
        st.session_state['df_original'] = df_original
        st.success(f"✅ Sample data generated! Shape: {df_original.shape}")
    
    df_original = st.session_state['df_original']
    
    has_outcome = 'Outcome' in df_original.columns
    
    if has_outcome:
        st.info("📊 **Training Mode:** Outcome column detected - you can train models with this data")
    else:
        st.warning("🔮 **Prediction Mode:** No Outcome column - this data is ready for predictions with a trained model")
    
    st.subheader("📋 Data Preview")
    st.dataframe(df_original.head(10), use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Rows", df_original.shape[0])
    with col2:
        st.metric("📈 Total Columns", df_original.shape[1])
    with col3:
        st.metric("❓ Missing Values", df_original.isnull().sum().sum())
    with col4:
        if has_outcome:
            diabetes_pct = (df_original['Outcome'].sum() / len(df_original) * 100)
            st.metric("🔴 Diabetes Rate", f"{diabetes_pct:.1f}%")
    
    st.subheader("📊 Statistical Summary")
    st.dataframe(df_original.describe(), use_container_width=True)
    
    missing_count = df_original.isnull().sum().sum()
    
    if missing_count > 0:
        st.subheader("🔍 Missing Data Visualization")
        
        missing_data = df_original.isnull().sum()
        missing_percent = (missing_data / len(df_original)) * 100
        
        missing_df = pd.DataFrame({
            'Feature': missing_data.index,
            'Missing Count': missing_data.values,
            'Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Feature', y='Missing Count', 
                        title='Missing Values by Feature',
                        color='Percentage',
                        color_continuous_scale='Reds',
                        text='Percentage')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔧 Handle Missing Values")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            imputation_method = st.selectbox(
                "Select imputation method:",
                ["mean", "median", "knn"],
                help="Choose how to fill missing values"
            )
        
        method_descriptions = {
            "mean": "📊 Replaces missing values with the average of the column",
            "median": "📈 Replaces missing values with the middle value (good for outliers)",
            "knn": "🎯 Uses similar patients to predict missing values (most accurate)"
        }
        
        st.info(method_descriptions[imputation_method])
        
        if st.button("✨ Apply Imputation", type="primary"):
            with st.spinner("🔄 Imputing missing values..."):
                target_col = 'Outcome' if has_outcome else None
                df_imputed = impute_missing_values(df_original, method=imputation_method, target_col=target_col)
                st.session_state['df_imputed'] = df_imputed
                st.session_state['imputation_method'] = imputation_method
                st.markdown("<div class='success-box'>✅ Missing values imputed successfully!</div>", unsafe_allow_html=True)
                st.balloons()
        
        if 'df_imputed' in st.session_state:
            df_imputed = st.session_state['df_imputed']
            st.subheader("✨ Cleaned Data Preview")
            st.dataframe(df_imputed.head(10), use_container_width=True)
            
            remaining_missing = df_imputed.isnull().sum().sum()
            st.metric("Remaining Missing Values", remaining_missing)
            
            csv = df_imputed.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned Dataset",
                data=csv,
                file_name="cleaned_diabetes_data.csv",
                mime="text/csv"
            )
    else:
        st.success("✅ No missing values found in the dataset!")
        st.session_state['df_imputed'] = df_original
    
    if 'df_imputed' in st.session_state and has_outcome:
        df_viz = st.session_state['df_imputed']
        
        st.markdown("---")
        st.subheader("📊 Data Visualizations")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Distributions", "🔥 Correlations", "⚖️ Comparisons"])
        
        with viz_tab1:
            st.markdown("### Feature Distributions")
            
            numeric_cols = [col for col in df_viz.columns if col != 'Outcome']
            selected_feature = st.selectbox("Select feature to visualize:", numeric_cols)
            
            fig = px.histogram(df_viz, x=selected_feature, color='Outcome',
                             title=f'Distribution of {selected_feature} by Diabetes Status',
                             color_discrete_map={0: '#48dbfb', 1: '#ff6b6b'},
                             labels={'Outcome': 'Has Diabetes'},
                             marginal='box')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.markdown("### Correlation Heatmap")
            
            corr_matrix = df_viz.corr()
            
            fig = px.imshow(corr_matrix, 
                           text_auto='.2f',
                           color_continuous_scale='RdBu_r',
                           title='Feature Correlation Matrix',
                           aspect='auto')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **🔍 How to read this:**
            - **Red colors:** Positive correlation (values increase together)
            - **Blue colors:** Negative correlation (one increases, other decreases)
            - **Numbers closer to 1 or -1:** Stronger relationship
            """)
        
        with viz_tab3:
            st.markdown("### Diabetes vs Non-Diabetes Comparison")
            
            feature_compare = st.selectbox("Select feature to compare:", numeric_cols, key='compare')
            
            fig = go.Figure()
            
            for outcome in [0, 1]:
                data = df_viz[df_viz['Outcome'] == outcome][feature_compare].dropna()
                label = 'Has Diabetes' if outcome == 1 else 'No Diabetes'
                color = '#ff6b6b' if outcome == 1 else '#48dbfb'
                
                fig.add_trace(go.Box(
                    y=data,
                    name=label,
                    marker_color=color,
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title=f'{feature_compare} Comparison: Diabetes vs Non-Diabetes',
                yaxis_title=feature_compare,
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif sidebar_selection == "🤖 Train Models":
    st.header("🤖 Train Machine Learning Models")
    
    if 'df_imputed' not in st.session_state:
        st.warning("⚠️ Please upload and prepare your data first in 'Data Upload & Analysis'")
        st.stop()
    
    df = st.session_state['df_imputed']
    
    if 'Outcome' not in df.columns:
        st.error("❌ Cannot train models without 'Outcome' column. Please upload training data with outcomes.")
        st.stop()
    
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test data percentage:", 10, 40, 20) / 100
        
    with col2:
        selected_models = st.multiselect(
            "Select models to train:",
            ["Random Forest", "XGBoost", "Logistic Regression"],
            default=["Random Forest", "XGBoost"]
        )
    
    if not selected_models:
        st.warning("Please select at least one model to train")
        st.stop()
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("🔄 Training models... This may take a minute..."):
            
            X = df.drop(columns=['Outcome'])
            y = df['Outcome']
            
            feature_names = X.columns.tolist()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}...")
                
                if model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                elif model_name == "XGBoost":
                    model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
                else:
                    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                results[model_name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                progress_bar.progress((idx + 1) / len(selected_models))
            
            status_text.text("✅ Training complete!")
            
            st.session_state['training_results'] = results
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = feature_names
            st.session_state['X_test_scaled'] = X_test_scaled
            st.session_state['y_test'] = y_test
            
            st.markdown("<div class='success-box'>🎉 All models trained successfully!</div>", unsafe_allow_html=True)
            st.balloons()
    
    if 'training_results' in st.session_state:
        results = st.session_state['training_results']
        
        st.markdown("---")
        st.subheader("📊 Model Performance Comparison")
        
        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        st.success(f"🏆 Best Model: **{best_model_name}** with {results[best_model_name]['accuracy']:.4f} accuracy")
        
        st.subheader("📈 Detailed Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 ROC Curves", "📊 Confusion Matrices", "⭐ Feature Importance", "📉 Metrics Comparison"])
        
        with tab1:
            fig = go.Figure()
            
            for name, result in results.items():
                fpr, tpr, _ = roc_curve(st.session_state['y_test'], result['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'{name} (AUC = {roc_auc:.3f})',
                    mode='lines',
                    line=dict(width=3)
                ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(dash='dash', color='gray', width=2)
            ))
            
            fig.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **📖 Understanding ROC Curves:**
            - **Higher curve = Better model**
            - **AUC (Area Under Curve):** 1.0 = perfect, 0.5 = random guessing
            - Shows tradeoff between finding diabetes cases vs false alarms
            """)
        
        with tab2:
            selected_model_cm = st.selectbox("Select model:", list(results.keys()))
            
            cm = results[selected_model_cm]['confusion_matrix']
            
            fig = px.imshow(cm,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['No Diabetes', 'Diabetes'],
                           y=['No Diabetes', 'Diabetes'],
                           color_continuous_scale='Blues',
                           text_auto=True,
                           title=f'Confusion Matrix - {selected_model_cm}')
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            tn, fp, fn, tp = cm.ravel()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("✅ True Negatives", tn)
            col2.metric("❌ False Positives", fp)
            col3.metric("⚠️ False Negatives", fn)
            col4.metric("✅ True Positives", tp)
            
            st.info("""
            **📖 Understanding Confusion Matrix:**
            - **True Positives (TP):** Correctly identified diabetes cases
            - **True Negatives (TN):** Correctly identified non-diabetes
            - **False Positives (FP):** Predicted diabetes but actually healthy
            - **False Negatives (FN):** Missed diabetes cases (most critical!)
            """)
        
        with tab3:
            selected_model_fi = st.selectbox("Select model:", list(results.keys()), key='fi')
            
            model = results[selected_model_fi]['model']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                st.warning("Feature importance not available for this model")
                importances = None
            
            if importances is not None:
                feature_importance_df = pd.DataFrame({
                    'Feature': st.session_state['feature_names'],
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(feature_importance_df, x='Importance', y='Feature',
                           title=f'Feature Importance - {selected_model_fi}',
                           orientation='h',
                           color='Importance',
                           color_continuous_scale='Viridis')
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🏥 Medical Interpretation")
                
                top_features = feature_importance_df.tail(3)
                
                for idx, row in top_features.iterrows():
                    feature = row['Feature']
                    importance = row['Importance']
                    
                    explanations = {
                        'Glucose': "🔴 **Glucose** is the PRIMARY diabetes indicator. High blood sugar shows the body cannot process sugar properly.",
                        'BMI': "🟠 **BMI (Body Mass Index)** measures body fat. Excess weight causes insulin resistance.",
                        'Age': "🟡 **Age** matters because insulin production declines and resistance increases with aging.",
                        'DiabetesPedigreeFunction': "🟢 **Genetic Risk** represents family history - a strong predictor of diabetes.",
                        'Pregnancies': "🔵 **Pregnancies** can stress insulin-producing cells and increase risk.",
                        'Insulin': "🟣 **Insulin levels** show how well the body regulates blood sugar.",
                        'BloodPressure': "🟤 **Blood Pressure** often coexists with diabetes (metabolic syndrome).",
                        'SkinThickness': "⚪ **Skin Thickness** can indicate body fat and insulin resistance."
                    }
                    
                    explanation = explanations.get(feature, f"{feature} contributes to diabetes prediction.")
                    st.markdown(f"**{importance:.3f}** - {explanation}")
        
        with tab4:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            
            for metric in metrics_to_plot:
                metric_data = pd.DataFrame({
                    'Model': list(results.keys()),
                    metric.capitalize(): [results[m][metric] for m in results.keys()]
                })
                
                fig = px.bar(metric_data, x='Model', y=metric.capitalize(),
                           title=f'{metric.capitalize()} Comparison',
                           color=metric.capitalize(),
                           color_continuous_scale='Teal',
                           text=metric.capitalize())
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💾 Save Best Model")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            model_to_save = st.selectbox("Select model to save:", list(results.keys()))
        
        with col2:
            st.write("")
            st.write("")
            if st.button("💾 Save Model"):
                model = results[model_to_save]['model']
                scaler = st.session_state['scaler']
                feature_names = st.session_state['feature_names']
                metrics = {
                    'accuracy': results[model_to_save]['accuracy'],
                    'precision': results[model_to_save]['precision'],
                    'recall': results[model_to_save]['recall'],
                    'f1': results[model_to_save]['f1']
                }
                
                filename = save_model(model_to_save, model, scaler, feature_names, metrics)
                st.success(f"✅ Model saved as: {filename}")
                st.balloons()

elif sidebar_selection == "🔮 Make Predictions":
    st.header("🔮 Make Predictions on New Data")
    
    st.subheader("📁 Load Prediction Data")
    
    pred_file = st.file_uploader("Upload CSV file with patient data (without Outcome column)", type=['csv'], key='pred')
    
    if pred_file is not None:
        try:
            df_predict = pd.read_csv(pred_file)
            st.success(f"✅ File uploaded! Shape: {df_predict.shape}")
            st.dataframe(df_predict.head(), use_container_width=True)
            
            if df_predict.isnull().sum().sum() > 0:
                st.warning("⚠️ Missing values detected. Applying automatic imputation...")
                df_predict = impute_missing_values(df_predict, method='knn', target_col=None)
                st.success("✅ Missing values handled!")
            
            st.session_state['df_predict'] = df_predict
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()
    
    if 'df_predict' not in st.session_state:
        st.info("👆 Please upload a CSV file with patient data to make predictions")
        st.stop()
    
    df_predict = st.session_state['df_predict']
    
    st.markdown("---")
    st.subheader("🤖 Select Model")
    
    model_option = st.radio("Choose model source:", ["Use Saved Model", "Use Recently Trained Model"])
    
    if model_option == "Use Saved Model":
        saved_models = get_saved_models()
        
        if not saved_models:
            st.warning("⚠️ No saved models found. Please train and save a model first.")
            st.stop()
        
        model_choices = [f"{m['name']} - {m['timestamp']} (Acc: {m['accuracy']:.4f})" for m in saved_models]
        selected_model_idx = st.selectbox("Select saved model:", range(len(model_choices)), format_func=lambda x: model_choices[x])
        
        if st.button("📥 Load Model"):
            model_pkg = load_model(saved_models[selected_model_idx]['filename'])
            st.session_state['loaded_model'] = model_pkg
            st.success(f"✅ Loaded: {model_pkg['model_name']}")
    
    elif model_option == "Use Recently Trained Model":
        if 'training_results' not in st.session_state:
            st.warning("⚠️ No recently trained models. Please train a model first in 'Train Models'.")
            st.stop()
        
        results = st.session_state['training_results']
        selected_model_name = st.selectbox("Select model:", list(results.keys()))
        
        model_pkg = {
            'model': results[selected_model_name]['model'],
            'scaler': st.session_state['scaler'],
            'feature_names': st.session_state['feature_names'],
            'model_name': selected_model_name
        }
        st.session_state['loaded_model'] = model_pkg
        st.success(f"✅ Using: {selected_model_name}")
    
    if 'loaded_model' in st.session_state:
        st.markdown("---")
        
        if st.button("🚀 Generate Predictions", type="primary"):
            with st.spinner("🔄 Generating predictions..."):
                model_pkg = st.session_state['loaded_model']
                model = model_pkg['model']
                scaler = model_pkg['scaler']
                feature_names = model_pkg['feature_names']
                
                X_pred = df_predict[feature_names]
                X_pred_scaled = scaler.transform(X_pred)
                
                predictions = model.predict(X_pred_scaled)
                probabilities = model.predict_proba(X_pred_scaled)[:, 1]
                
                df_results = df_predict.copy()
                df_results['Prediction'] = predictions
                df_results['Diabetes_Probability'] = probabilities
                df_results['Risk_Level'] = pd.cut(probabilities, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
                
                st.session_state['prediction_results'] = df_results
                st.session_state['predictions'] = predictions
                st.session_state['probabilities'] = probabilities
                st.session_state['X_pred_scaled'] = X_pred_scaled
                
                st.markdown("<div class='success-box'>🎉 Predictions generated successfully!</div>", unsafe_allow_html=True)
                st.balloons()
        
        if 'prediction_results' in st.session_state:
            df_results = st.session_state['prediction_results']
            
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total = len(df_results)
                st.metric("👥 Total Patients", total)
            
            with col2:
                diabetes_count = (df_results['Prediction'] == 1).sum()
                st.metric("🔴 Diabetes Predicted", diabetes_count)
            
            with col3:
                high_risk = (df_results['Risk_Level'] == 'High').sum()
                st.metric("⚠️ High Risk", high_risk)
            
            with col4:
                avg_prob = df_results['Diabetes_Probability'].mean()
                st.metric("📊 Avg Risk Score", f"{avg_prob:.2%}")
            
            st.dataframe(df_results, use_container_width=True)
            
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="diabetes_predictions.csv",
                mime="text/csv"
            )
            
            st.subheader("📈 Risk Distribution")
            
            fig = px.histogram(df_results, x='Diabetes_Probability',
                             title='Distribution of Diabetes Risk Scores',
                             nbins=30,
                             color_discrete_sequence=['#667eea'])
            fig.update_layout(height=400)
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Decision Threshold")
            
            st.plotly_chart(fig, use_container_width=True)

elif sidebar_selection == "👤 Individual Analysis":
    st.header("👤 Individual Patient Analysis")
    st.markdown("### 🔬 Detailed Explanation for Each Patient")
    
    if 'prediction_results' not in st.session_state:
        st.warning("⚠️ Please make predictions first in 'Make Predictions' section")
        st.stop()
    
    df_results = st.session_state['prediction_results']
    model_pkg = st.session_state['loaded_model']
    
    patient_idx = st.selectbox(
        "Select patient to analyze:",
        range(len(df_results)),
        format_func=lambda x: f"Patient #{x+1} - Risk: {df_results.iloc[x]['Diabetes_Probability']:.1%}"
    )
    
    patient_data = df_results.iloc[patient_idx]
    prediction = st.session_state['predictions'][patient_idx]
    probability = st.session_state['probabilities'][patient_idx]
    
    risk_class = 'risk-high' if probability > 0.7 else 'risk-medium' if probability > 0.3 else 'risk-low'
    risk_emoji = '🔴' if probability > 0.7 else '🟡' if probability > 0.3 else '🟢'
    risk_text = 'HIGH RISK' if probability > 0.7 else 'MEDIUM RISK' if probability > 0.3 else 'LOW RISK'
    
    st.markdown(f"""
    <div class='{risk_class}'>
        <h2>{risk_emoji} Patient #{patient_idx+1} - {risk_text}</h2>
        <h3>Diabetes Probability: {probability:.1%}</h3>
        <p>Prediction: {'HAS DIABETES' if prediction == 1 else 'NO DIABETES'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📋 Patient Information")
    
    feature_names = model_pkg['feature_names']
    patient_features = patient_data[feature_names].values.reshape(1, -1)
    
    col1, col2 = st.columns(2)
    
    for idx, feature in enumerate(feature_names):
        value = patient_data[feature]
        if idx % 2 == 0:
            col1.metric(feature, f"{value:.2f}")
        else:
            col2.metric(feature, f"{value:.2f}")
    
    st.markdown("---")
    st.subheader("🔬 Feature Contribution Analysis")
    
    X_pred_scaled = st.session_state['X_pred_scaled']
    
    contributions_df = explain_individual_prediction(
        model_pkg['model'],
        X_pred_scaled,
        feature_names,
        X_pred_scaled[patient_idx].reshape(1, -1),
        prediction,
        probability
    )
    
    fig = plot_individual_explanation(
        contributions_df,
        f"Patient #{patient_idx+1}",
        prediction,
        probability
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🏥 Medical Interpretation")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;'>
        <h4>📖 Understanding the Analysis:</h4>
        <p><strong>Red bars (→)</strong> show features INCREASING diabetes risk</p>
        <p><strong>Blue bars (←)</strong> show features DECREASING diabetes risk</p>
        <p>Longer bars = stronger contribution to the prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    top_features = contributions_df.head(5)
    
    for idx, row in top_features.iterrows():
        feature = row['Feature']
        value = row['Patient Value']
        avg_value = row['Average Value']
        contribution = row['Contribution']
        
        explanation = get_medical_explanation(feature, value, avg_value, contribution)
        
        with st.expander(f"🔍 {feature} - Impact: {abs(contribution):.3f}"):
            st.markdown(f"**Patient Value:** {value:.2f}")
            st.markdown(f"**Average Value:** {avg_value:.2f}")
            st.markdown(f"**Difference:** {value - avg_value:.2f}")
            st.markdown("---")
            st.markdown(explanation)
    
    st.markdown("---")
    st.subheader("📊 Comparison with Average Patient")
    
    comparison_data = []
    for feature in feature_names[:5]:
        comparison_data.append({
            'Feature': feature,
            'This Patient': patient_data[feature],
            'Average': contributions_df[contributions_df['Feature'] == feature]['Average Value'].values[0]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comparison_df['Feature'],
        y=comparison_df['This Patient'],
        name='This Patient',
        marker_color='#ff6b6b'
    ))
    
    fig.add_trace(go.Bar(
        x=comparison_df['Feature'],
        y=comparison_df['Average'],
        name='Average Patient',
        marker_color='#48dbfb'
    ))
    
    fig.update_layout(
        title='Patient vs Average Comparison (Top 5 Features)',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("💡 Clinical Recommendations")
    
    if probability > 0.7:
        st.error("""
        **⚠️ HIGH RISK - Immediate Action Recommended:**
        - Schedule comprehensive diabetes screening
        - Consider HbA1c and fasting glucose tests
        - Lifestyle intervention counseling
        - Regular monitoring essential
        - Evaluate family history and genetic factors
        """)
    elif probability > 0.3:
        st.warning("""
        **🟡 MEDIUM RISK - Preventive Measures Advised:**
        - Monitor key indicators regularly
        - Lifestyle modifications recommended
        - Diet and exercise counseling
        - Follow-up screening in 6 months
        """)
    else:
        st.success("""
        **✅ LOW RISK - Maintain Healthy Lifestyle:**
        - Continue current healthy habits
        - Routine screening as per guidelines
        - Stay physically active
        - Maintain healthy weight
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏥 <strong>Advanced Diabetes Prediction System</strong></p>
    <p>Powered by AI | Medical-Grade Analysis | Individual Patient Insights</p>
    <p style='font-size: 12px;'>⚕️ For medical professional use. Always consult with healthcare providers for diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)
