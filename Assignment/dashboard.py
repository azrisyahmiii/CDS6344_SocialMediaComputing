# Updated Social Media Computing Dashboard
# Integrated with Fixed ABSA Implementation
# How to run: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import pickle
import joblib
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import your fixed ABSA utilities
try:
    import absa_utils_improved as absa
    absa_available = True
    print("✓ ABSA utilities loaded successfully")
except ImportError:
    absa_available = False
    print("⚠ ABSA utilities not found")

# Configure Streamlit page
st.set_page_config(
    page_title="✈️ Airline Sentiment Analysis Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #7ecf87; /* darker green */
        border: 1px solid #5fa96a;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #5bb6ce; /* darker blue */
        border: 1px solid #399ab3;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data
def load_main_data():
    """Load main dataset"""
    try:
        df = pd.read_csv('data/cleaned_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Could not find data/cleaned_data.csv")
        return None

@st.cache_data
def load_absa_data():
    """Load ABSA results"""
    try:
        absa_df = pd.read_csv('data/absa_results_fixed.csv')
        with open('data/absa_summary_fixed.pkl', 'rb') as f:
            absa_summary = pickle.load(f)
        return absa_df, absa_summary
    except FileNotFoundError:
        st.warning("⚠ ABSA results not found. Please run the ABSA analysis first.")
        return None, None

@st.cache_data
def load_model_results():
    """Load model performance results"""
    try:
        with open('models/traditional_ml/results_summary.pkl', 'rb') as f:
            ml_results = pickle.load(f)
        with open('models/deep_learning/deep_learning_results.pkl', 'rb') as f:
            dl_results = pickle.load(f)
        return ml_results, dl_results
    except FileNotFoundError:
        st.warning("⚠ Model results not found")
        return None, None

# Load all data
df = load_main_data()
absa_df, absa_summary = load_absa_data()
ml_results, dl_results = load_model_results()

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Analysis Type",
    ["🏠 Overview", "📊 Sentiment Analysis", "🔍 Model Comparison", "🎯 ABSA Explorer", "💬 Interactive Analysis"]
)

# Main dashboard title
st.title("✈️ Airline Sentiment Analysis Dashboard")
st.markdown("*Comprehensive NLP analysis with traditional ML, deep learning, and aspect-based sentiment analysis*")

# Overview Page
if page == "🏠 Overview":
    st.header("📋 Project Overview")
    
    # Key metrics in columns
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Tweets", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Airlines", df['airline'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            if absa_summary:
                st.metric("Aspects Found", absa_summary['total_aspects'])
            else:
                st.metric("Models Trained", "6")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            if ml_results:
                st.metric("Best F1-Score", f"{ml_results['best_f1_score']:.3f}")
            else:
                st.metric("Avg Accuracy", "78%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Distribution")
        if df is not None:
            sentiment_counts = df['airline_sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Breakdown",
                color_discrete_map={'negative': '#ff4444', 'neutral': '#888888', 'positive': '#44ff44'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Airlines Distribution")
        if df is not None:
            airline_counts = df['airline'].value_counts()
            fig = px.bar(
                x=airline_counts.values,
                y=airline_counts.index,
                orientation='h',
                title="Tweets by Airline",
                color=airline_counts.values,
                color_continuous_scale='blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Project highlights
    st.subheader("🎯 Project Highlights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>🤖 Machine Learning</h4>
        <ul>
        <li>Logistic Regression</li>
        <li>Support Vector Machine</li>
        <li>Random Forest</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🧠 Deep Learning</h4>
        <ul>
        <li>LSTM Networks</li>
        <li>Bidirectional LSTM</li>
        <li>Pre-trained BERT/RoBERTa</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-box">
        <h4>🎯 ABSA Analysis</h4>
        <ul>
        <li>Aspect Extraction</li>
        <li>Sentiment per Aspect</li>
        <li>Business Insights</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Sentiment Analysis Page
elif page == "📊 Sentiment Analysis":
    st.header("📊 Detailed Sentiment Analysis")
    
    if df is not None:
        # Airline selector
        selected_airline = st.selectbox("🏢 Select Airline", ["All Airlines"] + list(df['airline'].unique()))
        
        # Filter data based on selection
        if selected_airline == "All Airlines":
            filtered_df = df
        else:
            filtered_df = df[df['airline'] == selected_airline]
        
        # Tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📈 Distribution Analysis", "☁️ Word Clouds", "📋 Sample Tweets"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Counts")
                sentiment_counts = filtered_df['airline_sentiment'].value_counts()
                fig = px.bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    color=sentiment_counts.index,
                    color_discrete_map={'negative': '#ff4444', 'neutral': '#888888', 'positive': '#44ff44'},
                    title=f"Sentiment Distribution - {selected_airline}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Text Length Analysis")
                filtered_df['text_length'] = filtered_df['cleaned_text'].str.split().str.len()
                avg_length = filtered_df.groupby('airline_sentiment')['text_length'].mean().reset_index()
                
                fig = px.bar(
                    avg_length,
                    x='airline_sentiment',
                    y='text_length',
                    color='airline_sentiment',
                    color_discrete_map={'negative': '#ff4444', 'neutral': '#888888', 'positive': '#44ff44'},
                    title="Average Text Length by Sentiment"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("☁️ Word Clouds by Sentiment")
            
            sentiment_choice = st.selectbox("Select Sentiment", ['positive', 'neutral', 'negative'])
            
            sentiment_texts = filtered_df[filtered_df['airline_sentiment'] == sentiment_choice]['cleaned_text']
            
            if len(sentiment_texts) > 0:
                text_combined = ' '.join(sentiment_texts.astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=100,
                    colormap='viridis'
                ).generate(text_combined)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Word Cloud - {sentiment_choice.title()} Sentiment', fontsize=16, fontweight='bold')
                st.pyplot(fig)
            else:
                st.warning(f"No {sentiment_choice} tweets found for {selected_airline}")
        
        with tab3:
            st.subheader("📋 Sample Tweets")
            
            for sentiment in ['positive', 'negative', 'neutral']:
                st.markdown(f"**{sentiment.title()} Examples:**")
                examples = filtered_df[filtered_df['airline_sentiment'] == sentiment]['original_text'].head(3)
                for i, tweet in enumerate(examples, 1):
                    st.write(f"{i}. {tweet}")
                st.markdown("---")

# Model Comparison Page
elif page == "🔍 Model Comparison":
    st.header("🔍 Model Performance Comparison")
    
    if ml_results and dl_results:
        # Combine all results
        all_models = []
        
        # Add traditional ML results
        for model in ml_results['all_results']:
            all_models.append({
                'Model': model['Model'],
                'Type': 'Traditional ML',
                'Accuracy': model['Accuracy'],
                'F1-Score': model['F1-Score'],
                'Precision': model['Precision'],
                'Recall': model['Recall']
            })
        
        # Add deep learning results
        for model in dl_results['complete_comparison']:
            if model['Model'] in ['LSTM', 'Bidirectional LSTM', 'Twitter-RoBERTa']:
                model_type = 'Deep Learning' if 'LSTM' in model['Model'] else 'Transformer'
                all_models.append({
                    'Model': model['Model'],
                    'Type': model_type,
                    'Accuracy': model['Accuracy'],
                    'F1-Score': model['F1-Score'],
                    'Precision': model['Precision'],
                    'Recall': model['Recall']
                })
        
        comparison_df = pd.DataFrame(all_models)
        
        # Performance comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Accuracy',
                color='Type',
                title="Model Accuracy Comparison",
                color_discrete_map={'Traditional ML': '#3498db', 'Deep Learning': '#e74c3c', 'Transformer': '#f39c12'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                comparison_df,
                x='Model',
                y='F1-Score',
                color='Type',
                title="Model F1-Score Comparison",
                color_discrete_map={'Traditional ML': '#3498db', 'Deep Learning': '#e74c3c', 'Transformer': '#f39c12'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
        
        st.subheader("🏆 Best Performing Model")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", best_model['Model'])
        with col2:
            st.metric("Accuracy", f"{best_model['Accuracy']:.3f}")
        with col3:
            st.metric("F1-Score", f"{best_model['F1-Score']:.3f}")
        with col4:
            st.metric("Type", best_model['Type'])
        
        # Detailed comparison table
        st.subheader("📊 Detailed Performance Table")
        st.dataframe(comparison_df.round(4), use_container_width=True)
    
    else:
        st.warning("⚠ Model results not available. Please ensure model training is complete.")

# ABSA Explorer Page
elif page == "🎯 ABSA Explorer":
    st.header("🎯 Aspect-Based Sentiment Analysis")
    
    if absa_df is not None and absa_summary is not None:
        # ABSA Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Aspects", absa_summary['total_aspects'])
        with col2:
            st.metric("Unique Categories", absa_summary['unique_aspects'])
        with col3:
            avg_conf = absa_summary.get('average_confidence', 0)
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
        with col4:
            neg_pct = absa_summary['sentiment_distribution'].get('negative', 0) / absa_summary['total_aspects'] * 100
            st.metric("Negative %", f"{neg_pct:.1f}%")
        
        # ABSA Visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Aspect Overview", "🔥 Heatmap Analysis", "📋 Detailed Insights"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Aspect Frequency")
                aspect_counts = absa_df['aspect'].value_counts()
                fig = px.bar(
                    x=aspect_counts.values,
                    y=aspect_counts.index,
                    orientation='h',
                    title="Most Mentioned Aspects",
                    color=aspect_counts.values,
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Sentiment Distribution")
                sentiment_counts = absa_df['sentiment'].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="ABSA Sentiment Breakdown",
                    color_discrete_map={'negative': '#ff4444', 'neutral': '#888888', 'positive': '#44ff44'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🔥 Aspect vs Sentiment Heatmap")
            
            # Create heatmap data
            heatmap_data = pd.crosstab(absa_df['aspect'], absa_df['sentiment'])
            
            fig = px.imshow(
                heatmap_data,
                title="Aspect-Sentiment Co-occurrence Matrix",
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            fig.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Aspect"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Business insights
            st.subheader("💡 Key Business Insights")
            
            # Find most problematic aspects
            aspect_sentiment = absa_df.groupby('aspect')['sentiment'].apply(lambda x: (x == 'negative').mean() * 100)
            worst_aspects = aspect_sentiment.nlargest(3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🚨 Most Problematic Aspects:**")
                for aspect, neg_pct in worst_aspects.items():
                    if aspect in absa_summary['aspect_descriptions']:
                        desc = absa_summary['aspect_descriptions'][aspect]
                        st.write(f"• **{aspect.title()}**: {neg_pct:.1f}% negative")
                        st.write(f"  *{desc}*")
            
            with col2:
                st.markdown("**📈 Improvement Opportunities:**")
                # Find aspects with mixed sentiment (potential for improvement)
                mixed_aspects = absa_df.groupby('aspect')['sentiment'].apply(
                    lambda x: len(x.unique()) > 1
                ).sum()
                st.write(f"• {mixed_aspects} aspects show mixed sentiment")
                st.write("• Focus on converting neutral/negative to positive")
                st.write("• Service training can address negative feedback")
        
        with tab3:
            st.subheader("📋 Detailed Aspect Analysis")
            
            # Aspect selector
            selected_aspect = st.selectbox("Select Aspect for Analysis", absa_df['aspect'].unique())
            
            aspect_data = absa_df[absa_df['aspect'] == selected_aspect]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment breakdown for selected aspect
                sentiment_breakdown = aspect_data['sentiment'].value_counts()
                fig = px.bar(
                    x=sentiment_breakdown.index,
                    y=sentiment_breakdown.values,
                    color=sentiment_breakdown.index,
                    color_discrete_map={'negative': '#ff4444', 'neutral': '#888888', 'positive': '#44ff44'},
                    title=f"Sentiment for {selected_aspect.title()}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistics
                st.markdown(f"**Statistics for {selected_aspect.title()}:**")
                total_mentions = len(aspect_data)
                avg_confidence = aspect_data['confidence'].mean()
                
                st.write(f"• Total mentions: {total_mentions}")
                st.write(f"• Average confidence: {avg_confidence:.2f}")
                
                for sentiment in ['negative', 'neutral', 'positive']:
                    count = (aspect_data['sentiment'] == sentiment).sum()
                    pct = count / total_mentions * 100
                    st.write(f"• {sentiment.title()}: {count} ({pct:.1f}%)")
            
            # Sample tweets for selected aspect
            st.markdown(f"**Sample Tweets mentioning {selected_aspect.title()}:**")
            sample_tweets = aspect_data.sample(min(5, len(aspect_data)))
            
            for _, tweet in sample_tweets.iterrows():
                sentiment_color = {'negative': '🔴', 'neutral': '⚪', 'positive': '🟢'}[tweet['sentiment']]
                st.write(f"{sentiment_color} **{tweet['sentiment'].title()}** (conf: {tweet['confidence']:.2f})")
                st.write(f"*{tweet['original_text']}*")
                st.write("---")
    
    else:
        st.warning("⚠ ABSA results not available. Please run the ABSA analysis first.")
        st.info("💡 Run the fixed ABSA implementation to see aspect-based insights here.")

# Interactive Analysis Page
elif page == "💬 Interactive Analysis":
    st.header("💬 Interactive Tweet Analysis")
    
    st.markdown("Enter a tweet below to see real-time sentiment analysis and aspect detection!")
    
    # Text input
    user_tweet = st.text_area(
        "✍️ Enter your tweet:",
        placeholder="Example: The flight was delayed but the staff was very helpful!",
        height=100
    )
    
    # Analysis button
    if st.button("🔍 Analyze Tweet", type="primary"):
        if user_tweet.strip():
            if absa_available:
                # Perform ABSA analysis
                try:
                    results = absa.analyze_tweet_complete_fixed(user_tweet)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Overall Analysis")
                        
                        sentiment = results['overall_sentiment']
                        confidence = results['confidence']
                        
                        # Display overall sentiment with color
                        sentiment_colors = {
                            'positive': '🟢',
                            'negative': '🔴', 
                            'neutral': '⚪'
                        }
                        
                        st.markdown(f"**Overall Sentiment:** {sentiment_colors[sentiment]} **{sentiment.title()}**")
                        st.markdown(f"**Confidence:** {confidence:.2f}")
                        
                        # Progress bar for confidence
                        st.progress(confidence)
                    
                    with col2:
                        st.subheader("🎯 Aspects Detected")
                        
                        if results['aspects']:
                            for i, aspect in enumerate(results['aspects'], 1):
                                st.markdown(f"**{i}. {aspect['aspect'].title()}**")
                                st.markdown(f"   *{aspect['description']}*")
                                st.markdown(f"   Sentiment: {sentiment_colors[aspect['sentiment']]} {aspect['sentiment'].title()}")
                                st.markdown("---")
                        else:
                            st.info("No specific airline service aspects detected in this tweet.")
                    
                    # Business interpretation
                    if results['aspects']:
                        st.subheader("💼 Business Interpretation")
                        
                        negative_aspects = [a for a in results['aspects'] if a['sentiment'] == 'negative']
                        positive_aspects = [a for a in results['aspects'] if a['sentiment'] == 'positive']
                        
                        if negative_aspects:
                            st.markdown("**🚨 Areas for Improvement:**")
                            for aspect in negative_aspects:
                                st.write(f"• {aspect['aspect'].title()}: {aspect['description']}")
                        
                        if positive_aspects:
                            st.markdown("**✅ Positive Feedback:**")
                            for aspect in positive_aspects:
                                st.write(f"• {aspect['aspect'].title()}: {aspect['description']}")
                
                except Exception as e:
                    st.error(f"❌ Error analyzing tweet: {e}")
                    st.info("Please check that the ABSA utilities are properly configured.")
            
            else:
                st.warning("⚠ ABSA analysis not available. Showing basic sentiment only.")
                
                # Simple fallback analysis
                positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'delayed']
                
                tweet_lower = user_tweet.lower()
                pos_count = sum(1 for word in positive_words if word in tweet_lower)
                neg_count = sum(1 for word in negative_words if word in tweet_lower)
                
                if neg_count > pos_count:
                    sentiment = "🔴 Negative"
                elif pos_count > neg_count:
                    sentiment = "🟢 Positive"
                else:
                    sentiment = "⚪ Neutral"
                
                st.markdown(f"**Basic Sentiment:** {sentiment}")
        
        else:
            st.warning("⚠ Please enter a tweet to analyze!")
    
    # Sample tweets for testing
    st.subheader("💡 Try These Sample Tweets:")
    
    sample_tweets = [
        "The flight was delayed for 3 hours but the staff was very helpful",
        "Flight on time but food was awful", 
        "Great booking experience on the website and smooth check-in process",
        "Lost my baggage and nobody at customer service could help me",
        "Excellent crew service and comfortable seats, will fly again!"
    ]
    
    for i, sample in enumerate(sample_tweets, 1):
        if st.button(f"Try Sample {i}", key=f"sample_{i}"):
            st.session_state.sample_tweet = sample
            st.rerun()
    
    # Display selected sample tweet
    if 'sample_tweet' in st.session_state:
        st.text_area("Selected Sample:", value=st.session_state.sample_tweet, height=68, disabled=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 30px;'>
<h4>🎓 Social Media Computing Project</h4>
<p>Comprehensive sentiment analysis using traditional ML, deep learning, and ABSA</p>
<p><strong>Technologies:</strong> Python • Scikit-learn • TensorFlow • Transformers • Streamlit • SpaCy</p>
</div>
""", unsafe_allow_html=True)

# Debug information (only show if data is missing)
if df is None or absa_df is None:
    with st.expander("🔧 Debug Information"):
        st.markdown("**Expected File Structure:**")
        st.code("""
data/
├── cleaned_data.csv
├── absa_results_fixed.csv
├── absa_summary_fixed.pkl
└── ...

models/
├── traditional_ml/results_summary.pkl
├── deep_learning/deep_learning_results.pkl
└── ...

absa_utils_fixed.py
        """)
        
        st.markdown("**Troubleshooting:**")
        st.write("1. Ensure all data files are in the correct locations")
        st.write("2. Run the preprocessing and model training notebooks")
        st.write("3. Run the fixed ABSA implementation")
        st.write("4. Check that absa_utils_fixed.py is in the main directory")