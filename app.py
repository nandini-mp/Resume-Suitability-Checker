import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import PyPDF2
import io
import base64

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_resources()
stop_words = set(stopwords.words('english'))

# Page configuration
st.set_page_config(page_title="Resume Classifier", layout="wide")
st.title("📄 Smart Resume Classifier")
st.markdown("Upload your resume and find out if it's suitable for your target role!")

# Sidebar for job role selection
st.sidebar.header("🎯 Target Job Role")
job_roles = ["Data Scientist", "Web Developer", "Software Engineer", "AI/ML Engineer"]
selected_role = st.sidebar.selectbox("Choose a role:", job_roles)

# Load and cache training data
@st.cache_data
def load_training_data():
    try:
        df = pd.read_csv("training_resumes.csv")
        return df
    except:
        st.error("Training data file not found. Please ensure 'training_resumes.csv' is in the same directory.")
        return None

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', str(text).lower())  # Remove special characters and convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()       # Remove extra whitespace
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Extract text from resume file
def extract_resume_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        try:
            resume_bytes = io.BytesIO(uploaded_file.getvalue())
            text = extract_text_from_pdf(resume_bytes)
            if not text:
                st.warning("Could not extract text from this PDF. It might be scanned or image-based.")
                return None
            return text
        except:
            st.error("Failed to process PDF. Please try another file.")
            return None
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        st.error("Unsupported file format. Please upload a PDF or TXT file.")
        return None

# Define role-specific skills
role_skills = {
    "Data Scientist": ["python", "r", "sql", "machine learning", "deep learning", "statistics",
                      "data analysis", "data visualization", "pandas", "numpy", "scikit-learn",
                      "tensorflow", "pytorch", "big data", "predictive modeling"],

    "Web Developer": ["html", "css", "javascript", "react", "angular", "vue", "node.js",
                     "express", "api", "frontend", "backend", "responsive design", "web design",
                     "bootstrap", "jquery"],

    "Software Engineer": ["java", "c++", "python", "algorithms", "data structures", "oop",
                         "software development", "git", "databases", "agile", "testing",
                         "problem solving", "architecture", "ci/cd", "design patterns"],

    "AI/ML Engineer": ["machine learning", "deep learning", "neural networks", "tensorflow",
                      "pytorch", "keras", "computer vision", "nlp", "reinforcement learning",
                      "data modeling", "feature engineering", "model deployment", "python",
                      "algorithms", "optimization"]
}

# Main function
def main():
    df = load_training_data()
    if df is None:
        return

    # Filter data for selected role
    role_data = df[df['Role'] == selected_role]

    if role_data.empty:
        st.error(f"No training data available for {selected_role}. Please select another role.")
        return

    # Resume upload section
    st.header("📤 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume in PDF or TXT format", type=["pdf", "txt"])

    if uploaded_file:
        with st.spinner("Processing your resume..."):
            # Extract text from resume
            resume_text = extract_resume_text(uploaded_file)

            if resume_text:
                # Show preview of extracted text
                with st.expander("Resume Text Preview"):
                    st.text_area("Extracted Text", resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""), height=200)

                # Preprocess resume text
                processed_resume = preprocess_text(resume_text)

                # Prepare data for classification
                X = role_data['Text'].apply(preprocess_text)
                y = role_data['Outcome']

                # Create TF-IDF vectorizer
                tfidf = TfidfVectorizer(max_features=5000)
                X_tfidf = tfidf.fit_transform(X)

                # Train SVM classifier
                model = SVC(kernel='linear', probability=True, random_state=42)
                model.fit(X_tfidf, y)

                # Transform resume text and predict
                resume_tfidf = tfidf.transform([processed_resume])
                prediction = model.predict(resume_tfidf)[0]
                probability = model.predict_proba(resume_tfidf)[0][1] * 100

                # Display results
                st.header("🔍 Analysis Results")

                if prediction == 1:
                    st.success(f"✅ Your resume is suitable for the role of {selected_role}!")
                    st.metric("Suitability Score", f"{probability:.1f}%")

                    # Highlight matching skills
                    matching_skills = []
                    for skill in role_skills[selected_role]:
                        if skill in processed_resume or skill in resume_text.lower():
                            matching_skills.append(skill)

                    if matching_skills:
                        st.subheader("💪 Your Strengths")
                        st.write(f"Your resume highlights these relevant skills: **{', '.join(matching_skills)}**")
                else:
                    st.error(f"❌ Your resume needs improvement for the {selected_role} role.")
                    st.metric("Current Match", f"{probability:.1f}%")

                    # Identify missing skills
                    missing_skills = []
                    for skill in role_skills[selected_role]:
                        if skill not in processed_resume and skill not in resume_text.lower():
                            missing_skills.append(skill)

                    if missing_skills:
                        st.subheader("🚀 Improvement Suggestions")
                        st.write("Consider adding these skills to your resume:")

                        cols = st.columns(3)
                        for i, skill in enumerate(missing_skills[:9]):  # Show top 9 missing skills
                            cols[i % 3].markdown(f"- **{skill}**")

                        st.info("💡 Tip: Focus on adding relevant experience and projects that demonstrate these skills!")

if __name__ == "__main__":
    main()
