import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random

# Download all required NLTK data
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f'Downloading {resource}...')
            nltk.download(resource, quiet=True)

# Download NLTK data at startup
print("Initializing NLTK resources...")
download_nltk_data()

def create_job_categories():
    return {
        'Software Engineering': {
            'titles': ['Software Engineer', 'Full Stack Developer', 'Backend Engineer'],
            'skills': ['Python', 'Java', 'JavaScript', 'SQL', 'AWS', 'Docker'],
            'salary_ranges': {
                'Entry Level': (70000, 100000),
                'Mid Level': (90000, 140000),
                'Senior Level': (130000, 200000)
            }
        },
        'Data Science': {
            'titles': ['Data Scientist', 'Machine Learning Engineer', 'AI Engineer'],
            'skills': ['Python', 'Machine Learning', 'SQL', 'Statistics', 'TensorFlow'],
            'salary_ranges': {
                'Entry Level': (75000, 105000),
                'Mid Level': (95000, 145000),
                'Senior Level': (135000, 210000)
            }
        }
    }

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text using basic preprocessing."""
        if not isinstance(text, str):
            return ""
        # Simple text cleaning without tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())  # Remove extra whitespace
        return text
    
    def preprocess_job_data(self, jobs_df):
        """Preprocess job listings data."""
        df = jobs_df.copy()
        df['description_cleaned'] = df['description'].apply(self.clean_text)
        df['title_cleaned'] = df['title'].apply(self.clean_text)
        return df

class JobListingsGenerator:
    def __init__(self):
        self.job_categories = create_job_categories()
        self.companies = ['TechCorp', 'DataSys', 'InnovateNow', 'FutureTech', 'CloudScale']
        self.locations = ['San Francisco, CA', 'New York, NY', 'Remote', 'Seattle, WA']

    def generate_job_description(self, category, title, skills, experience_level):
        description = f"""
{title} position
Required Skills: {', '.join(skills)}
Experience Level: {experience_level}
We are seeking a {experience_level} {title} with expertise in {', '.join(skills)}.
The ideal candidate will have strong problem-solving abilities and excellent communication skills.
"""
        return description.strip()

    def generate_dataset(self, num_listings=20):
        jobs_data = []
        job_id = 1000

        for _ in range(num_listings):
            category = random.choice(list(self.job_categories.keys()))
            category_data = self.job_categories[category]
            experience_level = random.choice(['Entry Level', 'Mid Level', 'Senior Level'])
            
            title = random.choice(category_data['titles'])
            skills = random.sample(category_data['skills'], min(3, len(category_data['skills'])))
            salary_range = category_data['salary_ranges'][experience_level]

            job_entry = {
                'job_id': job_id,
                'title': title,
                'category': category,
                'company': random.choice(self.companies),
                'location': random.choice(self.locations),
                'experience_level': experience_level,
                'skills_required': skills,
                'salary_min': salary_range[0],
                'salary_max': salary_range[1],
                'description': self.generate_job_description(
                    category, title, skills, experience_level
                ),
                'posting_date': (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
            }
            
            jobs_data.append(job_entry)
            job_id += 1

        return pd.DataFrame(jobs_data)

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.job_vectors = None
        self.jobs_df = None
    
    def fit(self, jobs_df):
        self.jobs_df = jobs_df
        text_data = jobs_df['description_cleaned'] + ' ' + jobs_df['title_cleaned']
        self.job_vectors = self.vectorizer.fit_transform(text_data)
    
    def recommend_jobs(self, user_profile, n_recommendations=5):
        user_text = ' '.join(user_profile['skills'])
        user_vector = self.vectorizer.transform([user_text])
        similarity_scores = cosine_similarity(user_vector, self.job_vectors)
        
        top_indices = similarity_scores[0].argsort()[-n_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            job = self.jobs_df.iloc[idx]
            recommendations.append({
                'job_id': job.name,
                'title': job['title'],
                'similarity_score': similarity_scores[0][idx],
                'skills_match': len(set(job['skills_required']).intersection(set(user_profile['skills'])))
            })
        
        return recommendations

def print_job_details(job, score=None, skills_match=None):
    print(f"\nTitle: {job['title']}")
    print(f"Company: {job['company']}")
    print(f"Location: {job['location']}")
    print(f"Skills Required: {', '.join(job['skills_required'])}")
    if score is not None:
        print(f"Similarity Score: {score:.2f}")
    if skills_match is not None:
        print(f"Skills Match: {skills_match}")

def main():
    print("Starting Job Recommendation System Demo...")
    
    # 1. Generate job listings
    print("\n1. Generating job listings...")
    generator = JobListingsGenerator()
    job_listings = generator.generate_dataset(20)
    print(f"Generated {len(job_listings)} job listings")
    
    # Print sample of generated jobs
    print("\nSample of generated jobs:")
    for _, job in job_listings.head(3).iterrows():
        print_job_details(job)
    
    # 2. Create sample user profile
    user_profile = {
        'user_id': 1,
        'skills': ['Python', 'Machine Learning', 'SQL'],
        'experience': 'Mid Level',
        'preferred_title': 'Data Scientist'
    }
    print("\n2. Created sample user profile:")
    print(f"Skills: {', '.join(user_profile['skills'])}")
    print(f"Experience: {user_profile['experience']}")
    
    # 3. Preprocess job data
    print("\n3. Preprocessing job data...")
    preprocessor = DataPreprocessor()
    processed_jobs = preprocessor.preprocess_job_data(job_listings)
    
    # 4. Initialize and train recommender
    print("\n4. Training recommendation system...")
    recommender = ContentBasedRecommender()
    recommender.fit(processed_jobs)
    
    # 5. Get recommendations
    print("\n5. Generating recommendations...")
    recommendations = recommender.recommend_jobs(user_profile, n_recommendations=5)
    
    # 6. Display recommendations
    print("\nTop Job Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        job = job_listings.iloc[rec['job_id']]
        print(f"\nRecommendation {i}:")
        print_job_details(job, rec['similarity_score'], rec['skills_match'])

if __name__ == "__main__":
    main()