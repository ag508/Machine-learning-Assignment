# Machine-learning-Assignment
# Job Recommendation System

A Python-based job recommendation system that uses natural language processing and content-based filtering to match job seekers with relevant job listings. The system generates synthetic job data and provides personalized job recommendations based on user skills and preferences.

## Features

- **Synthetic Job Data Generation**
  - Creates realistic job listings with varied titles, skills, and salary ranges
  - Supports multiple job categories (Software Engineering, Data Science)
  - Includes company information, locations, and experience levels

- **Advanced Text Processing**
  - Utilizes NLTK for natural language processing
  - Implements TF-IDF vectorization for text analysis
  - Includes custom text cleaning and normalization

- **Content-Based Recommendation Engine**
  - Matches users with jobs based on skill similarity
  - Calculates recommendation scores using cosine similarity
  - Provides detailed matching metrics including similarity scores and skill matches

## Requirements

```
pandas
numpy
scikit-learn
nltk
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/job-recommendation-system.git
cd job-recommendation-system
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn nltk
```

3. The system will automatically download required NLTK data on first run

## Usage

Run the main script to see the system in action:

```bash
python job_recommendation.py
```

The demo will:
1. Generate a sample dataset of job listings
2. Create a sample user profile
3. Preprocess the job data
4. Train the recommendation system
5. Generate and display personalized job recommendations

## Project Structure

- `JobListingsGenerator`: Creates synthetic job data with realistic attributes
- `DataPreprocessor`: Handles text cleaning and normalization
- `ContentBasedRecommender`: Implements the recommendation algorithm
- Helper functions for data display and system initialization

## Sample Output

The system will display recommendations in this format:
```
Recommendation 1:
Title: Data Scientist
Company: DataSys
Location: San Francisco, CA
Skills Required: Python, Machine Learning, SQL
Similarity Score: 0.85
Skills Match: 3
```

## Key Features

1. **Dynamic Job Categories**
   - Predefined job categories with corresponding skills and salary ranges
   - Easy to extend with additional categories and attributes

2. **Flexible Recommendation System**
   - Combines skill matching with content-based filtering
   - Configurable number of recommendations
   - Detailed similarity metrics

3. **Data Preprocessing**
   - Robust text cleaning
   - Handles missing data
   - Normalizes job descriptions and titles

## Future Improvements

- Add collaborative filtering capabilities
- Implement real-time job data fetching
- Add user interface for interaction
- Include more sophisticated matching algorithms
- Add support for resume parsing

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
