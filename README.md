#**INTRO**
**CineScope – AI Movie Recommendation System**
CineScope is a personalized movie recommendation engine powered by FastAPI, XGBoost, and Truncated SVD.
It features a simple, responsive interface that allows users to filter movies by genre, year, actor, or rating and explore related recommendations.

#**Overview**
CineScope is a hybrid recommendation system that combines:
Collaborative filtering (Truncated SVD)
Content-based features (genre embeddings, ratings, etc.)
Boosted regression (XGBoost) for improved prediction accuracy
The web app provides:
Movie search and filtering
“Something Like This” exploration mode
A minimal, dark-themed frontend
Integration with a FastAPI backend for real-time recommendations


#**Features**
FastAPI backend for machine learning predictions
Real-time recommendation API (similar and personalized movies)
Responsive frontend using HTML, CSS, and JavaScript
Smart filters for genre, year, actor, language, and rating
Lightweight one-click automation script (start_app.ps1)
Easy dataset extension and retraining support



#**Tech Stack**
Layer	Tools / Libraries
Backend	FastAPI, Python, Uvicorn
Machine Learning	XGBoost, Scikit-learn, TruncatedSVD
Frontend	HTML, CSS (TailwindCSS), JavaScript
Automation	PowerShell
Dataset	MovieLens Small Dataset


#**Project Structure**
MovieRecommend/
│
├── backend/
│   ├── app.py                # FastAPI backend server
│   ├── data/                 # Processed data (MovieLens)
│   ├── __pycache__/          # Cache files
│   └── requirements.txt
│
├── frontend/
│   ├── index.html            # Main landing page
│   ├── movies.html           # Movie listings with filters
│   ├── recommendations.html  # Similar movie explorer
│
├── start_app.ps1             # Automation script
└── README.md                 # Project documentation



#**How to Run Locally**
--Option 1: One-Click Start (Windows)
Double-click start_app.ps1
Wait for the consoles to open automatically
Access:
Backend → http://127.0.0.1:8000
Frontend → http://127.0.0.1:5500/frontend/index.html

--Option 2: Manual Setup
Navigate to backend
cd backend
Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Run backend
uvicorn app:app --reload

--Then open a new terminal:
Serve frontend
cd ..
python -m http.server 5500
Open the browser and visit:
http://127.0.0.1:5500/frontend/index.html


#**API Endpoints**
Endpoint	Method	Description
/health	GET	Check API status
/api/movies?limit=10	GET	Retrieve sample movie list
/api/similar?movie_id=1	GET	Fetch similar movies
/api/recommend	POST	Generate hybrid recommendations



#**Future Improvements**
Add TMDB API for live movie posters and trailers
Add user authentication and profile preferences
Introduce neural network-based recommendations
Add dark/light theme toggle
Include usage analytics dashboard

#**Author**
Suhaeb
VIT-AP University, Amaravathi
B.Tech in CSE
Email: suhaebn.shaik@gmail.com


#**License**
This project is licensed under the MIT License.
See the LICENSE file for more details.

#**Acknowledgements**
MovieLens Dataset by GroupLens Research – https://grouplens.org/datasets/movielens/
FastAPI Documentation – https://fastapi.tiangolo.com/
TailwindCSS Framework – https://tailwindcss.com/
Scikit-learn Documentation – https://scikit-learn.org/stable/
XGBoost Documentation – https://xgboost.readthedocs.io/en/stable/
