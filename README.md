# Argunet 🤖

> **A real-time ML-powered debate platform that evaluates arguments based on logic, relevance, civility, and strength. Built using BERT, FastAPI, and Streamlit.**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![BERT Model](https://img.shields.io/badge/BERT-Transformers-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Project Overview

Argunet is an intelligent debate evaluation platform designed to foster critical thinking and structured argumentation. It combines state-of-the-art natural language processing with real-time evaluation to provide comprehensive feedback on argument quality.

### 🎯 Core Mission

To democratize debate analysis and help users improve their critical thinking skills through **fair, data-driven argument evaluation** using advanced machine learning techniques.

---

## 🧠 Technical Architecture

### AI/ML Stack
- **NLP Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Framework**: Hugging Face Transformers
- **ML Purpose**: Multi-dimensional argument analysis and classification

### Backend
- **API Framework**: FastAPI (async, high-performance)
- **Server**: Uvicorn
- **Language**: Python 3.8+

### Frontend
- **UI Framework**: Streamlit
- **User Experience**: Interactive, real-time debate interface
- **Real-time Updates**: WebSocket support

---

## ✨ Key Features

### 🔍 Multi-Dimensional Argument Scoring
Evaluates arguments across four critical dimensions:
- **Logic Score** - Coherence, reasoning validity, and logical fallacy detection
- **Relevance Score** - Topic alignment and addressing the proposition
- **Civility Score** - Tone analysis, respect, and communication quality
- **Strength Score** - Evidence quality, argument construction, and persuasiveness

### ⚡ Real-Time Analysis
- Instant feedback on argument submission
- Live scoring with detailed breakdowns
- Immediate improvement suggestions

### 👥 Interactive Debate Format
- Structured debate rounds
- Turn-based argument exchange
- Comparative argument analysis
- Historical debate tracking

### 📊 Analytics & Insights
- Argument performance metrics
- User skill progression tracking
- Debate statistics and trends
- Detailed scoring explanations

### 🤖 Intelligent Processing
- Automatic logical fallacy detection
- Sentiment and tone analysis
- Argument summarization
- Topic understanding

---

## 🏗️ Project Structure

```
argunet/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── models/              # BERT model interfaces
│   │   ├── routes/              # API endpoints
│   │   ├── schemas/             # Pydantic models for validation
│   │   └── utils/               # Helper functions
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app.py                   # Streamlit main application
│   ├── pages/                   # Multi-page Streamlit app
│   └── components/              # Reusable UI components
├── models/                      # Pre-trained BERT models
├── tests/
│   ├── test_api.py
│   └── test_scoring.py
├── README.md
├── .gitignore
└── docker-compose.yml
```

---

## 🔧 Scoring Methodology

### BERT-Based Evaluation
The platform uses a fine-tuned BERT model to analyze arguments:

1. **Input Processing**: Argument tokenization and embedding generation
2. **Context Understanding**: Contextual analysis of the debate topic
3. **Multi-Task Learning**: Simultaneous evaluation across four dimensions
4. **Score Aggregation**: Weighted scoring system (0-100)
5. **Reasoning Output**: Explainable AI - why each score was assigned

### Scoring Formula
```
Overall Score = (0.25 × Logic) + (0.25 × Relevance) + (0.25 × Civility) + (0.25 × Strength)
```

---

## 🎓 How It Works

### Argument Evaluation Flow
```
User Input Argument
        ↓
Tokenization & Embedding
        ↓
BERT Contextual Analysis
        ↓
Multi-Dimensional Scoring
        ↓
Feedback Generation
        ↓
Real-Time Display
```

### Debate Lifecycle
1. **Topic Proposal** - Define the debate proposition
2. **Argument Submission** - Users submit arguments for/against
3. **Real-Time Scoring** - Instant AI-powered evaluation
4. **Feedback Loop** - Detailed scoring explanations
5. **Comparative Analysis** - View all arguments side-by-side
6. **Debate Conclusion** - Statistics and winner determination

---

## 📚 Model Details

### BERT Configuration
- **Model Type**: BERT-base-uncased
- **Vocabulary Size**: 30,522 tokens
- **Hidden Layers**: 12
- **Attention Heads**: 12
- **Fine-tuning**: Custom trained on debate corpus

### Input/Output Specifications
**Input**: Text arguments (50-500 tokens recommended)
**Output**: 
```json
{
  "logic_score": 85,
  "relevance_score": 92,
  "civility_score": 88,
  "strength_score": 79,
  "overall_score": 86,
  "feedback": "Strong argument with minor logical considerations",
  "fallacies_detected": ["minor appeal to emotion"]
}
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended for optimal model performance)
- Internet connection for initial model download

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/Fadhil0315/argunet.git
cd argunet
```

#### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure Environment
```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

#### 5. Download BERT Model
```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('bert-base-uncased'); \
AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')"
```

---

## 📋 Requirements

### Core Dependencies
```
fastapi==0.104.0
uvicorn[standard]==0.24.0
streamlit==1.28.0
transformers==4.34.0
torch==2.0.0
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0
pydantic==2.4.0
python-dotenv==1.0.0
requests==2.31.0
```

### Optional Dependencies
```
pytest==7.4.0           # Testing
pytest-cov==4.1.0       # Coverage reports
black==23.10.0          # Code formatting
pylint==3.0.0           # Linting
jupyter==1.0.0          # Notebooks
```

### System Requirements
- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 2GB for models and dependencies
- **GPU**: Optional (NVIDIA CUDA 11.8+ for acceleration)

---

## ⚙️ Configuration

### Environment Variables (.env)
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Model Configuration
MODEL_NAME=bert-base-uncased
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=32

# Streamlit Configuration
STREAMLIT_PORT=8501
STREAMLIT_SERVER_HEADLESS=True

# Database (if applicable)
DATABASE_URL=sqlite:///./argunet.db

# CORS Settings
ALLOWED_ORIGINS=["http://localhost:3000"]
```

---

## 💻 Usage

### Running the Backend API
```bash
# Start FastAPI server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Server will be available at: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Running the Frontend
```bash
# In a new terminal (with venv activated)
streamlit run frontend/app.py

# Open browser: http://localhost:8501
```

### Using Docker (Optional)
```bash
# Build and run with docker-compose
docker-compose up --build

# Services will be available at:
# - API: http://localhost:8000
# - Frontend: http://localhost:8501
```

---

## 🔌 API Endpoints

### Base URL
```
http://localhost:8000/api/v1
```

### Evaluate Argument
**POST** `/arguments/evaluate`

```bash
curl -X POST "http://localhost:8000/api/v1/arguments/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "argument": "Climate change is primarily caused by human activities because CO2 levels correlate with industrial revolution.",
    "topic": "Climate change causes",
    "position": "for"
  }'
```

**Response:**
```json
{
  "id": "arg_12345",
  "logic_score": 85,
  "relevance_score": 92,
  "civility_score": 95,
  "strength_score": 79,
  "overall_score": 88,
  "feedback": "Well-structured argument with strong evidence correlation",
  "fallacies_detected": [],
  "timestamp": "2024-05-21T10:30:00Z"
}
```

### Create Debate
**POST** `/debates/create`

```bash
curl -X POST "http://localhost:8000/api/v1/debates/create" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Should AI be regulated?",
    "description": "Debate on AI regulation necessity",
    "topic": "Artificial Intelligence"
  }'
```

### Submit Argument to Debate
**POST** `/debates/{debate_id}/submit`

```bash
curl -X POST "http://localhost:8000/api/v1/debates/123/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "argument": "Your argument text here",
    "position": "for"
  }'
```

### Get Debate Results
**GET** `/debates/{debate_id}/results`

```bash
curl "http://localhost:8000/api/v1/debates/123/results"
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage Report
```bash
pytest tests/ --cov=backend --cov-report=html
# Open htmlcov/index.html in browser
```

### Run Specific Test File
```bash
pytest tests/test_scoring.py -v
```

### Test Argument Scoring
```bash
pytest tests/test_scoring.py::test_bert_scoring -v
```

---

## 📊 Example Usage

### Interactive Debate Session

**Step 1: Create a debate**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/debates/create",
    json={
        "title": "Is remote work more productive?",
        "topic": "Work Environment"
    }
)
debate_id = response.json()["id"]
```

**Step 2: Submit arguments**
```python
arg1 = {
    "argument": "Remote work reduces commute time by 2 hours daily...",
    "position": "for"
}
response = requests.post(
    f"http://localhost:8000/api/v1/debates/{debate_id}/submit",
    json=arg1
)
print(response.json())
```

**Step 3: View results**
```python
results = requests.get(
    f"http://localhost:8000/api/v1/debates/{debate_id}/results"
).json()
print(f"Winning argument: {results['winner']}")
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/argunet.git
cd argunet
```

### Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### Make Changes & Commit
```bash
git add .
git commit -m "Add feature: description of changes"
```

### Push & Create Pull Request
```bash
git push origin feature/your-feature-name
```

### Contribution Guidelines
- Follow PEP 8 style guide
- Write descriptive commit messages
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass: `pytest tests/`

---

## 🗺️ Roadmap

- [ ] Multi-language support (French, Spanish, Mandarin)
- [ ] Advanced user analytics dashboard
- [ ] Debate tournaments and competitions
- [ ] Mobile app (React Native)
- [ ] Integration with social platforms
- [ ] Custom fine-tuned models per domain
- [ ] Real-time collaboration features
- [ ] Blockchain-based argument verification
- [ ] Voice argument support with speech-to-text
- [ ] Advanced logical fallacy database

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Support & Community

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/Fadhil0315/argunet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Fadhil0315/argunet/discussions)
- **Email**: fadhil0315@example.com

### Report a Bug
Found a bug? [Create an issue](https://github.com/Fadhil0315/argunet/issues/new) with:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment details

---

## 👨‍💼 Authors & Contributors

- **Fadhil0315** - Project Creator & Lead Developer

---

## 🙏 Acknowledgments

- Hugging Face for the BERT model and Transformers library
- FastAPI and Streamlit communities
- Debate enthusiasts and early testers

---

## 📞 Contact

- **GitHub**: [@Fadhil0315](https://github.com/Fadhil0315)
- **Repository**: [Argunet](https://github.com/Fadhil0315/argunet)

---

**Made with ❤️ to improve critical thinking and debate quality worldwide.**
