# Scam Detection Gelato 🍦

A Chrome extension for real-time scam detection using LLM-RAG technology, featuring a delightful gelato-themed UI.

## 🌟 Features

- **Real-time Text Analysis**: Detect potential scams as you browse
- **Smart Selection**: Analyze selected text with a friendly gelato interface
- **Risk Assessment**: Multi-level risk evaluation with visual feedback
- **Pattern Recognition**: Identify common scam patterns and suspicious content
- **Similar Case Detection**: Compare with known scam cases
- **Interactive UI**: Engaging gelato-themed interface with smooth animations
- **Detailed Reports**: Comprehensive analysis with actionable recommendations

## 🛠 Technology Stack

- **Frontend**: JavaScript (Chrome Extension)
- **Backend**: Python (FastAPI)
- **AI/ML**: 
  - RAG (Retrieval Augmented Generation)
  - FAISS for similarity search
  - LLM integration (DeepSeek Chat)
- **Database**: PostgreSQL
- **Additional Tools**: 
  - pandas for data processing
  - numpy for numerical operations
  - dotenv for configuration management

## 📋 Requirements

```txt
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
openai==1.3.5
pandas==2.1.3
numpy==1.26.2
psycopg2-binary==2.9.9
faiss-cpu==1.7.4
pydantic==2.5.2
requests==2.31.0
```

## 🚀 Installation & Setup

1. **Backend Setup**:
```bash
# Clone the repository
git clone https://github.com/yourusername/scam-detection-gelato.git

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configurations

# Start the backend server
uvicorn api:app --reload
```

2. **Chrome Extension Setup**:
- Open Chrome and navigate to `chrome://extensions/`
- Enable "Developer mode"
- Click "Load unpacked" and select the `chrome-extension` directory

## 🗂 Project Structure

```
scam-detection-gelato/
├── api.py                 # FastAPI backend server
├── requirements.txt       # Python dependencies
├── .env                  # Environment configuration
├── data/
│   └── faiss_index.bin   # FAISS similarity index
├── rag/
│   └── rag_system.py     # RAG implementation
├── utils/
│   └── faiss_manager.py  # FAISS utilities
└── chrome-extension/
    ├── manifest.json     # Extension configuration
    ├── content/
    │   └── content.js    # Content script
    ├── styles/
    │   └── risk-visuals.css  # Styling
    └── icons/            # Extension icons
```

## 📊 Component Analysis

### Backend Components

1. **API Server (api.py)**
- Handles text analysis requests
- Manages database connections
- Coordinates RAG system operations
- **Key Features**:
  - Async request handling
  - Error management
  - Response formatting
- **Improvement Opportunities**:
  - Add request rate limiting
  - Implement caching
  - Add batch processing

2. **RAG System (rag_system.py)**
- Implements core analysis logic
- Manages LLM integration
- Handles pattern matching
- **Key Features**:
  - Multi-level risk assessment
  - Pattern recognition
  - Similar case retrieval
- **Improvement Opportunities**:
  - Enhance pattern matching
  - Implement multi-language support
  - Add model fallback options

### Frontend Components

1. **Content Script (content.js)**
- Manages text selection
- Handles UI interactions
- Processes analysis results
- **Key Features**:
  - Smart positioning
  - Smooth animations
  - Error handling
- **Improvement Opportunities**:
  - Add keyboard shortcuts
  - Implement context menu integration
  - Add history tracking

2. **Styling (risk-visuals.css)**
- Defines visual appearance
- Manages animations
- Handles responsive design
- **Key Features**:
  - Smooth transitions
  - Risk-level theming
  - Responsive layout
- **Improvement Opportunities**:
  - Add dark mode
  - Improve accessibility
  - Add more animation variants

## 🔄 Data Flow

1. **Text Selection**:
   - User selects text
   - Content script validates selection
   - Gelato icon appears

2. **Analysis Request**:
   - User clicks gelato icon
   - Request sent to backend
   - Loading animation shown

3. **Backend Processing**:
   - Text analyzed by RAG system
   - Similar cases retrieved
   - Risk assessment performed

4. **Result Display**:
   - Results returned to frontend
   - UI updated with risk level
   - Report panel displayed

## 🛡 Security Considerations

- API endpoint protection
- Data sanitization
- Error handling
- Rate limiting
- CORS configuration

## 🔜 Future Improvements

1. **Technical Enhancements**:
   - Implement offline mode
   - Add browser sync support
   - Improve performance optimization

2. **Feature Additions**:
   - URL scanning
   - Automatic language detection
   - Custom risk thresholds
   - User preferences

3. **UI/UX Improvements**:
   - More interactive animations
   - Customizable themes
   - Improved accessibility
   - Mobile support

4. **Backend Optimizations**:
   - Enhanced caching
   - Load balancing
   - Database optimization
   - API versioning

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct before submitting pull requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for LLM technology
- Facebook Research for FAISS
- FastAPI team
- Chrome Extensions community

## 📞 Support

For support, please open an issue in the GitHub repository or contact [your-email@domain.com].
