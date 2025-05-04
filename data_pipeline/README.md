# Survey Data Pipeline

This module implements the data science abstraction layer for survey analysis. It uses vector embeddings stored in Qdrant to enable sophisticated analysis and AI-powered insights.

## Architecture

The data pipeline follows a multi-tiered architecture:

1. **Data Lake & Vector Database Integration**
   - Raw survey data stored in MongoDB
   - Vector embeddings stored in Qdrant
   - Analysis metadata stored for caching

2. **Multi-tiered Analysis Engine**
   - Base Analysis (Real-time)
   - Metric-Specific Analysis (Near Real-time)
   - Cross-Metric Intelligence (Background Processing)
   - Predictive Insights (Scheduled Processing)

3. **AI Integration Layer**
   - OpenAI for embedding generation and analysis
   - Specialized prompts for different analysis tasks
   - Hybrid approach combining statistical methods and AI

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- OpenAI API key

### Installation

1. Install dependencies:

```bash
pip install -r data_pipeline/requirements.txt
```

2. Ensure Qdrant is running:

```bash
docker-compose up -d qdrant
```

3. Set environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key
export QDRANT_URL=http://localhost:6333
```

## Usage

### Process a Survey

To process a survey and generate embeddings:

```bash
python -m data_pipeline.process_survey <survey_id>
```

This will:
1. Retrieve survey data from MongoDB
2. Generate embeddings for questions, responses, and metrics
3. Store embeddings in Qdrant collections

### Integration with Main Application

The data pipeline is designed to be called from the main application. You can integrate it by:

1. Importing the process_survey function
2. Calling it when a survey is created or updated
3. Triggering analysis when needed

```python
from data_pipeline.process_survey import process_survey

# Process a survey
await process_survey(survey_id)
```

## Testing

The data pipeline includes a comprehensive test suite with both unit and integration tests.

### Running Tests

To run the full test suite:

```bash
cd data_pipeline
./run_tests.sh
```

This will execute all unit and integration tests with detailed output.

### Test Structure

- **Unit Tests**: Test individual components in isolation
  - Embedding Service
  - Qdrant Client Service
  - Data Processing Functions

- **Integration Tests**: Test interactions between components
  - Survey Processing Pipeline
  - Data Flow Between Components

### Writing New Tests

When adding new features, ensure you:
1. Add unit tests for individual functions
2. Update integration tests if component interactions change
3. Use the provided fixtures in `conftest.py` for mocking dependencies

## Next Steps

See the full architecture document at `docs/survey-insights-architecture.md` for the complete implementation roadmap. 