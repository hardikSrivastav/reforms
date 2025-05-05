# RForms Data Pipeline

The RForms Data Pipeline provides utilities for fetching, transforming, and analyzing survey data from the RForms API.

## Setup

1. Create a virtual environment and install dependencies:

```bash
cd data_pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up environment variables (create a `.env` file in the `data_pipeline` directory):

```
OPENAI_API_KEY=your_openai_api_key
```

## Fetching Data from API

The `api_fetcher.py` utility allows you to fetch survey data from the RForms API:

```bash
# Fetch survey data and responses
python -m data_pipeline.utils.api_fetcher \
  --survey_id 4 \
  --base_url "http://localhost:8000" \
  --output_dir "./data" \
  --all
```

Options:
- `--survey_id`: ID of the survey to fetch
- `--base_url`: Base URL of the API
- `--api_key`: API key for authentication (if required)
- `--output_dir`: Directory to save files to (default: ./data)
- `--all`: Fetch all responses (may be slow for large surveys)

## Processing Survey Data

The main processing script transforms and analyzes survey data:

```bash
# Process survey data
python -m data_pipeline.process_survey \
  --survey "./data/survey_4.json" \
  --responses "./data/responses_4.json" \
  --output "./data/analysis_4.json"
```

Options:
- `--survey`: Path to survey data JSON file
- `--responses`: Path to responses data JSON file
- `--output`: Path to write results to (default: analysis_results.json)

## Analysis Output

The analysis output includes:

- Survey information
- Metric-specific analysis
  - Statistical analysis
  - Visualizations
  - AI-generated insights
- Cross-metric analysis
  - Correlation analysis
  - Key relationships
  - Visualizations
- AI-generated survey summary

## Customizing Analysis

You can customize the analysis by modifying the following modules:

- `utils/data_transformers.py`: Data transformation utilities
- `analysis/metric_analysis.py`: Metric-specific analysis
- `analysis/cross_metric_analysis.py`: Cross-metric analysis
- `services/ai_insights.py`: AI-powered insights generation
