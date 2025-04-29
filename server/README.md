# rForms FastAPI Backend

This directory contains the FastAPI backend for the rForms application.

## Getting Started

### Prerequisites

- Python 3.11+
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Server

Start the development server:

```bash
uvicorn app.main:app --reload
```

Or use the provided script:

```bash
chmod +x run.sh  # Make executable
./run.sh
```

The server will be available at http://localhost:8000

### API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

### Project Structure

```
app/
├── api/            # API route handlers
├── core/           # Core application logic
├── db/             # Database models and operations
├── services/       # Business logic services
├── schemas/        # Pydantic schemas
└── main.py         # Application entry point
```

### Adding New Endpoints

1. Create your route handlers in the appropriate file under `app/api/`
2. If needed, add Pydantic models in `app/schemas/`
3. Add business logic in `app/services/`
4. Register your routes in `app/main.py` 