# rForms Project Structure

This document outlines the project structure for the rForms application, a goal-oriented adaptive questionnaire system.

## Root Directory Structure

```
/rforms
├── client/                 # Frontend React application
├── server/                 # Backend FastAPI application
├── common/                 # Shared code and types
├── db/                     # Database migrations and schemas
├── docker/                 # Docker configuration
├── infra/                  # Infrastructure as Code
├── docs/                   # Documentation
└── .github/                # GitHub workflows and templates
```

## Client Directory (`/client`)

```
/client
├── public/                 # Static assets
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── admin/          # Admin-specific components
│   │   ├── survey/         # Survey-specific components
│   │   └── ui/             # ShadcnUI components and extensions
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utility functions
│   ├── pages/              # Page components
│   │   ├── admin/          # Admin portal pages
│   │   └── survey/         # Survey respondent pages
│   ├── services/           # API client services
│   ├── store/              # State management
│   ├── styles/             # Global styles and Tailwind config
│   └── types/              # TypeScript type definitions
├── tests/                  # Frontend tests
├── .eslintrc.js            # ESLint configuration
├── package.json            # NPM package definition
├── tailwind.config.js      # Tailwind CSS configuration
└── tsconfig.json           # TypeScript configuration
```

## Server Directory (`/server`)

```
/server
├── app/
│   ├── api/                # API route handlers
│   │   ├── auth/           # Authentication endpoints
│   │   ├── goals/          # Goal management endpoints
│   │   ├── sessions/       # Survey session endpoints
│   │   └── analytics/      # Analytics endpoints
│   ├── core/               # Core application logic
│   │   ├── config.py       # Application configuration
│   │   ├── security.py     # Authentication and security
│   │   └── logging.py      # Logging configuration
│   ├── db/                 # Database models and operations
│   │   ├── models/         # SQLAlchemy models
│   │   ├── repositories/   # Data access repositories
│   │   └── session.py      # Database session management
│   ├── services/           # Business logic services
│   │   ├── llm/            # LLM integration services
│   │   ├── metrics/        # Metrics mapping services
│   │   └── questions/      # Question generation services
│   ├── schemas/            # Pydantic schemas for request/response
│   └── main.py             # Application entry point
├── tests/                  # Backend tests
├── pyproject.toml          # Python project configuration
├── requirements.txt        # Python dependencies
└── Dockerfile              # Server container definition
```

## Database Directory (`/db`)

```
/db
├── migrations/             # Alembic migration scripts
├── init/                   # Initial database setup scripts
├── scripts/                # Database utility scripts
└── alembic.ini             # Alembic configuration
```

## Docker Directory (`/docker`)

```
/docker
├── docker-compose.yml      # Multi-container Docker composition
├── docker-compose.dev.yml  # Development overrides
├── .env.example            # Example environment variables
├── nginx/                  # Nginx configuration for production
└── postgres/               # PostgreSQL configuration
```

## Common Directory (`/common`)

```
/common
├── schemas/                # Shared data schemas
├── prompts/                # LLM prompt templates
└── types/                  # Shared TypeScript types
```

## Infrastructure Directory (`/infra`)

```
/infra
├── terraform/              # Terraform IaC
│   ├── modules/            # Reusable Terraform modules
│   ├── environments/       # Environment-specific configurations
│   └── variables.tf        # Terraform variables
├── kubernetes/             # Kubernetes manifests (if needed)
└── scripts/                # Deployment and infrastructure scripts
```

## Documentation Directory (`/docs`)

```
/docs
├── rforms.md               # PRD document
├── project_structure.md    # This document
├── api/                    # API documentation
├── architecture/           # Architecture diagrams and docs
└── setup/                  # Setup and installation guides
```

## GitHub Directory (`/.github`)

```
/.github
├── workflows/              # GitHub Actions workflow definitions
│   ├── ci.yml              # Continuous Integration
│   └── deploy.yml          # Deployment workflow
├── ISSUE_TEMPLATE/         # Issue templates
└── PULL_REQUEST_TEMPLATE.md # PR template
``` 