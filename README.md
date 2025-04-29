# rForms: Goal-Oriented Adaptive Questionnaire

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)

rForms is an AI-powered adaptive survey platform that enables businesses to create goal-oriented questionnaires without writing a single question. Simply define your high-level goal (e.g., "measure consumer sentiment for Dish X"), and rForms will automatically generate, serve, and analyze adaptive surveys that intelligently drill down on the metrics that matter to you.

## ✨ Features

- 🧠 **AI-Generated Metrics & Questions**: Enter a goal, get survey-ready metrics and questions
- 🔄 **Adaptive Questioning**: Questions adapt based on previous answers
- 📊 **Real-time Analytics**: Instant insights on metrics and completion rates
- 🔒 **Secure & Scalable**: JWT authentication and horizontally scalable architecture
- 📱 **Modern UI**: Clean, responsive interface built with React, TypeScript, and Tailwind CSS

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v16+)
- [Python](https://www.python.org/) (v3.11+)
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- [PostgreSQL](https://www.postgresql.org/) (v14+)
- [Redis](https://redis.io/) (v6+)

### Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/rforms.git
   cd rforms
   ```

2. **Set up environment variables**

   ```bash
   cp docker/.env.example docker/.env
   # Edit the .env file with your configuration
   ```

3. **Start the development environment**

   ```bash
   docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
   ```

4. **Install frontend dependencies**

   ```bash
   cd client
   npm install
   npm run dev
   ```

5. **Install backend dependencies**

   ```bash
   cd server
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🏗️ Project Structure

For a detailed breakdown of the project structure, please refer to [docs/project_structure.md](docs/project_structure.md).

## 📚 Documentation

- [Product Requirements Document](docs/rforms.md)
- [API Documentation](http://localhost:8000/docs) (when running locally)
- [Architecture Overview](docs/architecture/) (coming soon)

## 🧪 Testing

### Frontend Tests

```bash
cd client
npm test
```

### Backend Tests

```bash
cd server
pytest
```

## 🔧 Technology Stack

- **Frontend**: React, TypeScript, Tailwind CSS, ShadcnUI
- **Backend**: Python 3.11, FastAPI
- **Database**: PostgreSQL (primary), Redis (caching/session)
- **LLM & AI**: OpenAI API (via Python SDK)
- **Deployment**: Docker, Vercel (frontend), AWS/GCP (backend)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📬 Contact

Project Link: [https://github.com/yourusername/rforms](https://github.com/yourusername/rforms)

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/yourusername">Hardik Srivastava</a>
</p> 