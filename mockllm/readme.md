1. https://github.com/Zipstack/rubberduck?tab=readme-ov-file
2. https://github.com/StacklokLabs/mockllm
-
Backend
Language: Python 3
Framework: FastAPI (Chosen for its high performance, support for asynchronous code, and automatic Swagger documentation generation).
Server: Uvicorn (ASGI web server implementation for Python).
Database: SQLite (Used for lightweight, local storage of metrics, logs, and proxy configurations).
ORM / Data Validation: SQLAlchemy alongside Pydantic (for strict type checking and data serialization).
Proxying/Routing: httpx (Asynchronous HTTP client used to forward your requests to upstream models like OpenAI or Anthropic).
Frontend
Core Library: React (Functional components with Hooks).
Language: TypeScript (For static typing and safer code).
Build Tool: Vite (For extremely fast hot-module-reloading and optimized production builds).
Styling: Tailwind CSS (Utility-first CSS framework, used extensively for the dark-mode "glass" aesthetics).
Data Visualization: Recharts (A composable charting library built on React components, used for the interactive metrics dashboard).
Icons: Lucide React (Clean, consistent iconography throughout the UI).
