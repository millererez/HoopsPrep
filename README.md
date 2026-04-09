# HoopsPrep: Agentic AI System for NBA Pre-Game Briefings

## About The Project
**HoopsPrep** is an end-to-end AI system that generates structured, broadcast-style NBA pre-game briefings in seconds. 

This project automates the manual research process for sports analysts, fans and fantasy basketball players.  
By quickly synthesizing raw ESPN statistics and dynamic web narratives into an easily digestible format, it functions similar to tools used by sports analysts and broadcasters to prepare for game night.

Rather than acting as a simple API wrapper, HoopsPrep leverages an **Agentic AI Architecture** to autonomously fetch real-time data, perform dynamic Retrieval-Augmented Generation (RAG), and evaluate its own output before presenting it to the user.

## Try it Live: [hoopsprep.onrender.com](https://hoopsprep.onrender.com/)

## Demo


https://github.com/user-attachments/assets/3c0f6d70-6867-4c20-ae5b-749230ccc731



## Key Technical Highlights
* **Agentic Workflow & Reflection (LangGraph):** Implements a multi-agent Directed Acyclic Graph (DAG) featuring parallel data fetching and a single-pass review-and-refine loop powered by OpenAI models, improving output consistency and factual grounding.
* **Within-Session RAG (ChromaDB):** Employs an ephemeral RAG pattern. Live search data is fetched via Tavily, chunked, and upserted on-the-fly using `OpenAI text-embedding-3-small`. The composer node queries this immediate context, ensuring up-to-the-minute relevance rather than relying on a static historical corpus.
* **Date-Scoped Caching (SQLite):** To optimize performance and reduce API costs, the backend implements a date-scoped local cache. Briefings requested for the same game on the same day skip the LLM pipeline, while next-day requests trigger a fresh generation.
* **Strict Rate Limiting (SlowAPI):** Protects against abuse and request spikes with strict IP-based limits: `10 requests/minute` for the daily scoreboard and `3 requests/minute` for the intensive briefing generation endpoint.
* **Responsive Frontend (Streamlit):** A clean, dark-mode optimized UI designed for quick interaction. Features seamless clipboard integration using `pyperclip` (with a raw text fallback) to instantly export the generated insights.
* **Full Containerization:** Both the backend API and frontend UI are fully dockerized, allowing for a seamless one-command deployment.

## System Architecture

The core logic is orchestrated via LangGraph, utilizing parallel execution and a dedicated review cycle:

```text
START 
  ├──► Data Specialist (ESPN API + Stats) ──────┐
  └──► Context Extractor (Tavily Search + RAG) ─┤ (Parallel Execution)
                                                ▼
                                        Narrative Composer
                                                │
                                            Reviewer (Validates structure and factual consistency)
                                            /       \
                             (Issues Found) /         \ (Approved)
                                  Rewrite Node         │
                                          \           / (Single-pass only)
                                           ▼         ▼
                                       Assemble Node (Final Output)
                                                │
                                               END

```
## Technology Stack
- **AI & Orchestration:** Python, LangGraph, OpenAI  
  - **LLMs:** gpt-4o, gpt-4o-mini  
  - **Embeddings:** text-embedding-3-small  
  - **Vector Store:** ChromaDB
    
  *Different models are used across agents based on task complexity and cost-performance tradeoffs.*


* **Backend:** FastAPI, Pydantic, SQLite, SlowAPI

* **Frontend:** Streamlit

* **External APIs:** ESPN, Tavily

* **DevOps:** Docker, Docker Compose, Render(Cloud Deployment)

## Running Locally (For Developers)
### Prerequisites

- Docker  
- Docker Compose  

> **Note:** Running this project locally requires your own active **OpenAI** and **Tavily** API keys.


### Environment Setup
Create a `.env` file in the root directory and add the following:

```env
# Required external APIs
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Internal authentication (create a secure string of your choice)
HOOPSPREP_API_KEY=your_secure_custom_key
```

### Installation & Execution (One-Command Launch)
The entire application (FastAPI backend + Streamlit frontend) is fully containerized. Run the following command:
```
docker-compose up -d --build
```
### Access the Application
Frontend UI: http://localhost:8501  
Backend API(Swagger docs): http://localhost:8000/docs

To stop the application, run:
```
docker-compose down
```

### Background
This project was inspired by my experience at the "Kol HaSport" sports communications course, where I learned the meticulous process of preparing broadcast briefings. This is my attempt to solve that real-world problem through modern Software Engineering.
