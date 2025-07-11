# 🤖 AgentFabric: Agentic AI System

AgentFabric is a multi-agent AI system that uses a planner, web search, and summarizer agents to accomplish user-defined goals. Powered by Hugging Face models and LangChain.

## Features

- Planner agent using FLAN-T5
- Real-time DuckDuckGo search
- Automatic article summarization
- Shared memory between agents
- Streamlit UI

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py