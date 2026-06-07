# Polyglot NIM - Document Q&A System

Ask questions about any document using NVIDIA NIM AI, built with 4 programming languages each doing what they do best.

## Architecture

| Language | Role | Why |
|----------|------|-----|
| **Python** | AI brain (embeddings + LLM) | Best ML ecosystem, NVIDIA NIM SDK |
| **Java** | Business API + auth | Enterprise-grade Spring Boot |
| **C++** | Document chunking | Raw I/O speed for large files |
| **Go** | API gateway | Lightweight, high-concurrency proxy |

## Quick Start

### Prerequisites
- Python 3.10+, Java 17, Go 1.21+, C++ compiler (g++)
- NVIDIA API key from [build.nvidia.com](https://build.nvidia.com)

### Setup

```bash
git clone https://github.com/KiMu-420/polyglot-nim.git
cd polyglot-nim
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt