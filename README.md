# forza4

## Setup

1. Copia il file `.env.example` in `.env`:
   ```bash
   cp .env.example .env
   ```

2. Apri `.env` e incolla le tue API key prese da [OpenRouter](https://openrouter.ai/keys).

3. Installa le dipendenze con `uv`:
   ```bash
   uv sync
   ```

4. Attiva il virtual environment:
   ```bash
   # Linux/macOS
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

5. Avvia l'applicazione:
   ```bash
   uv run main.py
   ```