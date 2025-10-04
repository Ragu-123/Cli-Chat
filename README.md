# Local Command-Line Chatbot

A command-line chat application that runs a text generation model locally. This project uses the `TinyLlama-1.1B-Chat` model from Hugging Face and features a sliding window mechanism to maintain short-term conversational memory.

## Features

- **Local Model Inference**: All processing is done locally, requiring no API calls after the initial model download.
- **Conversational Memory**: The application retains the context of the last three conversational turns to ensure coherent and relevant responses.
- **Modular Architecture**: The source code is organized into separate modules for model loading, memory management, and the user interface, promoting maintainability.
- **Interactive CLI**: Provides a simple and continuous command-line interface for user interaction, with a graceful exit command.

## Setup

### Prerequisites

- Python 3.8+
- `pip` package installer

### Installation

1.  **Clone the project:**
    ```bash
    git clone <your-repository-url>
    cd <project-directory>
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the chat application, execute the main interface script from the project's root directory:

```bash
python interface.py