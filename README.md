# Local Command-Line Chatbot

A command-line chat application that runs a text generation model locally. This project uses the `HuggingFaceTB/SmolLM2-360M-Instruct` model from Hugging Face and features a sliding window mechanism to maintain short-term conversational memory.

# Demonstration Video
Watch a walkthrough and explanation of this project here: [YouTube Explanation Video](https://youtu.be/Yx5KKt7v-1A?si=02ay0YVgckcXm2r7)

## Features

* **Local Model Inference**: All processing is done locally, requiring no API calls after the initial model download.
* **Conversational Memory**: The application retains the context of the last three conversational turns to ensure coherent and relevant responses.
* **Modular Architecture**: The source code is organized into separate modules for model loading, memory management, and the user interface, promoting maintainability.
* **Interactive CLI**: Provides a simple and continuous command-line interface for user interaction, with a graceful exit command.

## Setup

### Prerequisites

* Python 3.8+
* `pip` package installer

### Installation

1. **Clone the project:**

   ```bash
   git clone https://github.com/Ragu-123/Cli-Chat.git
   cd Cli-Chat
   ```

2. **Set up a Python virtual environment:**

   ```bash
   python -m venv cli-chat
   source cli-chat/bin/activate  # On Windows: cli-chat\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the chat application, execute the main interface script from the project's root directory:

```bash
python interface.py
```

### Sample Interaction

Below is a short example of the chatbot in action:

```
C:\Users\SEC\Downloads\banoa company\cli-chat>python interface.py
Initializing model: HuggingFaceTB/SmolLM2-360M-Instruct...
Device set to use cpu
Model, tokenizer and pipeline initialized successfully.

Chat session started. Type '/exit' to quit.
--------------------------------------------------
User: whats the capital of tamilnadu?
Bot: The capital of Tamil Nadu is Chennai.
User: what about kerala?
Bot: The capital of Kerala is Thiruvananthapuram.
User: /exit
Exiting. Goodbye!
```

## Documentation

For more detailed documentation, setup instructions, and usage examples, see the [Wiki](https://github.com/Ragu-123/Cli-Chat/wiki).
