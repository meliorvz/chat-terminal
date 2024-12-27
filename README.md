# Multi-Model Chat Interface

A versatile Python-based chat interface that supports multiple AI models including OpenAI's GPT models, Anthropic's Claude models, XAI's Grok models, Deepseek models, and local Ollama models. This interface provides a unified command-line experience for interacting with various AI models.

## Features

- Support for multiple AI backends:
  - OpenAI (GPT-4 and variants)
  - Anthropic (Claude 3 models)
  - XAI (Grok models)
  - Deepseek (Deepseek Chat)
  - Ollama (Local models)
- Interactive command-line interface with rich text formatting
- Real-time model switching
- Adjustable model parameters (temperature, max tokens)
- Conversation history management
- Detailed logging
- Multi-line input support with Ctrl+J for submission

## Prerequisites

- Python 3.8 or higher
- API keys for the services you plan to use (OpenAI, Anthropic, XAI)
- Ollama installed locally (if using local models)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-model-chat.git
   cd multi-model-chat
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   XAI_API_KEY=your_xai_key_here
   DEEPSEEK_API_KEY=your_deepseek_key_here
   ```

## Usage

1. Start the chat interface:
   ```bash
   python chat_interface.py
   ```

2. Available commands:
   - `quit`: Exit the chat interface
   - `clear`: Start a new conversation
   - `backend <backend_name>`: Switch between 'anthropic', 'openai', 'ollama', 'xai', or 'deepseek' backends
   - `model <model_name>`: Switch to a specific model
   - `set <parameter> <value>`: Adjust model parameters
   - `list models`: View available models
   - `help`: View available commands

3. Type your message and press Ctrl+J to send (acts like Ctrl+Enter)

## Configuration

The interface comes with pre-configured models and parameters, which can be modified in the `models` dictionary within the `ChatInterface` class.

## Logging

All interactions and errors are logged to `chat_interface.log` for debugging and monitoring purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 