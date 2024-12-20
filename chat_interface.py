# chat_interface.py

import os
import subprocess
import requests
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.theme import Theme
from rich import print
import openai
import anthropic

# Import new error classes directly from openai
from openai import APIError, APIConnectionError, APIStatusError, RateLimitError

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML


class ChatInterface:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # ========== LOGGING CONFIGURATION ==========
        # Clear existing logs by opening in 'w' mode instead of 'a' mode
        with open('chat_interface.log', 'w') as f:
            f.write('')  # Clear the file
        
        logging.basicConfig(
            filename='chat_interface.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # ========== MODEL REGISTRY ==========
        self.models = {
            # Anthropic Models
            "claude-haiku": {
                "backend": "anthropic",
                "model_id": "claude-3-haiku-20240307",
                "label": "Claude Haiku",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            },
            "claude-sonnet": {
                "backend": "anthropic",
                "model_id": "claude-3-5-sonnet-latest",
                "label": "Claude Sonnet",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            },
            # OpenAI Models
            "gpt-4o": {
                "backend": "openai",
                "model_id": "gpt-4o",
                "label": "GPT-4o",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
                "parameter_map": {
                    "max_tokens": "max_tokens"  # Default mapping
                }
            },
            "gpt-4o-mini": {
                "backend": "openai",
                "model_id": "gpt-4o-mini",
                "label": "GPT-4o Mini",
                "parameters": {
                    "temperature": 0.5,
                    "max_tokens": 2048,
                },
                "parameter_map": {
                    "max_tokens": "max_tokens"
                }
            },
            "o1-preview": {
                "backend": "openai",
                "model_id": "o1-preview",
                "label": "ChatGPT o1 Preview",
                "parameters": {
                    "temperature": 1,
                    "max_tokens": 4096,
                },
                "parameter_map": {
                    "max_tokens": "max_completion_tokens"
                }
            },
            "o1-mini": {
                "backend": "openai",
                "model_id": "o1-mini",
                "label": "ChatGPT o1 Mini",
                "parameters": {
                    "temperature": 1,
                    "max_tokens": 2048,
                },
                "parameter_map": {
                    "max_tokens": "max_completion_tokens"
                }
            },
            "o1": {
                "backend": "openai",
                "model_id": "o1-2024-12-17",
                "label": "ChatGPT o1",
                "parameters": {
                    "temperature": 1,
                    "max_tokens": 4096,
                },
                "parameter_map": {
                    "max_tokens": "max_completion_tokens"
                }
            },
            # XAI Grok Models
            "grok-beta": {
                "backend": "xai",
                "model_id": "grok-beta",
                "label": "Grok Beta",
                "parameters": {
                    "temperature": 0.6,
                    "max_tokens": 2048,
                }
            },
            "grok-2-1212": {
                "backend": "xai",
                "model_id": "grok-2-1212",
                "label": "Grok 2-1212",
                "parameters": {
                    "temperature": 0.5,
                    "max_tokens": 1024,
                }
            },
            # Ollama Models
            "llama3.2": {
                "backend": "ollama",
                "model_id": "llama3.2",
                "label": "Llama3.2",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                }
            }
        }

        # ========== INITIAL CONFIGURATION ==========
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("[warning] OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = self.openai_api_key

        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("[warning] Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None

        self.xai_api_key = os.getenv("XAI_API_KEY")
        if not self.xai_api_key:
            print("[warning] XAI API key not found. Please set the XAI_API_KEY environment variable.")

        # Set initial model
        self.current_model_key = "llama3.2"
        self.current_model = self.models[self.current_model_key]

        # Initialize conversation
        self.conversation = []

        # ========== UI SETUP ==========
        self.custom_theme = Theme({
            "info": "cyan",
            "warning": "yellow",
            "error": "bold red",
            "user": "green",
            "assistant": "blue"
        })

        self.console = Console(theme=self.custom_theme)

        self.session = PromptSession()
        self.bindings = KeyBindings()

        # Use Ctrl+J as a stand-in for Ctrl+Enter
        @self.bindings.add('c-j')
        def _(event):
            """
            When Ctrl+J is pressed, accept the input.
            """
            event.app.exit(result=event.app.current_buffer.text)

    # ========== METHODS ==========
    def show_welcome(self):
        self.console.clear()
        self.console.print(Panel.fit(
            "[bold cyan]Chat Interface[/]\n"
            "[dim]Press Ctrl+J to send your message (acts like Ctrl+Enter).[/]\n"
            "[dim]Type 'quit' to end or 'clear' to start new conversation.[/]\n"
            "[dim]Type 'backend <backend_name>' to switch backends.[/]\n"
            "[dim]Type 'model <model_name>' to switch models.[/]\n"
            "[dim]Type 'set <parameter> <value>' to adjust model parameters.[/]\n"
            "[dim]Type 'list models' to view all available models.[/]\n"
            "[dim]Type 'help' to view available commands.[/]",
            border_style="cyan"
        ))

    def build_prompt(self):
        prompt = ""
        for msg in self.conversation:
            if self.current_model["backend"] == "ollama":
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            elif self.current_model["backend"] == "anthropic":
                if msg["role"] == "user":
                    prompt += f"Human: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            elif self.current_model["backend"] == "openai":
                # Handled via API messages, no custom formatting needed
                pass
            elif self.current_model["backend"] == "xai":
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
        if self.conversation and self.conversation[-1]["role"] == "user":
            if self.current_model["backend"] == "ollama":
                prompt += "Assistant:"
            elif self.current_model["backend"] == "anthropic":
                prompt += "\n\nAssistant:"
            elif self.current_model["backend"] == "xai":
                prompt += "\n\nAssistant:"
        return prompt.strip()

    def get_response_from_anthropic(self):
        if not self.anthropic_api_key:
            return "Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable."

        try:
            # Update to use the new Claude 3 API format
            response = self.anthropic_client.messages.create(
                model=self.current_model["model_id"],
                messages=[
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in self.conversation
                ],
                max_tokens=self.current_model["parameters"]["max_tokens"],
                temperature=self.current_model["parameters"]["temperature"]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logging.error(f"Anthropic error: {str(e)}")
            return "Sorry, I couldn't process that request."

    def get_response_from_openai(self, retries=3, backoff=2):
        if not self.openai_api_key:
            return "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."

        messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in self.conversation
        ]

        for attempt in range(retries):
            try:
                api_params = {}
                for param, value in self.current_model["parameters"].items():
                    mapped_param = self.current_model.get("parameter_map", {}).get(param, param)
                    api_params[mapped_param] = value

                # Log the request details (excluding sensitive info)
                logging.debug(f"Attempting OpenAI request with model: {self.current_model['model_id']}")
                logging.debug(f"Parameters: {api_params}")

                response = openai.chat.completions.create(
                    model=self.current_model["model_id"],
                    messages=messages,
                    **api_params
                )

                # Extract token usage
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens

                # Log token usage
                logging.info(f"OpenAI API usage - Prompt Tokens: {prompt_tokens}, "
                             f"Completion Tokens: {completion_tokens}, Total Tokens: {total_tokens}")

                # Optionally, display token usage to the user
                self.console.print(f"[info]Token Usage: Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}[/]")

                return response.choices[0].message.content.strip()
            except RateLimitError as e:
                logging.error(f"Rate limit error: {str(e)}")
                return f"Rate limit exceeded: {str(e)}"
            except APIConnectionError as e:
                logging.error(f"Connection error: {str(e)}")
                return f"Connection error: {str(e)}"
            except APIStatusError as e:
                logging.error(f"Status error: {str(e)}")
                return f"API status error: {str(e)}"
            except APIError as e:
                logging.error(f"OpenAI API error: {str(e)}")
                return f"OpenAI API error: {str(e)}"
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                return f"An unexpected error occurred: {str(e)}"

        return "Sorry, I'm experiencing high load. Please try again later."

    def get_response_from_xai(self):
        if not self.xai_api_key:
            return "XAI API key is not set. Please set the XAI_API_KEY environment variable."

        headers = {
            "Authorization": f"Bearer {self.xai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.current_model["model_id"],
            "messages": [{"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."}] + [
                {"role": "user", "content": msg["content"]} for msg in self.conversation if msg["role"] == "user"
            ],
            "temperature": self.current_model["parameters"]["temperature"],
            "max_tokens": self.current_model["parameters"]["max_tokens"],
            "stream": False
        }

        endpoint = "https://api.x.ai/v1/chat/completions"

        try:
            response = requests.post(endpoint, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                logging.error("XAI Grok returned no choices.")
                return "Sorry, I couldn't process that request."
        except requests.exceptions.RequestException as e:
            logging.error(f"XAI Grok error: {e}")
            return "Sorry, I couldn't process that request."

    def get_response_from_ollama(self):
        prompt = self.build_prompt()
        try:
            result = subprocess.run(
                ["ollama", "run", self.current_model["model_id"], prompt],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Ollama error: {e.stderr.strip()}")
            return "Sorry, I couldn't process that request."
        except Exception as e:
            logging.error(f"Ollama unexpected error: {str(e)}")
            return "Sorry, I couldn't process that request."

    def get_response(self):
        backend = self.current_model["backend"]
        if backend == "anthropic":
            return self.get_response_from_anthropic()
        elif backend == "openai":
            return self.get_response_from_openai()
        elif backend == "xai":
            return self.get_response_from_xai()
        elif backend == "ollama":
            return self.get_response_from_ollama()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def handle_backend_switch(self, new_backend):
        supported_backends = ["anthropic", "openai", "ollama", "xai"]
        if new_backend in supported_backends:
            if new_backend == "anthropic" and not self.anthropic_api_key:
                self.console.print("[warning]Anthropic API key not set. Cannot switch to Anthropic backend.[/]")
            elif new_backend == "openai" and not self.openai_api_key:
                self.console.print("[warning]OpenAI API key not set. Cannot switch to OpenAI backend.[/]")
            elif new_backend == "xai" and not self.xai_api_key:
                self.console.print("[warning]XAI API key not set. Cannot switch to XAI backend.[/]")
            else:
                models_for_backend = [key for key, value in self.models.items() if value["backend"] == new_backend]
                if models_for_backend:
                    self.current_model_key = models_for_backend[0]
                    self.current_model = self.models[self.current_model_key]
                    self.console.print(f"[info]Switched backend to: {new_backend.capitalize()} and model to: {self.current_model['label']}[/]")
                else:
                    self.console.print(f"[warning]No models found for backend: {new_backend}[/]")
        else:
            self.console.print(f"[warning]Unknown backend: {new_backend}. Supported backends are: {', '.join(supported_backends)}[/]")

    def handle_model_switch(self, model_key):
        if model_key in self.models:
            model_backend = self.models[model_key]["backend"]
            if model_backend == "anthropic" and not self.anthropic_api_key:
                self.console.print("[warning]Anthropic API key not set. Cannot switch to this model.[/]")
            elif model_backend == "openai" and not self.openai_api_key:
                self.console.print("[warning]OpenAI API key not set. Cannot switch to this model.[/]")
            elif model_backend == "xai" and not self.xai_api_key:
                self.console.print("[warning]XAI API key not set. Cannot switch to this model.[/]")
            else:
                self.current_model_key = model_key
                self.current_model = self.models[self.current_model_key]
                self.console.print(f"[info]Switched to model: {self.current_model['label']}[/]")
        else:
            self.console.print(f"[warning]Unknown model: {model_key}. Use 'list models' to see available models.[/]")

    def handle_set_parameter(self, param, value):
        if param in self.current_model["parameters"]:
            try:
                if param == "temperature":
                    value_converted = float(value)
                    if not (0.0 <= value_converted <= 1.0):
                        self.console.print("[warning]Temperature must be between 0 and 1.[/]")
                        return
                elif param == "max_tokens":
                    value_converted = int(value)
                    if not (0 < value_converted <= 8192):
                        self.console.print("[warning]max_tokens must be between 1 and 8192.[/]")
                        return
                else:
                    value_converted = value

                self.current_model["parameters"][param] = value_converted
                self.console.print(f"[info]Set {param} to {value_converted} for model: {self.current_model['label']}[/]")
            except ValueError:
                self.console.print("[warning]Invalid value type for the parameter.[/]")
        else:
            self.console.print(f"[warning]Parameter '{param}' is not supported for the current model.[/]")

    def handle_command(self, user_message):
        if user_message.lower() == 'quit':
            self.console.print("\n[bold red]Goodbye![/]")
            self.print_session_summary()
            exit(0)

        if user_message.lower() == 'clear':
            self.conversation = []
            self.show_welcome()
            return

        if user_message.lower().startswith("backend "):
            new_backend = user_message.lower().split("backend ", 1)[1].strip()
            self.handle_backend_switch(new_backend)
            return

        if user_message.lower().startswith("model "):
            model_key = user_message.lower().split("model ", 1)[1].strip()
            self.handle_model_switch(model_key)
            return

        if user_message.lower().startswith("set "):
            try:
                _, param, value = user_message.lower().split(" ", 2)
                self.handle_set_parameter(param, value)
            except ValueError:
                self.console.print("[warning]Invalid set command format. Use 'set <parameter> <value>'.[/]")
            return

        if user_message.lower() == 'help':
            self.console.print(Panel.fit(
                "[bold cyan]Available Commands[/]\n"
                "- `quit`: Exit the chat interface.\n"
                "- `clear`: Start a new conversation.\n"
                "- `backend <backend_name>`: Switch between 'anthropic', 'openai', 'ollama', or 'xai' backends.\n"
                "- `model <model_name>`: Switch to a specific model.\n"
                "- `set <parameter> <value>`: Adjust model parameters (e.g., temperature, max_tokens).\n"
                "- `list models`: List all available models.\n"
                "- `help`: View available commands.\n"
                "- Example: `backend anthropic`\n"
                "- Example: `model claude-haiku`\n"
                "- Example: `set temperature 0.8`\n"
                "- Example: `list models`",
                border_style="cyan"
            ))
            return

        if user_message.lower() == 'list models':
            model_list = "\n".join([
                f"- {key}: {value['label']} (Backend: {value['backend'].capitalize()})"
                for key, value in self.models.items()
            ])
            self.console.print(Panel.fit(
                f"[bold cyan]Available Models[/]\n{model_list}",
                border_style="cyan"
            ))
            return

    def print_session_summary(self):
        # Placeholder for session summary logic
        # Implement tracking of token usage as needed
        self.console.print("[bold cyan]Session Summary:[/]")
        # Example summary
        # self.console.print("OpenAI API Usage: ...")
        # self.console.print("Anthropic API Usage: ...")
        # self.console.print("XAI API Usage: ...")

    def run(self):
        self.show_welcome()

        while True:
            try:
                user_message = self.session.prompt(
                    "You: ",
                    key_bindings=self.bindings,
                    multiline=True,
                    mouse_support=False,
                    wrap_lines=True
                ).strip()

                if not user_message:
                    continue

                # Handle commands
                if self.is_command(user_message):
                    self.handle_command(user_message)
                    continue

                # Append user message to conversation
                self.conversation.append({"role": "user", "content": user_message})

                # Get response from the current backend
                with self.console.status(f"[assistant]{self.current_model['label']} is thinking...[/]", spinner="dots"):
                    response_text = self.get_response()

                # Append assistant response to conversation
                self.conversation.append({"role": "assistant", "content": response_text})

                # Display assistant response
                self.console.print(
                    Panel(
                        Markdown(response_text),
                        title=f"[assistant]{self.current_model['label']}[/]",
                        border_style="blue",
                        padding=(1, 2)
                    )
                )

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Goodbye![/]")
                self.print_session_summary()
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                self.console.print(f"\n[error]An error occurred: {str(e)}[/]")
                self.console.print("[warning]Please try again.[/]")

    def is_command(self, message):
        commands = ['quit', 'clear', 'backend', 'model', 'set', 'list models', 'help']
        return any(message.lower().startswith(cmd) for cmd in commands)


if __name__ == "__main__":
    chat_interface = ChatInterface()
    chat_interface.run()