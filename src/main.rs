use std::io::{self, BufRead, Write};

use anyhow::Result;
use clap::{Parser, Subcommand};

// Import the LLM module defined in llm.rs
mod llm;
use llm::{Message, OllamaClient};

// Default address where Ollama listens when installed locally
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
// Default model to use; small enough to run without a GPU
const DEFAULT_MODEL: &str = "tinyllama";
// Instruction given to the LLM at the start of every conversation
const SYSTEM_PROMPT: &str = "You are a helpful Ubuntu Desktop assistant. Answer questions clearly and concisely.";

// Top-level CLI struct; clap uses the fields and attributes to build argument parsing
#[derive(Parser)]
#[command(name = "ubuntu-desktop-help", about = "Ubuntu Desktop Help CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

// All available subcommands
#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat {
        // Ollama server URL; can be overridden by the OLLAMA_URL env var or --ollama-url flag
        #[arg(long, env = "OLLAMA_URL", default_value = DEFAULT_OLLAMA_URL)]
        ollama_url: String,
        // Model name to use for chat; can be overridden by OLLAMA_MODEL env var or --model flag
        #[arg(long, env = "OLLAMA_MODEL", default_value = DEFAULT_MODEL)]
        model: String,
    },
}

// Entry point; #[tokio::main] sets up the async runtime so we can use .await
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat { ollama_url, model } => run_chat(ollama_url, model).await,
    }
}

// Runs the interactive chat loop, sending user input to Ollama and printing replies
async fn run_chat(ollama_url: String, model: String) -> Result<()> {
    let client = OllamaClient::new(ollama_url, model);
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // Conversation history sent with every request so the LLM has context
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: SYSTEM_PROMPT.to_string(),
    }];

    loop {
        // Print prompt and flush immediately so it appears before the user types
        print!("> ");
        stdout.flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF (Ctrl-D)
            Ok(_) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                if input.eq_ignore_ascii_case("exit") {
                    break;
                }

                // Append the user turn before sending so the LLM sees the full history
                messages.push(Message {
                    role: "user".to_string(),
                    content: input.to_string(),
                });

                match client.chat(&messages).await {
                    Ok(reply) => {
                        println!("{reply}\n");
                        // Store the assistant reply so future turns have full context
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: reply,
                        });
                    }
                    Err(e) => {
                        eprintln!("Error: {e}");
                        // Remove the user message to keep history consistent with what the LLM has seen
                        messages.pop();
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        }
    }

    Ok(())
}
