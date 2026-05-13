use std::io::{self, BufRead, Write};
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

// Import the LLM module defined in llm.rs
mod llm;
use llm::{CopilotClient, LlmClient, Message, OllamaClient};

// Import the docs chunking module defined in docs.rs
mod docs;
use docs::load_chunks;

// Default address where Ollama listens when installed locally
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
// Default model to use; small enough to run without a GPU
const DEFAULT_MODEL: &str = "deepseek-r1:1.5b";
// Default directory to load documentation markdown files from
const DEFAULT_DOCS_DIR: &str = "docs";
// Instruction given to the LLM at the start of every conversation
const SYSTEM_PROMPT: &str = include_str!("../cli-system-prompt.md");

// Top-level CLI struct; clap uses the fields and attributes to build argument parsing
#[derive(Parser)]
#[command(name = "ubuntu-desktop-help", about = "Ubuntu Desktop Help CLI")]
struct Cli {
    // Local Ollama model name (e.g. tinyllama, phi3:mini); mutually exclusive with --copilot
    #[arg(long, env = "OLLAMA_MODEL", default_value = DEFAULT_MODEL, global = true, conflicts_with = "copilot")]
    model: String,

    // Use GitHub Copilot via the GitHub Models API instead of a local model; mutually exclusive with --model
    #[arg(long, global = true, conflicts_with = "model")]
    copilot: bool,

    #[command(subcommand)]
    command: Commands,
}

// All available subcommands
#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat {
        // Ollama server URL; only used when --copilot is not set
        #[arg(long, env = "OLLAMA_URL", default_value = DEFAULT_OLLAMA_URL)]
        ollama_url: String,
        // Directory containing markdown documentation files to inject as context
        #[arg(long, env = "DOCS_DIR", default_value = DEFAULT_DOCS_DIR)]
        docs_dir: String,
    },
}

// Entry point; #[tokio::main] sets up the async runtime so we can use .await
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat { ollama_url, docs_dir } => {
            run_chat(ollama_url, cli.model, cli.copilot, docs_dir).await
        }
    }
}

// Runs the interactive chat loop, sending user input to the chosen LLM backend and printing replies
async fn run_chat(ollama_url: String, model: String, use_copilot: bool, docs_dir: String) -> Result<()> {
    // Build the appropriate backend based on whether --copilot was passed
    let client = if use_copilot {
        eprintln!("Authenticating with GitHub Copilot…");
        LlmClient::Copilot(CopilotClient::create().await?)
    } else {
        LlmClient::Ollama(OllamaClient::new(ollama_url, model))
    };
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // Load and chunk documentation files; format each chunk with its source label
    // so the LLM knows where the information comes from
    let chunks = load_chunks(&docs_dir);
    let docs_context: String = chunks
        .iter()
        .map(|c| format!("\n\n[Source: {}]\n{}", c.source, c.text))
        .collect();
    let system_content = format!("{SYSTEM_PROMPT}{docs_context}");

    // Conversation history sent with every request so the LLM has context
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: system_content,
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

                // Show a spinner while waiting for the first token from the LLM
                let spinner = ProgressBar::new_spinner();
                spinner.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner} Thinking…")
                        .unwrap(),
                );
                spinner.enable_steady_tick(Duration::from_millis(80));

                // Pass a callback that clears the spinner the moment the first token arrives
                match client.chat(&messages, || spinner.finish_and_clear()).await {
                    Ok(reply) => {
                        // Tokens were already printed by the streaming chat call; just add spacing
                        println!();
                        // Store the assistant reply so future turns have full context
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: reply,
                        });
                    }
                    Err(e) => {
                        spinner.finish_and_clear();
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
