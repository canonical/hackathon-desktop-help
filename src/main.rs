use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;

use anyhow::Result;
use clap::{Parser, Subcommand};

// Import the LLM module defined in llm.rs
mod llm;
use llm::{Message, OllamaClient};

// Default address where Ollama listens when installed locally
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
// Default model to use; small enough to run without a GPU
const DEFAULT_MODEL: &str = "deepseek-r1:1.5b";
// Default directory to load documentation markdown files from
const DEFAULT_DOCS_DIR: &str = "docs";
// Instruction given to the LLM at the start of every conversation
const SYSTEM_PROMPT: &str = "You are a helpful Ubuntu Desktop assistant. Answer questions clearly and concisely.";

// Top-level CLI struct; clap uses the fields and attributes to build argument parsing
#[derive(Parser)]
#[command(name = "ubuntu-desktop-help", about = "Ubuntu Desktop Help CLI")]
struct Cli {
    // Model name to use with Ollama; can be overridden by OLLAMA_MODEL env var or --model flag
    #[arg(long, env = "OLLAMA_MODEL", default_value = DEFAULT_MODEL, global = true)]
    model: String,

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
        Commands::Chat { ollama_url, docs_dir } => run_chat(ollama_url, cli.model, docs_dir).await,
    }
}

// Reads all .md files from `dir` and returns their contents joined with separators.
// Files that cannot be read are skipped with a warning printed to stderr.
fn load_docs(dir: &str) -> String {
    let path = Path::new(dir);

    // Return empty string silently if the directory doesn't exist
    if !path.is_dir() {
        eprintln!("Warning: docs directory '{dir}' not found; proceeding without documentation context.");
        return String::new();
    }

    let mut combined = String::new();

    // Read directory entries; skip if the directory itself can't be listed
    let mut entries: Vec<_> = match fs::read_dir(path) {
        Ok(iter) => iter.filter_map(|e| e.ok()).collect(),
        Err(e) => {
            eprintln!("Warning: could not read docs directory '{dir}': {e}");
            return String::new();
        }
    };

    // Sort entries by file name for deterministic ordering
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let entry_path = entry.path();

        // Only process files with a .md extension
        if entry_path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }

        match fs::read_to_string(&entry_path) {
            Ok(content) => {
                // Add a header so the LLM knows which file each excerpt comes from
                combined.push_str(&format!(
                    "\n\n--- Documentation: {} ---\n{content}",
                    entry_path.display()
                ));
            }
            Err(e) => eprintln!("Warning: could not read '{}': {e}", entry_path.display()),
        }
    }

    combined
}

// Runs the interactive chat loop, sending user input to Ollama and printing replies
async fn run_chat(ollama_url: String, model: String, docs_dir: String) -> Result<()> {
    let client = OllamaClient::new(ollama_url, model);
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // Load documentation files and append them to the system prompt so the LLM
    // can answer questions using their content
    let docs = load_docs(&docs_dir);
    let system_content = format!("{SYSTEM_PROMPT}{docs}");

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
