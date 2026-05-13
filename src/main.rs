use std::io::{self, BufRead, Write};

use anyhow::Result;
use clap::{Parser, Subcommand};

mod llm;
use llm::{Message, OllamaClient};

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "tinyllama";
const SYSTEM_PROMPT: &str = "You are a helpful Ubuntu Desktop assistant. Answer questions clearly and concisely.";

#[derive(Parser)]
#[command(name = "ubuntu-desktop-help", about = "Ubuntu Desktop Help CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat {
        #[arg(long, env = "OLLAMA_URL", default_value = DEFAULT_OLLAMA_URL)]
        ollama_url: String,
        #[arg(long, env = "OLLAMA_MODEL", default_value = DEFAULT_MODEL)]
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat { ollama_url, model } => run_chat(ollama_url, model).await,
    }
}

async fn run_chat(ollama_url: String, model: String) -> Result<()> {
    let client = OllamaClient::new(ollama_url, model);
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    let mut messages = vec![Message {
        role: "system".to_string(),
        content: SYSTEM_PROMPT.to_string(),
    }];

    loop {
        print!("> ");
        stdout.flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                if input.eq_ignore_ascii_case("exit") {
                    break;
                }

                messages.push(Message {
                    role: "user".to_string(),
                    content: input.to_string(),
                });

                match client.chat(&messages).await {
                    Ok(reply) => {
                        println!("{reply}\n");
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: reply,
                        });
                    }
                    Err(e) => {
                        eprintln!("Error: {e}");
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
