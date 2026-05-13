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

mod gui;
mod search_provider;

// Default address where Ollama listens when installed locally
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
// Default model to use; small enough to run without a GPU
const DEFAULT_MODEL: &str = "deepseek-r1:1.5b";
// Default directory to load documentation markdown files from
const DEFAULT_DOCS_DIR: &str = "docs";
// Instruction given to the LLM at the start of every conversation
pub(crate) const SYSTEM_PROMPT: &str = "You are a helpful Ubuntu Desktop assistant. Answer questions clearly and concisely. The user you are talking to is running Ubuntu. Do not offer advice on alternative operating systems. Prefer strongly the information that you receive as context within a session.";

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
    /// Run the GNOME Shell search provider over D-Bus (auto-activated; not normally run by hand)
    Dbus,
    /// Open the GTK answer window for a single query (invoked by the search provider)
    Gui {
        /// The question to ask; remaining args are joined with spaces. If absent,
        /// the window opens with a placeholder message instead of calling the LLM.
        #[arg(trailing_var_arg = true)]
        query: Vec<String>,
    },
}

// Plain entry point: the gui subcommand needs to run GTK's main loop, so we
// build a tokio runtime only for the subcommands that actually need it.
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat { ollama_url, docs_dir } => {
            tokio_runtime()?.block_on(run_chat(ollama_url, cli.model, cli.copilot, docs_dir))
        }
        Commands::Dbus => {
            tokio_runtime()?.block_on(search_provider::run())
        }
        Commands::Gui { query } => gui::run(query.join(" "), cli.copilot, cli.model),
    }
}

fn tokio_runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
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

                // Pass two callbacks: one fires when the spinner should clear,
                // one fires per streamed token so we can print it to stdout.
                let on_first_token = || spinner.finish_and_clear();
                let on_token = |t: &str| {
                    print!("{t}");
                    let _ = io::stdout().flush();
                };
                match client.chat(&messages, on_first_token, on_token).await {
                    Ok(reply) => {
                        // Tokens were printed by the on_token callback; add a trailing newline
                        println!();
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
