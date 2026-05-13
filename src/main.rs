use std::io::{self, BufRead, Write};

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ubuntu-desktop-help", about = "Ubuntu Desktop Help CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat => run_chat(),
    }
}

fn run_chat() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().expect("Failed to flush stdout");

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF (Ctrl-D)
            Ok(_) => {
                let input = line.trim();
                if input.eq_ignore_ascii_case("exit") {
                    break;
                }
                println!("hello");
            }
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        }
    }
}
