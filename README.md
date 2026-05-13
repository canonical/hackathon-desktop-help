# Desktop help app

An experimental LLM-based application written in Rust for users to get answers about using Ubuntu.

## Build Instructions

### Prerequisites

- **Rust** (1.70+): Install from [rustup.rs](https://rustup.rs/)
- **Ollama**: Local LLM inference engine

### Setup

First, clone the repository and change into the root of the repo.

#### 1. Install Ollama

```bash
sudo snap install ollama
```

#### 2. Pull the Required Model

The app uses `deepseek-r1:1.5b` by default. Pull it with:

```bash
ollama pull deepseek-r1:1.5b
```

To use a different model, pass it via the `--model` flag or `OLLAMA_MODEL` environment variable.

#### 3. Build the Application

```bash
cargo build --release
```

The binary will be available at `target/release/ubuntu-desktop-help`.

### Running

```bash
# Start the interactive chat interface
./target/release/ubuntu-desktop-help chat

# Or use the debug build for development
cargo run -- chat
```

### System Requirements

- Ollama service running (automatically started via snap)
- At least 2GB of RAM for the `deepseek-r1:1.5b` model
- ~2GB disk space for the model
