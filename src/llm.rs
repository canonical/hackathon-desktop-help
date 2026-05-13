use std::io::{self, Write};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

// A single message in a conversation, following the OpenAI/Ollama chat format
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Message {
    // Either "system", "user", or "assistant"
    pub role: String,
    pub content: String,
}

// The JSON body sent to Ollama's /api/chat endpoint
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    // When true, Ollama sends one JSON object per line as tokens are generated
    stream: bool,
}

// One line of the streaming response from Ollama's /api/chat endpoint
#[derive(Deserialize)]
struct StreamChunk {
    // Partial message; content is a single token or small group of tokens
    message: Message,
    // True on the final chunk, indicating the response is complete
    done: bool,
}

// HTTP client wrapping the Ollama REST API
pub struct OllamaClient {
    // reqwest's async HTTP client; reusing one instance is more efficient than creating per-request
    client: Client,
    // Base URL of the Ollama server, e.g. "http://localhost:11434"
    url: String,
    // Name of the model to use, e.g. "tinyllama"
    model: String,
}

impl OllamaClient {
    // Creates a new client; does not open any network connections yet
    pub fn new(url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            url,
            model,
        }
    }

    // Streams the assistant reply token-by-token, printing each token immediately as it arrives.
    // `on_first_token` is called once, just before the first character is printed — use it to
    // clear a spinner or other loading indicator.
    // Returns the full assembled reply string when done.
    pub async fn chat(&self, messages: &[Message], on_first_token: impl FnOnce()) -> Result<String> {
        let req = ChatRequest {
            model: &self.model,
            messages,
            stream: true,
        };

        // Remove any trailing slash to avoid double-slash in the URL
        let endpoint = format!("{}/api/chat", self.url.trim_end_matches('/'));
        let response = self
            .client
            .post(&endpoint)
            .json(&req)
            .send()
            .await?
            // Converts HTTP 4xx/5xx responses into an Err instead of silently returning bad JSON
            .error_for_status()?;

        // Ollama streams newline-delimited JSON; each chunk is one line
        let mut stream = response.bytes_stream();
        let mut full_reply = String::new();
        // Buffer to accumulate bytes until we have a complete newline-terminated JSON line
        let mut buf = Vec::new();
        let stdout = io::stdout();
        let mut out = stdout.lock();
        // Wrap in Option so we can call it exactly once with on_first_token.take()
        let mut on_first_token = Some(on_first_token);

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.context("error reading stream chunk")?;
            buf.extend_from_slice(&bytes);

            // Process all complete lines in the buffer (Ollama sends one JSON object per line)
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                // Split off the line, leaving the remainder in buf
                let line_bytes = buf.drain(..=pos).collect::<Vec<_>>();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let chunk: StreamChunk =
                    serde_json::from_str(line).context("failed to parse stream chunk")?;

                if !chunk.message.content.is_empty() {
                    // Fire the callback the first time we have content to print
                    if let Some(f) = on_first_token.take() {
                        f();
                    }
                    // Print the token immediately without a newline so output flows continuously
                    write!(out, "{}", chunk.message.content)?;
                    out.flush()?;
                    full_reply.push_str(&chunk.message.content);
                }

                if chunk.done {
                    break;
                }
            }
        }

        // Move to a new line after all tokens have been printed
        writeln!(out)?;

        Ok(full_reply)
    }
}
