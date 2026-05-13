use anyhow::Result;
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
    // When false, Ollama returns the full response at once instead of streaming tokens
    stream: bool,
}

// The JSON body returned by Ollama's /api/chat endpoint
#[derive(Deserialize)]
struct ChatResponse {
    message: Message,
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

    // Sends the full conversation history to Ollama and returns the assistant's reply text
    pub async fn chat(&self, messages: &[Message]) -> Result<String> {
        let req = ChatRequest {
            model: &self.model,
            messages,
            stream: false,
        };

        // Remove any trailing slash to avoid double-slash in the URL
        let endpoint = format!("{}/api/chat", self.url.trim_end_matches('/'));
        let resp: ChatResponse = self
            .client
            .post(&endpoint)
            .json(&req)
            .send()
            .await?
            // Converts HTTP 4xx/5xx responses into an Err instead of silently returning bad JSON
            .error_for_status()?
            .json()
            .await?;

        Ok(resp.message.content)
    }
}
