use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: Message,
}

pub struct OllamaClient {
    client: Client,
    url: String,
    model: String,
}

impl OllamaClient {
    pub fn new(url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            url,
            model,
        }
    }

    pub async fn chat(&self, messages: &[Message]) -> Result<String> {
        let req = ChatRequest {
            model: &self.model,
            messages,
            stream: false,
        };

        let endpoint = format!("{}/api/chat", self.url.trim_end_matches('/'));
        let resp: ChatResponse = self
            .client
            .post(&endpoint)
            .json(&req)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(resp.message.content)
    }
}
