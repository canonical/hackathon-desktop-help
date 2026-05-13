use std::collections::HashMap;

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, Value, VectorParamsBuilder,
};

use crate::docs::Chunk;

// Qdrant collection name for the documentation index
const COLLECTION: &str = "ubuntu-docs";
// Output dimension of BGE-small-en-v1.5; must match the Qdrant collection config
const EMBEDDING_DIM: u64 = 384;
// Default number of chunks to retrieve per user query
pub const TOP_K: usize = 5;

// Holds a live Qdrant connection and the embedding model, reused across all queries
pub struct RagStore {
    client: Qdrant,
    embedder: TextEmbedding,
}

impl RagStore {
    // Builds the vector index from the given chunks, recreating the Qdrant collection each run
    // so the index stays in sync with docs/.
    pub async fn build(chunks: &[Chunk], qdrant_url: &str) -> Result<Self> {
        // block_in_place tells the tokio scheduler that this thread will block,
        // allowing other tasks to run on remaining threads; needed because fastembed
        // init is CPU/IO bound and may download the model (~130 MB) on first run.
        let mut embedder = tokio::task::block_in_place(|| {
            TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::BGESmallENV15)
                    .with_show_download_progress(true),
            )
        })
        .context("failed to initialize embedding model")?;

        // Connect to the Qdrant gRPC endpoint (default port 6334)
        let client = Qdrant::from_url(qdrant_url)
            .build()
            .with_context(|| {
                format!(
                    "cannot connect to Qdrant at {qdrant_url}\n  \
                     Start it with: sudo snap install qdrant && qdrant"
                )
            })?;

        // Delete and recreate the collection so the index is always fresh
        let existing = client
            .list_collections()
            .await
            .context("failed to list Qdrant collections")?;
        if existing.collections.iter().any(|c| c.name == COLLECTION) {
            client
                .delete_collection(COLLECTION)
                .await
                .context("failed to delete existing Qdrant collection")?;
        }
        client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION)
                    .vectors_config(VectorParamsBuilder::new(EMBEDDING_DIM, Distance::Cosine)),
            )
            .await
            .context("failed to create Qdrant collection")?;

        // Embed all chunk texts; CPU-bound so we block in place
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings: Vec<Vec<f32>> =
            tokio::task::block_in_place(|| embedder.embed(texts, None))
                .context("failed to embed document chunks")?;

        // Each PointStruct pairs a vector with source path and text stored as payload
        let points: Vec<PointStruct> = chunks
            .iter()
            .zip(embeddings)
            .enumerate()
            .map(|(i, (chunk, vec))| {
                let payload: HashMap<String, Value> = [
                    ("source".to_string(), Value::from(chunk.source.clone())),
                    ("text".to_string(), Value::from(chunk.text.clone())),
                ]
                .into_iter()
                .collect();
                PointStruct::new(i as u64, vec, payload)
            })
            .collect();

        if !points.is_empty() {
            client
                .upsert_points(UpsertPointsBuilder::new(COLLECTION, points))
                .await
                .context("failed to upsert document vectors into Qdrant")?;
        }

        Ok(Self { client, embedder })
    }

    // Returns the top-k most relevant (source, text) pairs for the given query.
    pub async fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<(String, String)>> {
        // Embed the query with the same model used during indexing
        let query_text = query.to_string();
        let query_vec: Vec<f32> = tokio::task::block_in_place(|| {
            self.embedder.embed(vec![query_text], None)
        })
        .context("failed to embed query")?
        .into_iter()
        .next()
        .context("embedder returned no vector for query")?;

        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(COLLECTION, query_vec, top_k as u64)
                    .with_payload(true),
            )
            .await
            .context("Qdrant similarity search failed")?;

        let results = response
            .result
            .into_iter()
            .map(|point| {
                let source = payload_str(&point.payload, "source");
                let text = payload_str(&point.payload, "text");
                (source, text)
            })
            .collect();

        Ok(results)
    }
}

// Extracts a string value from a Qdrant payload map; returns empty string if absent or wrong type.
fn payload_str(payload: &HashMap<String, Value>, key: &str) -> String {
    use qdrant_client::qdrant::value::Kind;
    payload
        .get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|kind| match kind {
            Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}


