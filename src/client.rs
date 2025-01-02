use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use simplepir::{Matrix, Vector, query, recover_row};
use crate::embeddings::TextEmbedder;

#[derive(Debug, Deserialize)]
pub struct DatabaseConfig {
    pub mod_power: u32,
    pub secret_dimension: usize,
    pub plain_mod: u64,
    pub db_side_len: Option<usize>,
    pub server_hints: Option<(u64, u64)>,
    pub client_hints: Option<(Vec<Vec<u64>>, Vec<Vec<u64>>)>,
}

#[derive(Debug, Serialize)]
struct QueryRequest {
    query_cipher: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct QueryResponse {
    answer: Vec<u64>,
}

pub async fn get_db_config(base_url: &str) -> Result<DatabaseConfig> {
    let url = format!("{}/db-config", base_url);
    let response = Client::new().get(&url).send().await?;
    let config = response.json::<DatabaseConfig>().await?;
    Ok(config)
}

pub async fn query_embedding(base_url: &str, query_cipher: Vec<u64>) -> Result<Vec<u64>> {
    let url = format!("{}/query/embedding", base_url);
    let request = QueryRequest { query_cipher };
    let response = Client::new().post(&url).json(&request).send().await?;
    let result = response.json::<QueryResponse>().await?;
    Ok(result.answer)
}

pub async fn query_text(base_url: &str, query_cipher: Vec<u64>) -> Result<Vec<u64>> {
    let url = format!("{}/query/text", base_url);
    let request = QueryRequest { query_cipher };
    let response = Client::new().post(&url).json(&request).send().await?;
    let result = response.json::<QueryResponse>().await?;
    Ok(result.answer)
}

pub fn find_closest_index(embedding_answer: &[u64]) -> usize {
    embedding_answer
        .iter()
        .enumerate()
        .max_by_key(|&(_, value)| value)
        .map(|(index, _)| index)
        .unwrap_or(0)
}
