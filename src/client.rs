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
        .min_by_key(|&(_, value)| value)
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::EncodedString;

    #[tokio::test]
    async fn test_full_query_flow() -> Result<()> {
        let base_url = "http://127.0.0.1:8080";
        
        // Get database configuration
        println!("Fetching database config...");
        let config = get_db_config(base_url).await?;
        let db_side_len = config.db_side_len.expect("Database not initialized");
        let server_hints = config.server_hints.expect("Server hints not available");
        let client_hints = config.client_hints.expect("Client hints not available");
        println!("✓ Got database config");
        
        // Create query embedding
        println!("\nCreating query embedding...");
        let prompt = "What is the price of Bitcoin?";
        let embedder = TextEmbedder::new()?;
        let tensor = embedder.embed(&prompt)?;
        let embedded_query = crate::utils::scale_to_u64(tensor)?;
        let embedding_answer = query_embedding(base_url, embedded_query).await?;
        println!("✓ Got embedding answer");
        
        // Find closest index
        println!("\nFinding closest text...");
        let closest_index = find_closest_index(&embedding_answer);
        println!("✓ Found closest at index: {}", closest_index);
        
        // Query text database
        println!("\nQuerying text database...");
        let (client_state_txt, query_cipher_txt) = query(
            closest_index,
            db_side_len,
            config.secret_dimension,
            server_hints.1,
            config.plain_mod
        );
        let text_answer = query_text(base_url, query_cipher_txt.clone().data).await?;
        println!("✓ Got text answer");
        
        // Recover text
        println!("\nRecovering text...");
        let client_hint_matrix = Matrix::from_data(client_hints.1);
        let text_vector = recover_row(
            &client_state_txt,
            &client_hint_matrix,
            &Vector::from_vec(text_answer),
            &query_cipher_txt,
            config.plain_mod
        );
        let encoded = EncodedString(text_vector.data);
        let text: String = encoded.into();
        println!("Retrieved text: {}", text);
        
        Ok(())
    }
} 