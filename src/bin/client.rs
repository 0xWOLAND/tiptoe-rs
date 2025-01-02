use anyhow::Result;
use simplepir::{Matrix, Vector, query, recover_row, Database};
use tiptoe_rs::{
    embeddings::TextEmbedder,
    encoding::EncodedString,
};

#[tokio::main]
async fn main() -> Result<()> {
    let base_url = "http://127.0.0.1:8080";
    let query_text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is the price of Bitcoin?".to_string());
    
    println!("=== Tiptoe Client ===");
    println!("Query: {}", query_text);
    
    // Get database configuration
    println!("\nFetching database config...");
    let config = tiptoe_rs::client::get_db_config(base_url).await?;
    let db_side_len = config.db_side_len.expect("Database not initialized");
    let server_hints = config.server_hints.expect("Server hints not available");
    let client_hints = config.client_hints.expect("Client hints not available");
    println!("✓ Got database config");
    
    // Create query embedding
    println!("\nCreating query embedding...");
    let embedder = TextEmbedder::new()?;
    let tensor = embedder.embed(&query_text)?;
    let query_embedding = tiptoe_rs::utils::scale_to_u64(tensor)?;
    println!("✓ Created query embedding");
    
    let embedding_answer = tiptoe_rs::client::query_embedding(base_url, query_embedding).await?;
    println!("✓ Got embedding answer");
    
    // Find closest index
    println!("\nFinding closest text...");
    let closest_index = tiptoe_rs::client::find_closest_index(&embedding_answer);
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
    let query_data = query_cipher_txt.data.clone();
    let text_answer = tiptoe_rs::client::query_text(base_url, query_data).await?;
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
    println!("text_vector: {:?}", text_vector);
    let encoded = EncodedString(text_vector.data);
    let text: String = encoded.into();
    println!("\nResult: {}", text);
    
    Ok(())
} 