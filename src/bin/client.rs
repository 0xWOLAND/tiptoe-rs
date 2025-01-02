use anyhow::Result;
use simplepir::{regev::{encrypt, gen_secret_key}, query, recover_row, Database, Matrix, Vector};
use tiptoe_rs::{
    client::{find_closest_index, get_db_config, query_embedding, query_text}, embeddings::TextEmbedder, encoding::EncodedString, utils::scale_to_u64
};

#[tokio::main]
async fn main() -> Result<()> {
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
    let prompt = "Microsoft Corporation";
    let embedder = TextEmbedder::new()?;
    let tensor = embedder.embed(&prompt)?;
    let embedded_query = scale_to_u64(tensor)?;
    let query_vector = Vector::from_vec(embedded_query);
    
    // Encrypt the embedding query
    let client_hint_matrix_emb = Matrix::from_data(client_hints.0);
    let secret_key = gen_secret_key(config.secret_dimension, None);
    let query_cipher = encrypt(&secret_key, &client_hint_matrix_emb, &query_vector, config.plain_mod).1;
    
    // Send the encrypted query
    let embedding_answer = query_embedding(base_url, query_cipher.data).await?;
    println!("✓ Got embedding answer");
    
    // Find closest index
    println!("\nFinding closest text...");
    println!("\nScores for each entry:");
    for (i, &score) in embedding_answer.iter().enumerate() {
        println!("[{}] Score: {} ({})", i, score, if score == 0 { "exact match" } else { "different" });
    }
    
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