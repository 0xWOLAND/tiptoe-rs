pub mod embeddings;
pub mod utils;
pub mod encoding;
pub mod market_data;
pub mod clustering;
pub mod client;

pub const SCALE_FACTOR: f32 = 1_000_000.0;

#[cfg(test)]
mod tests {
    use embeddings::TextEmbedder;
    use encoding::{EncodedString, StringMatrix};
    use simplepir::{answer, query, recover_row, setup, Database};
    use utils::strings_to_embedding_matrix;

    use super::*;

    const SECRET_DIMENSION: usize = 10000;
    const MOD_POWER: u8 = 17;
    const PLAIN_MOD: u64 = 2_u64.pow(MOD_POWER as u32);

    #[test]
    pub fn test_e2e_embeddings() {
        let embedder = TextEmbedder::new().unwrap();
        let texts = vec![
            "This is the first text".to_string(),
            "This is the second text".to_string(), 
            "And this is the third one".to_string(),
        ];
        
        for text in &texts {
            let embedding = embedder.embed(text).unwrap();
            let scaled_embedding = embedding.as_ref().to_vec2::<f32>().unwrap()[0].iter().map(|&x| {
                (x * SCALE_FACTOR).round() as u64
            }).collect::<Vec<_>>();
            println!("Scaled embedding for '{}': {:?}", text, scaled_embedding);
        }

        let embedding_matrix = strings_to_embedding_matrix(&texts, &embedder).unwrap();

        println!("embedding_matrix: {:?}", embedding_matrix.data);

        let db = Database::from_matrix(embedding_matrix, MOD_POWER).unwrap();
        let compressed_db = db.compress().unwrap();
        let (server_hint, client_hint) = setup(&db, SECRET_DIMENSION);
        let db_side_len = db.side_len();

        for index in 0..3 {
            let (client_state, query_cipher) = query(index, db_side_len, SECRET_DIMENSION, server_hint, PLAIN_MOD);
            let answer_cipher = answer(&compressed_db, &query_cipher);
            let record = recover_row(&client_state, &client_hint, &answer_cipher, &query_cipher, PLAIN_MOD);
            println!("index: {}, record: {:?}", index, record.data);
        }
    }

    #[test]
    pub fn test_e2e_encoding() {
        let texts = vec![
            "This is the first text".to_string(),
            "This is the second text".to_string(), 
            "And this is the third one".to_string(),
        ];

        let encoded = StringMatrix::new(&texts);

        let db = Database::from_matrix(encoded.data, MOD_POWER).unwrap();
        let compressed_db = db.compress().unwrap();
        let (server_hint, client_hint) = setup(&db, SECRET_DIMENSION);
        let db_side_len = db.side_len();

        for index in 0..3 {
            let (client_state, query_cipher) = query(index, db_side_len, SECRET_DIMENSION, server_hint, PLAIN_MOD);
            let answer_cipher = answer(&compressed_db, &query_cipher);
            let record = recover_row(&client_state, &client_hint, &answer_cipher, &query_cipher, PLAIN_MOD);
            println!("record.data: {:?}", record.data);
            let encoded = EncodedString(record.data);
            let decoded: String = encoded.into();
            println!("index: {}, decoded: {}", index, decoded);
        }
    }
}
