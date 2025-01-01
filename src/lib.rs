pub mod embeddings;
pub mod utils;
pub mod encoding;

#[cfg(test)]
mod tests {
    use embeddings::TextEmbedder;
    use simplepir::{answer, query, recover, recover_row, setup, Database};
    use utils::strings_to_embedding_matrix;

    use super::*;

    #[test]
    pub fn test_e2e() {
        let embedder = TextEmbedder::new().unwrap();
        let texts = vec![
            "This is the first text".to_string(),
            "This is the second text".to_string(), 
            "And this is the third one".to_string(),
        ];
        
        for text in &texts {
            let embedding = embedder.embed(text).unwrap();
            let scaled_embedding = embedding.as_ref().to_vec2::<f32>().unwrap()[0].iter().map(|&x| {
                const SCALE_FACTOR: f32 = 1_00.0;
                let scaled = (x * SCALE_FACTOR).round() as u64;
                scaled
            }).collect::<Vec<_>>();
            println!("Scaled embedding for '{}': {:?}", text, scaled_embedding);
        }


        let secret_dimension = 10000;
        let mod_power = 17;
        let plain_mod = 2_u64.pow(mod_power as u32);

        let embedding_matrix = strings_to_embedding_matrix(&texts, &embedder).unwrap();

        println!("embedding_matrix: {:?}", embedding_matrix.data);

        let db = Database::from_matrix(embedding_matrix, mod_power).unwrap();
        let compressed_db = db.compress().unwrap();
        let (server_hint, client_hint) = setup(&db, secret_dimension);
        let db_side_len = db.side_len();

        for index in 0..3 {
            let (client_state, query_cipher) = query(index, db_side_len, secret_dimension, server_hint, plain_mod);
            let answer_cipher = answer(&compressed_db, &query_cipher);
            let record = recover_row(&client_state, &client_hint, &answer_cipher, &query_cipher, plain_mod);
            println!("index: {}, record: {:?}", index, record.data);
        }

        

    }
}