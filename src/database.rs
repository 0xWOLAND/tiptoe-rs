use anyhow::{Result, anyhow};
use candle_core::Tensor;
use simplepir::{Database, setup, query, answer, answer_uncompressed, recover, Matrix};
use crate::embeddings::TextEmbedder;

pub struct EmbeddingDatabase {
    embedder: TextEmbedder,
    database: Database,
    secret_dimension: usize,
    mod_power: u8,
    plain_mod: u64,
    server_hint: Option<u64>,
    client_hint: Option<Matrix>,
    db_side_len: usize,
}

impl EmbeddingDatabase {
    pub fn new() -> Result<Self> {
        let embedder = TextEmbedder::new()?;
        let secret_dimension = 2048;
        let mod_power = 3;
        let plain_mod = 2_u64.pow(mod_power as u32);
        
        let database = Database::new_random(1, mod_power);
        
        Ok(Self {
            embedder,
            database,
            secret_dimension,
            mod_power,
            plain_mod,
            server_hint: None,
            client_hint: None,
            db_side_len: 1,
        })
    }

    pub fn build_from_strings(&mut self, texts: &[String]) -> Result<()> {
        let embeddings: Result<Vec<_>> = texts.iter()
            .take(9)
            .map(|text| self.embedder.embed(text))
            .collect();
        let embeddings = embeddings?;
        
        let first_embedding = embeddings[0].flatten_all()?;
        let embedding_size = first_embedding.dim(0)?;
        println!("Embedding size: {}", embedding_size);
        
        let mut data = Vec::new();
        for embedding in embeddings {
            let values = embedding
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(|x| ((x.abs() * (self.plain_mod as f32 - 1.0)) as u64) % self.plain_mod)
                .collect::<Vec<_>>();
            data.extend(values);
        }
        
        let total_values = data.len();
        self.db_side_len = (total_values as f32).sqrt().ceil() as usize;
        
        println!("Total values: {}", total_values);
        println!("Calculated side length: {}", self.db_side_len);
        
        let square_size = self.db_side_len * self.db_side_len;
        while data.len() < square_size {
            data.push(0);
        }

        println!("Creating database with {} values", data.len());
        println!("Side length: {}", self.db_side_len);
        println!("First few values: {:?}", data.iter().take(10).collect::<Vec<_>>());
        println!("Last few values: {:?}", data.iter().rev().take(10).collect::<Vec<_>>());

        self.database = Database::from_vector(data, self.mod_power);
        println!("Database created with side length: {}", self.database.side_len());
        
        let (server_hint, client_hint) = setup(&self.database, self.secret_dimension);
        self.server_hint = Some(server_hint);
        self.client_hint = Some(client_hint);
        
        Ok(())
    }

    pub fn query(&self, index: usize) -> Result<Vec<u64>> {
        println!("Querying index {} with side length {}", index, self.db_side_len);
        
        let server_hint = self.server_hint.ok_or_else(|| anyhow!("Database not initialized"))?;
        let client_hint = self.client_hint.as_ref().ok_or_else(|| anyhow!("Database not initialized"))?;
        
        let (client_state, query_cipher) = query(
            index,
            self.db_side_len,
            self.secret_dimension,
            server_hint,
            self.plain_mod,
        );
        
        println!("Query cipher length: {}", query_cipher.len());
        println!("Database dimensions: {}x{}", self.database.side_len(), self.database.side_len());
        
        let answer_cipher = answer_uncompressed(&self.database, &query_cipher);
        
        let record = recover(
            &client_state,
            client_hint,
            &answer_cipher,
            &query_cipher,
            self.plain_mod,
        );
        
        Ok(vec![record])
    }
} 

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_database_operations() -> Result<()> {
        let mut db = EmbeddingDatabase::new()?;

        let texts = vec![
            "Hello world".to_string(),
            "This is a test".to_string(),
            "Another test sentence".to_string(),
            "Fourth test sentence".to_string(),
            "Fifth test sentence".to_string(),
            "Sixth test sentence".to_string(),
            "Seventh test sentence".to_string(),
            "Eighth test sentence".to_string(),
            "Ninth test sentence".to_string(),
        ];

        db.build_from_strings(&texts)?;

        for index in 0..texts.len() {
            let record = db.query(index)?;
            println!("Record for {}: {:?}", texts[index], record);
        }

        Ok(())
    }
}