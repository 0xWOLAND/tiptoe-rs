use std::cmp::Ordering;

use crate::{embedding::BertEmbedder, server::{Database, EmbeddingDatabase, EncodingDatabase}};
use anyhow::Result;
use nalgebra::DVector;
use num_bigint::BigInt;
use num_traits::One;
use simplepir::{generate_query, recover};


pub enum DatabaseType {
    EncodingDatabase,
    EmbeddingDatabase
}

pub struct Client {
    encoding_db: EncodingDatabase,
    embedding_db: EmbeddingDatabase,
    embedder: BertEmbedder
}

impl Client {
    pub fn new() -> Self {
        Self { encoding_db: EncodingDatabase::new(), embedding_db: EmbeddingDatabase::new(), embedder: BertEmbedder::new().unwrap() }
    }

    pub fn update(&mut self) -> Result<()> {
        self.encoding_db.update()?;
        self.embedding_db.update()?;

        Ok(())
    }

    fn adjust_embedding(embedding: DVector<BigInt>, m: usize) -> DVector<BigInt> {
        match embedding.len().cmp(&m) {
            Ordering::Equal => embedding,
            Ordering::Less => {
                let mut new_embedding = DVector::zeros(m);
                new_embedding.rows_mut(0, embedding.len()).copy_from(&embedding);
                new_embedding
            },
            Ordering::Greater => {
                embedding.rows(0, m).into()
            }
        }
    }

    pub fn query(&self, query: &str) -> Result<DVector<BigInt>> {
        let start = std::time::Instant::now();
        
        let embedding = self.embedder.encode_text(query)?;
        let m_embedding = self.embedding_db.params().m;
        let m_encoding = self.encoding_db.params().m;
        
        let embedding_start = std::time::Instant::now();
        let (s_embedding, query_embedding) = generate_query(self.embedding_db.params(), &Self::adjust_embedding(embedding, m_embedding), self.embedding_db.a());
        let response_embedding = self.embedding_db.respond(&query_embedding)?;
        let result_embedding: DVector<BigInt> = recover(self.embedding_db.hint(), &s_embedding, &response_embedding, self.embedding_db.params());
        println!("result_embedding: {:?}", result_embedding);
        let embedding_duration = embedding_start.elapsed();
        
        let vec_start = std::time::Instant::now();
        let result_vec = {
            let mut vec = DVector::zeros(result_embedding.len());
            let max_idx = result_embedding.iter()
                .enumerate()
                .max_by_key(|(_i, val)| val.clone())
                .map(|(i, _val)| i)
                .unwrap();
            println!("max_idx: {:?}", max_idx);
            vec[max_idx] = BigInt::one();
            vec
        };
        let vec_duration = vec_start.elapsed();
        
        let encoding_start = std::time::Instant::now();
        let (s, query) = generate_query(self.encoding_db.params(), &Self::adjust_embedding(result_vec, m_encoding), self.encoding_db.a());
        let response = self.encoding_db.respond(&query)?;
        let result = recover(self.encoding_db.hint(), &s, &response, self.encoding_db.params());
        let encoding_duration = encoding_start.elapsed();
        
        let total_duration = start.elapsed();
        
        println!("Timing breakdown:");
        println!("  Embedding phase: {:?}", embedding_duration);
        println!("  Vector processing: {:?}", vec_duration);
        println!("  Encoding phase: {:?}", encoding_duration);
        println!("  Total time: {:?}", total_duration);
        
        Ok(result)
    }
}

#[test]
fn test_client() {
    println!("Testing client...");
    let mut client = Client::new();
    client.update().unwrap();
    println!("querying...");
    let _result = client.query("Pacer Data and Digital Revoluti").unwrap();

}