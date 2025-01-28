use std::cmp::Ordering;

use crate::{embedding::BertEmbedder, server::{Database, EmbeddingDatabase, EncodingDatabase}};
use anyhow::Result;
use nalgebra::DVector;
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

    fn adjust_embedding(embedding: DVector<u64>, m: usize) -> DVector<u64> {
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

    pub fn query(&self, query: &str) -> Result<DVector<u64>> {
        let embedding = self.embedder.encode_text(query)?;

        let m_embedding = self.embedding_db.params().m;
        let m_encoding = self.encoding_db.params().m;

        let (s_embedding, query_embedding) = generate_query(self.embedding_db.params(), &Self::adjust_embedding(embedding, m_embedding), self.embedding_db.a());
        let response_embedding = self.embedding_db.respond(&query_embedding)?;
        let result_embedding: DVector<u64> = recover(self.embedding_db.hint(), &s_embedding, &response_embedding, self.embedding_db.params());

        println!("result_embedding: {:?}", result_embedding);
        
        let (s, query) = generate_query(self.encoding_db.params(), &Self::adjust_embedding(result_embedding, m_encoding), self.encoding_db.a());
        let response = self.encoding_db.respond(&query)?;
        let result = recover(self.encoding_db.hint(), &s, &response, self.encoding_db.params());
        Ok(result)
    }
}

#[test]
fn test_client() {
    let mut client = Client::new();
    client.update().unwrap();
    let result = client.query("What is the stock price of Apple?").unwrap();
    println!("{:?}", result);
}