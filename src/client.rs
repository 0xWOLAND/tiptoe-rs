use std::cmp::Ordering;
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use num_traits::One;
use simplepir::{generate_query, recover, SimplePIRParams};

use crate::{
    embedding::BertEmbedder,
    network::{AsyncDatabase, RemoteDatabase}, server::{Database, EmbeddingDatabase, EncodingDatabase},
};

// Each database can be either local or remote
pub enum DatabaseConnection<T> {
    Local(T),
    Remote(Box<dyn AsyncDatabase>),
}

impl<T: Database> DatabaseConnection<T> {
    async fn update(&mut self) -> Result<()> {
        match self {
            Self::Local(db) => db.update(),
            Self::Remote(db) => db.update().await,
        }
    }

    async fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        match self {
            Self::Local(db) => db.respond(query),
            Self::Remote(db) => db.respond(query).await,
        }
    }

    async fn params(&self) -> Result<SimplePIRParams> {
        match self {
            Self::Local(db) => Ok(db.params().clone()),
            Self::Remote(db) => db.get_params().await,
        }
    }

    async fn hint(&self) -> Result<DMatrix<BigInt>> {
        match self {
            Self::Local(db) => Ok(db.hint().clone()),
            Self::Remote(db) => db.get_hint().await,
        }
    }

    async fn a(&self) -> Result<DMatrix<BigInt>> {
        match self {
            Self::Local(db) => Ok(db.a().clone()),
            Self::Remote(db) => db.get_a().await,
        }
    }
}

// Unified client that works with both local and remote databases
pub struct Client {
    embedding_db: DatabaseConnection<EmbeddingDatabase>,
    encoding_db: DatabaseConnection<EncodingDatabase>,
    embedder: BertEmbedder,
}

impl Client {
    pub fn new_local() -> Result<Self> {
        Ok(Self {
            embedding_db: DatabaseConnection::Local(EmbeddingDatabase::new()),
            encoding_db: DatabaseConnection::Local(EncodingDatabase::new()),
            embedder: BertEmbedder::new()?,
        })
    }

    pub fn new_remote(embedding_url: String, encoding_url: String) -> Result<Self> {
        Ok(Self {
            embedding_db: DatabaseConnection::Remote(Box::new(RemoteDatabase::new(embedding_url))),
            encoding_db: DatabaseConnection::Remote(Box::new(RemoteDatabase::new(encoding_url))),
            embedder: BertEmbedder::new()?,
        })
    }

    pub async fn update(&mut self) -> Result<()> {
        self.encoding_db.update().await?;
        self.embedding_db.update().await?;
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

    pub async fn query(&self, query: &str) -> Result<DVector<BigInt>> {
        let embedding = self.embedder.embed_text(query)?;
        
        // Query embedding database
        let embedding_params = self.embedding_db.params().await?;
        let adjusted_embedding = Self::adjust_embedding(embedding, embedding_params.m);
        let (s_embedding, query_embedding) = generate_query(
            &embedding_params,
            &adjusted_embedding,
            &self.embedding_db.a().await?
        );
        
        let response_embedding = self.embedding_db.respond(&query_embedding).await?;
        let result_embedding = recover(
            &self.embedding_db.hint().await?,
            &s_embedding,
            &response_embedding,
            &embedding_params
        );
        
        // Convert to one-hot vector
        let result_vec = {
            let mut vec = DVector::zeros(result_embedding.len());
            let max_idx = result_embedding.iter()
                .enumerate()
                .max_by_key(|(_i, val)| val.clone())
                .map(|(i, _val)| i)
                .unwrap();
            vec[max_idx] = BigInt::one();
            vec
        };
        
        // Query encoding database
        let encoding_params = self.encoding_db.params().await?;
        let adjusted_result = Self::adjust_embedding(result_vec, encoding_params.m);
        let (s, query) = generate_query(
            &encoding_params,
            &adjusted_result,
            &self.encoding_db.a().await?
        );
        
        let response = self.encoding_db.respond(&query).await?;
        let result = recover(
            &self.encoding_db.hint().await?,
            &s,
            &response,
            &encoding_params
        );
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::decode_input;

    use super::*;
    use tokio::test;

    async fn run_test_queries(client: &mut Client) -> Result<()> {
        let names = vec![
            "Bitcoin",
            "Ethereum", 
            "SPDR S&P 500",
            "Tesla",
            "NASDAQ Composite"
        ];
        
        for i in 0..3 {
            println!("\nUpdate iteration {}...", i + 1);
            client.update().await?;
            
            for name in &names {
                println!("\nQuerying {}...", name);
                let result = client.query(name).await?;
                println!("Raw result: {:?}", result);
                
                let output = decode_input(&result)?;
                println!("Decoded output: {:?}", output);
            }
        }
        Ok(())
    }

    #[test]
    async fn test_local_client() -> Result<()> {
        let mut client = Client::new_local()?;
        run_test_queries(&mut client).await
    }

    #[test]
    async fn test_remote_client() -> Result<()> {
        let mut client = Client::new_remote(
            "http://localhost:3001".to_string(),
            "http://localhost:3000".to_string()
        )?;
        run_test_queries(&mut client).await
    }
}