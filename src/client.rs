use std::cmp::Ordering;

use crate::{embedding::BertEmbedder, server::{Database, EmbeddingDatabase, EncodingDatabase}, utils::{decode_data, decode_input}};
use anyhow::Result;
use nalgebra::DVector;
use num_bigint::BigInt;
use num_traits::One;
use rand::seq::IndexedRandom;
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
        let embedding = self.embedder.embed_text(query)?;
        let m_embedding = self.embedding_db.params().m;
        let m_encoding = self.encoding_db.params().m;
        
        let (s_embedding, query_embedding) = generate_query(self.embedding_db.params(), &Self::adjust_embedding(embedding, m_embedding), self.embedding_db.a());
        let response_embedding = self.embedding_db.respond(&query_embedding)?;
        let result_embedding: DVector<BigInt> = recover(self.embedding_db.hint(), &s_embedding, &response_embedding, self.embedding_db.params());
        
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
        
        let (s, query) = generate_query(self.encoding_db.params(), &Self::adjust_embedding(result_vec, m_encoding), self.encoding_db.a());
        let response = self.encoding_db.respond(&query)?;
        let result = recover(self.encoding_db.hint(), &s, &response, self.encoding_db.params());
        
        Ok(result)
    }
}

#[test]
fn test_client() {
    println!("Testing client...");
    let mut client = Client::new();
    
    // Define names to query
    let names = vec![
        "Bitcoin USD",
        "Ethereum USD", 
        "SPDR S&P 500",
        "Tesla",
        "NASDAQ Composite"
    ];
    
    // Run multiple updates
    for i in 0..3 {
        println!("\nUpdate iteration {}...", i + 1);
        client.update().unwrap();
        
        // Query each name
        for name in &names {
            println!("\nQuerying {}...", name);
            let result: DVector<BigInt> = client.query(name).unwrap();
            println!("Raw result: {:?}", result);
            
            let output = decode_input(&result).unwrap();
            println!("Decoded output: {:?}", output);
        }
        
    }
}

#[test]
fn test_client_query() {
    use std::collections::HashMap;
    use rand::seq::SliceRandom;
    use serde_json::Value;

    println!("Testing client query acceptance rate...");
    let mut client = Client::new();

    // Define base queries
    let symbols: HashMap<&str, &str> = [
        ("A", "Agilent"),
        ("APPL", "Apple"),
        ("GM", "General Motors Company"),
        ("MU", "Micron Technology"),
        ("TSLA", "Tesla"),
        ("ALI=F", "Aluminum Futures,Apr-2025"),
        ("CD=F", "Canadian Dollar Dec 20"),
        ("QM=F", "E-mini Crude Oil Futures,Mar-20"),
        ("^IXIC", "NASDAQ Composite"),
        ("EURUSD=X", "EUR/USD"),
        ("AUDUSD=X", "AUD/USD"),
        ("^DJT", "Dow Jones Transportation Average"),
        ("^HSI", "HANG SENG INDEX"),
        ("^VIX", "CBOE Volatility Index"),
        ("^TRFK-TC", "Pacer Data and Digital Revolution"),
        ("SPY", "SPDR S&P 500"),
        ("AWSHX", "Washington Mutual Invs Fd Cl A"),
        ("VOO", "Vanguard S&P 500 ETF"),
        ("XAIX.BE", "Xtr.(IE)-Art.Int.+Big Data ETFR"),
        ("BTC-USD", "Bitcoin USD"),
        ("ETH-USD", "Ethereum USD"),
    ]
    .iter()
    .copied()
    .collect();

    let query_templates = vec![
        "Tell me about {name}",
        "What is the latest price of {name}?",
        "How is {name} performing today?",
        "Give me details on {name}",
        "Fetch data for {name}",
        "What's happening with {name}?",
    ];

    let mut success_count = 0;
    let mut error_count = 0;

    let mut rng = rand::thread_rng();

    // Run multiple updates and queries
    for i in 0..3 {
        println!("\nUpdate iteration {}...", i + 1);
        client.update().unwrap();

        for (symbol, name) in symbols.iter() {
            let template = query_templates.choose(&mut rng).unwrap();
            let query = template.replace("{name}", name);

            println!("\nQuerying: {}", query);

            match client.query(&query) {
                Ok(result) => {
                    println!("Raw result: {:?}", result);
                    let output = decode_input(&result);

                    match output {
                        Ok(output) => {
                            println!("Decoded output: {:?}", output);
                            
                            let json_output: Value = serde_json::from_str(&output).unwrap_or_else(|_| Value::Null);
                            
                            let received_name = json_output["name"].as_str().unwrap_or("").trim();

                            if received_name == name.trim(){
                                success_count += 1;
                            } else {
                                error_count += 1;
                                println!(
                                    "Data mismatch: Expected ({}), but got ({})",
                                    name, received_name
                                );
                            }
                        },
                        Err(e) => {
                            error_count += 1;
                            println!("Decoding failed: {:?}", e);
                        }
                    }
                },
                Err(e) => {
                    error_count += 1;
                    println!("Query failed: {:?}", e)
                }
            }

            let total_attempts = success_count + error_count;
            let acceptance_rate = if total_attempts > 0 {
                (success_count as f64 / total_attempts as f64) * 100.0
            } else {
                0.0
            };
            println!("Current acceptance rate: {:.2}%", acceptance_rate);
        }
    }
}