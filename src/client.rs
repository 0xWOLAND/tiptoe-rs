use std::cmp::Ordering;

use crate::{embedding::BertEmbedder, server::{Database, EmbeddingDatabase, EncodingDatabase}, utils::{decode_data, decode_input}};
use anyhow::Result;
use nalgebra::DVector;
use num_bigint::BigInt;
use num_traits::One;
use rand::seq::IndexedRandom;
use serde_json::de;
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

    pub fn query_top_k(&self, query: &str, k: usize) -> Result<Vec<DVector<BigInt>>> {
        let embedding = self.embedder.embed_text(query)?;
        let m_embedding = self.embedding_db.params().m;
        let m_encoding = self.encoding_db.params().m;
        
        let (s_embedding, query_embedding) = generate_query(self.embedding_db.params(), &Self::adjust_embedding(embedding, m_embedding), self.embedding_db.a());
        let response_embedding = self.embedding_db.respond(&query_embedding)?;
        let result_embedding: DVector<BigInt> = recover(self.embedding_db.hint(), &s_embedding, &response_embedding, self.embedding_db.params());
        let top_indices: Vec<usize> = {
            let mut indexed_values: Vec<(usize, &BigInt)> = result_embedding.iter()
                .enumerate()
                .collect();
            indexed_values.sort_by(|(_i1, v1), (_i2, v2)| v2.cmp(v1));
            indexed_values.into_iter()
                .map(|(i, _val)| i)
                .collect()
        };
            
        Ok(top_indices[0..k].into_iter().map(|&idx| {
            let mut vec = DVector::zeros(result_embedding.len());
            vec[idx] = BigInt::one();

            let (s, query) = generate_query(self.encoding_db.params(), &Self::adjust_embedding(vec, m_encoding), self.encoding_db.a());
            let response = self.encoding_db.respond(&query).unwrap();
            let result = recover(self.encoding_db.hint(), &s, &response, self.encoding_db.params());

            result
        }).collect())
    }
}

#[test]
fn test_client() {
    println!("Testing client...");
    let mut client = Client::new();
    let k = 3;
    
    let names = vec![
        "Bitcoin USD",
        "Ethereum USD", 
        "SPDR S&P 500",
        "Tesla",
        "NASDAQ Composite"
    ];
    
    for i in 0..3 {
        println!("\nUpdate iteration {}...", i + 1);
        client.update().unwrap();
        
        for name in &names {
            println!("\nQuerying {}...", name);
            let result: Vec<DVector<BigInt>> = client.query_top_k(name, k).unwrap();

            result.iter().enumerate().for_each(|(i, r)| {
                let output = decode_input(&r);
                println!("Decoded {} output: {:?}", i, output);
            });
        }
    }
}

#[test]
fn bench_client_retrieval_accuracy() {
    use std::collections::HashMap;
    use rand::seq::SliceRandom;
    use serde_json::Value;
    use strsim::jaro_winkler;

    fn names_match(name1: &str, name2: &str) -> bool {
        let name1 = name1.trim().to_lowercase();
        let name2 = name2.trim().to_lowercase();
        
        // Exact match
        if name1 == name2 {
            return true;
        }

        // Check similarity using Jaro-Winkler distance
        let similarity = jaro_winkler(&name1, &name2);
        // Threshold of 0.9 means names need to be 90% similar
        similarity > 0.9
    }

    println!("Testing client query acceptance rate for both single and top-k queries...");
    let mut client = Client::new();
    let k = 3;

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

    let mut single_success_count = 0;
    let mut single_error_count = 0;
    let mut topk_success_count = 0;
    let mut topk_error_count = 0;
    let mut rng = rand::thread_rng();

    for i in 0..3 {
        println!("\nUpdate iteration {}...", i + 1);
        client.update().unwrap();

        for (symbol, name) in symbols.iter() {
            let template = query_templates.choose(&mut rng).unwrap();
            let query = template.replace("{name}", name);

            println!("\nQuerying: {}", query);

            // Test single query
            match client.query(&query) {
                Ok(result) => {
                    println!("Single query raw result: {:?}", result);
                    match decode_input(&result) {
                        Ok(output) => {
                            println!("Single query decoded output: {:?}", output);
                            
                            if let Ok(json_output) = serde_json::from_str::<Value>(&output) {
                                let received_name = json_output["name"].as_str().unwrap_or("").trim();
                                
                                if names_match(received_name, name) {
                                    single_success_count += 1;
                                    println!("Single query matched: '{}' with '{}'", received_name, name);
                                } else {
                                    single_error_count += 1;
                                    println!(
                                        "Single query data mismatch: Expected ({}), but got ({})",
                                        name, received_name
                                    );
                                }
                            }
                        },
                        Err(e) => {
                            single_error_count += 1;
                            println!("Single query decoding failed: {:?}", e);
                        }
                    }
                },
                Err(e) => {
                    single_error_count += 1;
                    println!("Single query failed: {:?}", e)
                }
            }

            // Test top-k query
            match client.query_top_k(&query, k) {
                Ok(results) => {
                    println!("Top-k query raw results: {:?}", results);
                    let mut found_match = false;
                    let mut match_position = None;

                    for (idx, result) in results.iter().enumerate() {
                        match decode_input(result) {
                            Ok(output) => {
                                println!("Top-k decoded output {}: {:?}", idx, output);
                                
                                if let Ok(json_output) = serde_json::from_str::<Value>(&output) {
                                    let received_name = json_output["name"].as_str().unwrap_or("").trim();
                                    
                                    if names_match(received_name, name) {
                                        found_match = true;
                                        match_position = Some(idx);
                                        println!("Found match at position {}", idx);
                                        println!("Matched: '{}' with '{}'", received_name, name);
                                        break;
                                    }
                                }
                            },
                            Err(e) => {
                                println!("Top-k decoding failed for result {}: {:?}", idx, e);
                            }
                        }
                    }

                    if found_match {
                        topk_success_count += 1;
                        println!("Successfully found match at position {:?}", match_position);
                    } else {
                        topk_error_count += 1;
                        println!("Expected name {} not found in top {} results", name, k);
                        println!("Top-k results: {:?}", results.iter().map(|r| decode_input(r)).collect::<Vec<_>>());
                    }
                },
                Err(e) => {
                    topk_error_count += 1;
                    println!("Top-k query failed: {:?}", e)
                }
            }

            // Print current stats
            let single_total = single_success_count + single_error_count;
            let topk_total = topk_success_count + topk_error_count;
            
            println!("\nCurrent Statistics:");
            println!("Single Query Acceptance Rate: {:.2}%", 
                if single_total > 0 { (single_success_count as f64 / single_total as f64) * 100.0 } else { 0.0 });
            println!("Top-k Query Acceptance Rate: {:.2}%", 
                if topk_total > 0 { (topk_success_count as f64 / topk_total as f64) * 100.0 } else { 0.0 });
        }
    }

    println!("\nFinal Statistics:");
    println!("Single Query:");
    println!("  Total Attempts: {}", single_success_count + single_error_count);
    println!("  Successes: {}", single_success_count);
    println!("  Errors: {}", single_error_count);
    println!("  Final acceptance rate: {:.2}%", 
        (single_success_count as f64 / (single_success_count + single_error_count) as f64) * 100.0);
    
    println!("\nTop-k Query:");
    println!("  Total Attempts: {}", topk_success_count + topk_error_count);
    println!("  Successes: {}", topk_success_count);
    println!("  Errors: {}", topk_error_count);
    println!("  Final acceptance rate: {:.2}%", 
        (topk_success_count as f64 / (topk_success_count + topk_error_count) as f64) * 100.0);
}