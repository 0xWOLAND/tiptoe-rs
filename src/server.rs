use std::process::Command;

use anyhow::Result;
use nalgebra::DMatrix;
use serde_json::Value;
use simplepir::*;

use crate::embedding::BertEmbedder;

pub struct Database {
    embeddings: DMatrix<u64>
}

impl Database {
    pub fn new() -> Self {
        Self { embeddings: DMatrix::zeros(384, 384) }
    }

    pub fn update(&mut self) -> Result<()> {
        let embedder = BertEmbedder::new().unwrap();
        let stock_json = Command::new("python").arg("src/python/stocks.py").output().unwrap();

        if !stock_json.status.success() {
            return Err(anyhow::anyhow!("Failed to update database"));
        }

        let stock_json = String::from_utf8(stock_json.stdout).unwrap();
        let stock_json: Vec<Value> = serde_json::from_str(&stock_json)?;
        let embeddings = embedder.encode_json_array(&stock_json)?;
        self.embeddings = embeddings;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update() {
        let mut db = Database::new();
        db.update().unwrap();
        println!("{:?}", db.embeddings);
    }
}