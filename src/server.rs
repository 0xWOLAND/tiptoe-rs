use std::process::Command;

use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use serde_json::Value;
use simplepir::*;

use crate::{embedding::BertEmbedder, utils::encode_data};

pub struct Database {
    embeddings: DMatrix<u64>,
    params: Option<SimplePIRParams>,
    encoded_data: DMatrix<u64>,
    hint: Option<DMatrix<u64>>,
    a: Option<DMatrix<u64>>
}

impl Database {
    pub fn new() -> Self {
        Self { embeddings: DMatrix::zeros(0, 0), params: None, encoded_data: DMatrix::zeros(0, 0), hint: None, a: None }
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
        let params = gen_params(embeddings.nrows(), 2048, 17);
        let (hint, a) = gen_hint(&params, &embeddings);

        let encoded_data = encode_data(&stock_json.iter().map(|v| v.to_string()).collect::<Vec<_>>())?;

        self.embeddings = embeddings;
        self.encoded_data = encoded_data;
        self.params = Some(params);
        self.hint = Some(hint);
        self.a = Some(a);
        
        Ok(())
    }

    pub fn response(&self, query: &DVector<u64>) -> Result<DVector<u64>> {
        let params = self.params.as_ref().unwrap();
        let answer = process_query(&self.embeddings, query, params.q);

        Ok(answer)
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