use std::process::Command;

use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use serde_json::Value;
use simplepir::*;

use crate::{embedding::BertEmbedder, utils::encode_data};

pub trait Database {
    fn new() -> Self;
    fn update(&mut self) -> Result<()>;
    fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>>;
    fn params(&self) -> &SimplePIRParams;
    fn hint(&self) -> &DMatrix<BigInt>;
    fn a(&self) -> &DMatrix<BigInt>;
}

pub struct SimplePirDatabase {
    params: Option<SimplePIRParams>,
    data: DMatrix<BigInt>,
    hint: Option<DMatrix<BigInt>>,
    a: Option<DMatrix<BigInt>>
}

impl SimplePirDatabase {
    pub fn new(data: DMatrix<BigInt>) -> Self {
        Self { data, params: None, hint: None, a: None }
    }

    pub fn update_db(&mut self, data: DMatrix<BigInt>) -> Result<()> {
        self.data = data;

        let params = gen_params(self.data.nrows(), self.data.ncols(), 64);
        let (hint, a) = gen_hint(&params, &self.data);

        self.params = Some(params);
        self.hint = Some(hint);
        self.a = Some(a);

        Ok(())
    }

    pub fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        let params = self.params.as_ref().unwrap();
        let answer = process_query(&self.data, query, params.q);

        Ok(answer)
    }

    fn params(&self) -> &SimplePIRParams {
        self.params.as_ref().unwrap()
    }

    fn hint(&self) -> &DMatrix<BigInt> {
        self.hint.as_ref().unwrap()
    }

    fn a(&self) -> &DMatrix<BigInt> {
        self.a.as_ref().unwrap()
    }
}

pub struct EmbeddingDatabase {
    db: SimplePirDatabase,
    embedder: BertEmbedder
}

impl Database for EmbeddingDatabase {
    fn new() -> Self {
        Self { db: SimplePirDatabase::new(DMatrix::zeros(1, 1)), embedder: BertEmbedder::new().unwrap() }
    }

    fn update(&mut self) -> Result<()> {
        let stock_json = Command::new("python").arg("src/python/stocks.py").output().unwrap();

        if !stock_json.status.success() {
            return Err(anyhow::anyhow!("Failed to update database"));
        }

        let stock_json = String::from_utf8(stock_json.stdout).unwrap();
        let stock_json: Vec<Value> = serde_json::from_str(&stock_json)?;

        println!("stock_json: {:?}", stock_json);

        let embeddings = self.embedder.embed_json_array(&stock_json)?;
        assert_eq!(embeddings.nrows(), embeddings.ncols());

        self.db.update_db(embeddings)?;

        Ok(())
    }

    fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        self.db.respond(query)
    }

    fn params(&self) -> &SimplePIRParams {
        self.db.params()
    }

    fn hint(&self) -> &DMatrix<BigInt> {
        self.db.hint()
    }

    fn a(&self) -> &DMatrix<BigInt> {
        self.db.a()
    }
}

pub struct EncodingDatabase {
    db: SimplePirDatabase,
}

impl Database for EncodingDatabase {
    fn new() -> Self {
        Self { db: SimplePirDatabase::new(DMatrix::zeros(1, 1)) }
    }

    fn update(&mut self) -> Result<()> {
        let stock_json = Command::new("python").arg("src/python/stocks.py").output().unwrap();

        if !stock_json.status.success() {
            return Err(anyhow::anyhow!("Failed to update database"));
        }

        let stock_json = String::from_utf8(stock_json.stdout).unwrap();
        let stock_json: Vec<Value> = serde_json::from_str(&stock_json)?;

        let encodings = encode_data(&stock_json.iter().map(|v| v.to_string()).collect::<Vec<String>>())?;
        assert_eq!(encodings.nrows(), encodings.ncols());

        self.db.update_db(encodings.transpose())?;

        Ok(())
    }

    fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        self.db.respond(query)
    }

    fn params(&self) -> &SimplePIRParams {
        self.db.params()
    }

    fn hint(&self) -> &DMatrix<BigInt> {
        self.db.hint()
    }

    fn a(&self) -> &DMatrix<BigInt> {
        self.db.a()
    }
}