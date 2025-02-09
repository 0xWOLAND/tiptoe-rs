// src/network.rs
use async_trait::async_trait;
use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::State,
};
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use num_traits::One;
use serde::{Serialize, Deserialize};
use std::{str::FromStr, sync::Arc, time::Duration};
use tokio::{sync::RwLock, time::interval};
use anyhow::Result;
use reqwest::Client as HttpClient;
use simplepir::{SimplePIRParams, generate_query, recover, gen_params};

use crate::{
    server::Database,
    embedding::BertEmbedder,
};

// Shared state for server
pub struct ServerState<T: Database + Send + Sync> {
    db: RwLock<T>,
}

// Request/Response types
#[derive(Serialize, Deserialize)]
pub struct QueryRequest {
    query: Vec<String>, // Serialized BigInt vector
}

#[derive(Serialize, Deserialize)]
pub struct QueryResponse {
    response: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ParamsData {
    m: usize,
    n: usize,
    q: String,
    p: String,
}

#[derive(Serialize, Deserialize)]
pub struct MatrixResponse {
    rows: usize,
    cols: usize,
    data: Vec<String>,
}

// Helper functions for serialization
fn serialize_vector(vec: &DVector<BigInt>) -> Vec<String> {
    vec.iter().map(|x| x.to_string()).collect()
}

fn deserialize_vector(vec: &[String]) -> DVector<BigInt> {
    let values: Vec<BigInt> = vec.iter()
        .map(|x| x.parse().unwrap())
        .collect();
    DVector::from_vec(values)
}

fn serialize_matrix(matrix: &DMatrix<BigInt>) -> MatrixResponse {
    MatrixResponse {
        rows: matrix.nrows(),
        cols: matrix.ncols(),
        data: matrix.iter().map(|x| x.to_string()).collect(),
    }
}

fn deserialize_matrix(response: &MatrixResponse) -> DMatrix<BigInt> {
    let data: Vec<BigInt> = response.data.iter()
        .map(|x| x.parse().unwrap())
        .collect();
    DMatrix::from_vec(response.rows, response.cols, data)
}

fn serialize_params(params: &SimplePIRParams) -> ParamsData {
    ParamsData {
        m: params.m,
        n: params.n,
        q: params.q.to_string(),
        p: params.p.to_string(),
    }
}

fn deserialize_params(data: &ParamsData) -> SimplePIRParams {
    let p = BigInt::from_str(&data.p).unwrap();
    let mod_power = (p.bits() - 1) as u32;
    gen_params(data.m, data.n, mod_power)
}

// Server setup and handlers
pub async fn run_server<T: Database + Send + Sync + 'static>(db: T, port: u16) {
    let state = Arc::new(ServerState {
        db: RwLock::new(db),
    });
    
    let update_state = Arc::clone(&state);
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(5)); // Update every minute
        loop {
            interval.tick().await;
            println!("Updating database...");
            let mut db = update_state.db.write().await;
                if let Err(e) = db.update() {
                    eprintln!("Error updating database: {:?}", e);
                }
            if let Err(e) = db.update() {
                eprintln!("Error updating database: {:?}", e);
            }
        }
    });

    let app = Router::new()
        .route("/query", post(handle_query::<T>))
        .route("/params", get(handle_params::<T>))
        .route("/hint", get(handle_hint::<T>))
        .route("/a", get(handle_a::<T>))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port).parse().unwrap();
    println!("Starting server on {}", addr);
    
    axum_server::bind(addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}


async fn handle_query<T: Database + Send + Sync>(
    State(state): State<Arc<ServerState<T>>>,
    Json(request): Json<QueryRequest>,
) -> Json<QueryResponse> {
    let query = deserialize_vector(&request.query);
    let db = state.db.read().await;
    let response = db.respond(&query).unwrap();
    Json(QueryResponse {
        response: serialize_vector(&response),
    })
}

async fn handle_update<T: Database + Send + Sync>(
    State(state): State<Arc<ServerState<T>>>,
) -> &'static str {
    let mut db = state.db.write().await;
    db.update().unwrap();
    "Database updated successfully"
}

async fn handle_params<T: Database + Send + Sync>(
    State(state): State<Arc<ServerState<T>>>,
) -> Json<ParamsData> {
    let db = state.db.read().await;
    Json(serialize_params(db.params()))
}

async fn handle_hint<T: Database + Send + Sync>(
    State(state): State<Arc<ServerState<T>>>,
) -> Json<MatrixResponse> {
    let db = state.db.read().await;
    Json(serialize_matrix(db.hint()))
}

async fn handle_a<T: Database + Send + Sync>(
    State(state): State<Arc<ServerState<T>>>,
) -> Json<MatrixResponse> {
    let db = state.db.read().await;
    Json(serialize_matrix(db.a()))
}

// Remote database implementation that connects to server
#[async_trait]
pub trait AsyncDatabase {
    async fn update(&mut self) -> Result<()>;
    async fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>>;
    async fn get_params(&self) -> Result<SimplePIRParams>;
    async fn get_hint(&self) -> Result<DMatrix<BigInt>>;
    async fn get_a(&self) -> Result<DMatrix<BigInt>>;
}

pub struct RemoteDatabase {
    client: HttpClient,
    base_url: String,
}

impl RemoteDatabase {
    pub fn new(base_url: String) -> Self {
        Self {
            client: HttpClient::builder()
                .build()
                .unwrap(),
            base_url,
        }
    }
}

#[async_trait]
impl AsyncDatabase for RemoteDatabase {
    async fn update(&mut self) -> Result<()> {
        self.client.post(&format!("{}/update", self.base_url))
            .send()
            .await?;
        Ok(())
    }

    async fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        let response: QueryResponse = self.client.post(&format!("{}/query", self.base_url))
            .json(&QueryRequest {
                query: serialize_vector(query),
            })
            .send()
            .await?
            .json()
            .await?;
        
        Ok(deserialize_vector(&response.response))
    }

    async fn get_params(&self) -> Result<SimplePIRParams> {
        let response: ParamsData = self.client.get(&format!("{}/params", self.base_url))
            .send()
            .await?
            .json()
            .await?;
        Ok(deserialize_params(&response))
    }

    async fn get_hint(&self) -> Result<DMatrix<BigInt>> {
        let response: MatrixResponse = self.client.get(&format!("{}/hint", self.base_url))
            .send()
            .await?
            .json()
            .await?;
        Ok(deserialize_matrix(&response))
    }

    async fn get_a(&self) -> Result<DMatrix<BigInt>> {
        let response: MatrixResponse = self.client.get(&format!("{}/a", self.base_url))
            .send()
            .await?
            .json()
            .await?;
        Ok(deserialize_matrix(&response))
    }
}

// Client that can work with both local and remote databases
pub struct NetworkClient {
    embedder: BertEmbedder,
    embedding_db: RemoteDatabase,
    encoding_db: RemoteDatabase,
}

impl NetworkClient {
    pub fn new(embedding_url: String, encoding_url: String) -> Result<Self> {
        Ok(Self {
            embedder: BertEmbedder::new()?,
            embedding_db: RemoteDatabase::new(embedding_url),
            encoding_db: RemoteDatabase::new(encoding_url),
        })
    }

    pub async fn update(&mut self) -> Result<()> {
        self.encoding_db.update().await?;
        self.embedding_db.update().await?;
        Ok(())
    }

    fn adjust_embedding(embedding: DVector<BigInt>, m: usize) -> DVector<BigInt> {
        match embedding.len().cmp(&m) {
            std::cmp::Ordering::Equal => embedding,
            std::cmp::Ordering::Less => {
                let mut new_embedding = DVector::zeros(m);
                new_embedding.rows_mut(0, embedding.len()).copy_from(&embedding);
                new_embedding
            },
            std::cmp::Ordering::Greater => {
                embedding.rows(0, m).into()
            }
        }
    }

    pub async fn query(&self, query: &str) -> Result<DVector<BigInt>> {
        let embedding = self.embedder.embed_text(query)?;
        
        let embedding_params = self.embedding_db.get_params().await?;
        let adjusted_embedding = Self::adjust_embedding(embedding, embedding_params.m);
        let (s_embedding, query_embedding) = generate_query(
            &embedding_params,
            &adjusted_embedding,
            &self.embedding_db.get_a().await?
        );
        
        let response_embedding = self.embedding_db.respond(&query_embedding).await?;
        let result_embedding = recover(
            &self.embedding_db.get_hint().await?,
            &s_embedding,
            &response_embedding,
            &embedding_params
        );
        
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
        
        let encoding_params = self.encoding_db.get_params().await?;
        let adjusted_result = Self::adjust_embedding(result_vec, encoding_params.m);
        let (s, query) = generate_query(
            &encoding_params,
            &adjusted_result,
            &self.encoding_db.get_a().await?
        );
        
        let response = self.encoding_db.respond(&query).await?;
        let result = recover(
            &self.encoding_db.get_hint().await?,
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

    #[test]
    async fn test_network_client() -> Result<()> {
        let mut client = NetworkClient::new(
            "http://localhost:3001".to_string(),
            "http://localhost:3000".to_string()
        )?;
        
        let names = vec![
            "Bitcoin USD",
            "Ethereum USD", 
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
}