use tiptoe_rs::{
    network::run_server,
    server::{Database, EmbeddingDatabase},
};

#[tokio::main]
async fn main() {
    let db = EmbeddingDatabase::new();
    run_server(db, 3001).await;
}
