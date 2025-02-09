use tiptoe_rs::{network::run_server, server::{EncodingDatabase, Database}};

#[tokio::main]
async fn main() {
    let db = EncodingDatabase::new();
    run_server(db, 3000).await;
}
