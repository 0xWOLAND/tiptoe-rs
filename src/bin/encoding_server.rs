use tiptoe_rs::{
    network::run_server,
    server::{Database, EncodingDatabase},
};

#[tokio::main]
async fn main() {
    let db = EncodingDatabase::new();
    run_server(db, 3000).await;
}
