use axum::{
    routing::get,
    Router,
    Json,
    extract::State,
};
use std::{net::SocketAddr, sync::Mutex, collections::HashMap};
use serde_json::json;
use tokio::signal;
use tiptoe_rs::{
    market_data::{get_market_prices, format_prices},
    embeddings::TextEmbedder,
    utils::strings_to_embedding_matrix,
    encoding::{StringMatrix, EncodedString},
};
use simplepir::{Database, Matrix, setup, query, answer, recover_row};

// Modulus must be less than 2^21 for compression to work
const MOD_POWER: u32 = 17;
const SECRET_DIMENSION: usize = 2048;
const PLAIN_MOD: u64 = 2_u64.pow(MOD_POWER);

#[derive(Clone)]
struct DatabaseState {
    embedding_db: Database,
    text_db: Database,
    server_hints: (u64, u64),
    client_hints: (Matrix, Matrix),
}

#[derive(Clone)]
struct AppState {
    last_update: std::sync::Arc<Mutex<String>>,
    stock_prices: std::sync::Arc<Mutex<HashMap<String, f64>>>,
    crypto_prices: std::sync::Arc<Mutex<HashMap<String, f64>>>,
    db_state: std::sync::Arc<Mutex<Option<DatabaseState>>>,
}

fn generate_market_texts(stocks: &HashMap<String, f64>, cryptos: &HashMap<String, f64>) -> Vec<String> {
    let mut texts = Vec::new();
    
    // Generate texts for stocks
    for (name, price) in stocks {
        texts.push(format!("{} is currently trading at ${:.2}", name, price));
        texts.push(format!("The stock price of {} is ${:.2} per share", name, price));
        texts.push(format!("Latest market update: {} shares are valued at ${:.2}", name, price));
    }
    
    // Generate texts for cryptocurrencies
    for (name, price) in cryptos {
        texts.push(format!("{} is currently valued at ${:.2}", name, price));
        texts.push(format!("The cryptocurrency {} is trading at ${:.2}", name, price));
        texts.push(format!("Latest crypto update: {} price is ${:.2}", name, price));
    }
    
    texts
}

async fn update_market_data(state: &AppState) {
    println!("\nFetching latest market data...");
    match get_market_prices().await {
        Ok((stocks, cryptos, timestamp)) => {
            println!("✓ Market data fetched successfully");
            
            // Update market data
            *state.last_update.lock().unwrap() = timestamp.clone();
            *state.stock_prices.lock().unwrap() = stocks.clone();
            *state.crypto_prices.lock().unwrap() = cryptos.clone();
            
            // Generate texts and update databases
            println!("Generating market texts...");
            let texts = generate_market_texts(&stocks, &cryptos);
            println!("✓ Generated {} text entries", texts.len());
            
            // Create embedding database
            println!("Initializing text embedder...");
            let embedder = TextEmbedder::new().unwrap();
            println!("✓ Text embedder initialized");
            
            println!("Creating embedding database...");
            let embedding_matrix = strings_to_embedding_matrix(&texts, &embedder).unwrap();
            let embedding_db = Database::from_matrix(embedding_matrix, MOD_POWER as u8).unwrap();
            let (server_hint_emb, client_hint_emb) = setup(&embedding_db, SECRET_DIMENSION);
            println!("✓ Embedding database created");
            
            // Create encoded text database
            println!("Creating text database...");
            let encoded = StringMatrix::new(&texts);
            let text_db = Database::from_matrix(encoded.data, MOD_POWER as u8).unwrap();
            let (server_hint_txt, client_hint_txt) = setup(&text_db, SECRET_DIMENSION);
            println!("✓ Text database created");
            
            // Update the databases in state
            println!("Updating database state...");
            *state.db_state.lock().unwrap() = Some(DatabaseState {
                embedding_db: embedding_db.clone(),
                text_db: text_db.clone(),
                server_hints: (server_hint_emb, server_hint_txt),
                client_hints: (client_hint_emb, client_hint_txt),
            });
            println!("✓ Database state updated");
            
            // Print the saved market data
            let formatted = format_prices(
                stocks,
                cryptos,
                timestamp.clone()
            );
            println!("\nMarket data updated at {}:\n{}", timestamp, formatted);
            
            // Print a sample query from the databases
            if let Some(ref db_state) = *state.db_state.lock().unwrap() {
                println!("\nTesting database query...");
                let index = 0;
                let db_side_len = db_state.text_db.side_len();
                
                // Query the text database
                let (client_state, query_cipher) = query(index, db_side_len, SECRET_DIMENSION, db_state.server_hints.1, PLAIN_MOD);
                let compressed_db = db_state.text_db.compress().unwrap();
                let answer_cipher = answer(&compressed_db, &query_cipher);
                let record = recover_row(&client_state, &db_state.client_hints.1, &answer_cipher, &query_cipher, PLAIN_MOD);
                let encoded = EncodedString(record.data);
                let decoded: String = encoded.into();
                println!("✓ Database query successful");
                println!("Sample text at index {}: {}", index, decoded);
            }
            println!("\n--- Update complete ---");
        }
        Err(e) => {
            eprintln!("❌ Failed to update market data: {}", e);
        }
    }
}

async fn health_check() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "timestamp": chrono::Local::now().to_rfc3339()
    }))
}

async fn get_market_data(State(state): State<AppState>) -> Json<serde_json::Value> {
    let timestamp = state.last_update.lock().unwrap().clone();
    let stocks = state.stock_prices.lock().unwrap().clone();
    let cryptos = state.crypto_prices.lock().unwrap().clone();
    
    let formatted = format_prices(stocks, cryptos, timestamp);
    Json(serde_json::from_str(&formatted).unwrap())
}

#[tokio::main]
async fn main() {
    println!("\n=== Starting Tiptoe Server ===");
    println!("Initializing state...");
    let state = AppState {
        last_update: std::sync::Arc::new(Mutex::new(String::new())),
        stock_prices: std::sync::Arc::new(Mutex::new(HashMap::new())),
        crypto_prices: std::sync::Arc::new(Mutex::new(HashMap::new())),
        db_state: std::sync::Arc::new(Mutex::new(None)),
    };
    println!("✓ State initialized");

    println!("\nPerforming initial market data update...");
    update_market_data(&state).await;

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("\n=== Server Configuration ===");
    println!("Address: http://{}", addr);
    println!("Update interval: 60 seconds");
    println!("MOD_POWER: {}", MOD_POWER);
    println!("SECRET_DIMENSION: {}", SECRET_DIMENSION);
    println!("PLAIN_MOD: {}", PLAIN_MOD);

    // Spawn background task for periodic updates
    println!("\nStarting background update task...");
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        println!("✓ Background task started");
        loop {
            interval.tick().await;
            update_market_data(&state_clone).await;
        }
    });

    println!("\nConfiguring routes...");
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/market-data", get(get_market_data))
        .with_state(state);
    println!("✓ Routes configured");

    println!("\n=== Server Starting ===");
    println!("Binding to address...");
    axum::serve(
        tokio::net::TcpListener::bind(addr).await.unwrap(),
        app.into_make_service(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .unwrap();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    println!("shutdown signal received")
}
