use axum::{
    routing::get,
    Router,
    Json,
    extract::State,
};
use std::{net::SocketAddr, sync::Mutex, collections::HashMap};
use serde_json::json;
use tokio::signal;
use tiptoe_rs::market_data::{get_market_prices, format_prices};

#[derive(Clone)]
struct AppState {
    last_update: std::sync::Arc<Mutex<String>>,
    stock_prices: std::sync::Arc<Mutex<HashMap<String, f64>>>,
    crypto_prices: std::sync::Arc<Mutex<HashMap<String, f64>>>,
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

async fn update_market_data(state: &AppState) {
    match get_market_prices().await {
        Ok((stocks, cryptos, timestamp)) => {
            *state.last_update.lock().unwrap() = timestamp.clone();
            *state.stock_prices.lock().unwrap() = stocks;
            *state.crypto_prices.lock().unwrap() = cryptos;
            
            // Print the saved market data
            let formatted = format_prices(
                state.stock_prices.lock().unwrap().clone(),
                state.crypto_prices.lock().unwrap().clone(),
                state.last_update.lock().unwrap().clone()
            );
            println!("\nMarket data updated at {}:\n{}", timestamp, formatted);
        }
        Err(e) => {
            eprintln!("Failed to update market data: {}", e);
        }
    }
}

#[tokio::main]
async fn main() {
    let state = AppState {
        last_update: std::sync::Arc::new(Mutex::new(String::new())),
        stock_prices: std::sync::Arc::new(Mutex::new(HashMap::new())),
        crypto_prices: std::sync::Arc::new(Mutex::new(HashMap::new())),
    };

    // Initial market data update
    update_market_data(&state).await;

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("Starting server at http://{}", addr);
    println!("Market data will update every minute");

    // Spawn background task for periodic updates
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            update_market_data(&state_clone).await;
        }
    });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/market-data", get(get_market_data))
        .with_state(state);

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
