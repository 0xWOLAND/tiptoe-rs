use anyhow::Result;
use chrono::{DateTime, Local};
use chrono_tz::US::Eastern;
use serde::Serialize;
use std::collections::HashMap;
use rand::prelude::*;

#[derive(Debug, Serialize)]
pub struct MarketData {
    timestamp: String,
    stocks: HashMap<String, String>,
    cryptocurrencies: HashMap<String, String>,
}

async fn fetch_market_data() -> Result<(HashMap<String, f64>, HashMap<String, f64>, String)> {
    let mut stocks = HashMap::new();
    let mut cryptos = HashMap::new();
    let mut rng = rand::thread_rng();

    // Mock stock prices with random variations
    let base_prices = [
        ("Apple Inc.", 190.0),
        ("NVIDIA Corporation", 480.0),
        ("Microsoft Corporation", 370.0),
        ("Amazon.com, Inc.", 145.0),
        ("Alphabet Inc.", 135.0),
        ("Meta Platforms, Inc.", 345.0),
        ("Tesla, Inc.", 240.0),
    ];

    for (name, base_price) in base_prices {
        let variation = rng.gen_range(-5.0..5.0);
        stocks.insert(name.to_string(), base_price + variation);
    }

    // Mock crypto prices with random variations
    let base_crypto_prices = [
        ("Bitcoin (BTC)", 95595.0),
        ("Ethereum (ETH)", 3410.0),
        ("Solana (SOL)", 204.0),
    ];

    for (name, base_price) in base_crypto_prices {
        let percent_change = rng.gen_range(-2.0..2.0) / 100.0; // -2% to +2% change
        let change = base_price * percent_change;
        cryptos.insert(name.to_string(), base_price + change);
    }

    let current_time: DateTime<Local> = Local::now();
    let est_time = current_time.with_timezone(&Eastern);
    let timestamp = est_time.format("%Y-%m-%d %I:%M:%S %p %Z").to_string();

    Ok((stocks, cryptos, timestamp))
}

pub async fn get_market_prices() -> Result<(HashMap<String, f64>, HashMap<String, f64>, String)> {
    fetch_market_data().await
}

pub fn format_prices(
    stock_prices: HashMap<String, f64>,
    crypto_prices: HashMap<String, f64>,
    timestamp: String,
) -> String {
    let mut output = MarketData {
        timestamp,
        stocks: HashMap::new(),
        cryptocurrencies: HashMap::new(),
    };

    for (name, price) in stock_prices {
        output
            .stocks
            .insert(name, format!("${:.2}", price));
    }

    for (name, price) in crypto_prices {
        output
            .cryptocurrencies
            .insert(name, format!("${:.2}", price));
    }

    serde_json::to_string_pretty(&output).unwrap_or_else(|_| "Error formatting prices".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_prices() {
        let mut stock_prices = HashMap::new();
        stock_prices.insert("Apple Inc.".to_string(), 150.50);

        let mut crypto_prices = HashMap::new();
        crypto_prices.insert("Bitcoin (BTC)".to_string(), 50000.00);

        let timestamp = "2024-01-01 12:00:00 PM EST".to_string();

        let formatted = format_prices(stock_prices, crypto_prices, timestamp);
        assert!(formatted.contains("Apple Inc."));
        assert!(formatted.contains("Bitcoin (BTC)"));
        assert!(formatted.contains("$150.50"));
        assert!(formatted.contains("$50000.00"));
    }

    #[tokio::test]
    async fn test_live_market_data() {
        if let Ok((stocks, cryptos, timestamp)) = get_market_prices().await {
            let formatted = format_prices(stocks, cryptos, timestamp);
            println!("Live Market Data:\n{}", formatted);
        }
    }
} 