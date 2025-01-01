use anyhow::Result;
use chrono::{DateTime, Local};
use chrono_tz::US::Eastern;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;
use yahoo_finance_api as yahoo;
use tokio;

#[derive(Debug, Serialize)]
pub struct MarketData {
    timestamp: String,
    stocks: HashMap<String, String>,
    cryptocurrencies: HashMap<String, String>,
}

fn get_stock_price(ticker: &str) -> Option<f64> {
    let rt = tokio::runtime::Runtime::new().ok()?;
    let provider = yahoo::YahooConnector::new().ok()?;
    let response = rt.block_on(provider.get_latest_quotes(ticker, "1d")).ok()?;
    response.last_quote().ok().map(|quote| quote.close)
}

#[derive(Deserialize)]
struct CoinGeckoResponse {
    #[serde(flatten)]
    prices: HashMap<String, HashMap<String, f64>>,
}

fn get_crypto_price(coin_id: &str) -> Option<f64> {
    let client = Client::new();
    let url = format!(
        "https://api.coingecko.com/api/v3/simple/price?ids={}&vs_currencies=usd",
        coin_id
    );

    match client.get(&url).send() {
        Ok(response) => {
            if let Ok(data) = response.json::<CoinGeckoResponse>() {
                data.prices
                    .get(coin_id)
                    .and_then(|prices| prices.get("usd"))
                    .copied()
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

pub fn get_market_prices() -> Result<(HashMap<String, f64>, HashMap<String, f64>, String)> {
    let stocks: HashMap<&str, &str> = [
        ("Apple Inc.", "AAPL"),
        ("NVIDIA Corporation", "NVDA"),
        ("Microsoft Corporation", "MSFT"),
        ("Amazon.com, Inc.", "AMZN"),
        ("Alphabet Inc.", "GOOGL"),
        ("Meta Platforms, Inc.", "META"),
        ("Tesla, Inc.", "TSLA"),
    ]
    .iter()
    .cloned()
    .collect();

    let cryptos: HashMap<&str, &str> = [
        ("Bitcoin (BTC)", "bitcoin"),
        ("Ethereum (ETH)", "ethereum"),
        ("Solana (SOL)", "solana"),
    ]
    .iter()
    .cloned()
    .collect();

    let current_time: DateTime<Local> = Local::now();
    let est_time = current_time.with_timezone(&Eastern);
    let timestamp = est_time.format("%Y-%m-%d %I:%M:%S %p %Z").to_string();

    let mut stock_prices = HashMap::new();
    for (name, ticker) in stocks {
        if let Some(price) = get_stock_price(ticker) {
            stock_prices.insert(name.to_string(), price);
        }
        thread::sleep(Duration::from_millis(500));
    }

    let mut crypto_prices = HashMap::new();
    for (name, coin_id) in cryptos {
        if let Some(price) = get_crypto_price(coin_id) {
            crypto_prices.insert(name.to_string(), price);
        }
        thread::sleep(Duration::from_millis(500));
    }

    Ok((stock_prices, crypto_prices, timestamp))
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

    #[test]
    fn test_live_market_data() {
        if let Ok((stocks, cryptos, timestamp)) = get_market_prices() {
            let formatted = format_prices(stocks, cryptos, timestamp);
            println!("Live Market Data:\n{}", formatted);
        }
    }
} 