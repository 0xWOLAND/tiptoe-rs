# Tiptoe-rs

A Private Information Retrieval (PIR) implementation in Rust that supports both local and networked operation modes.

## Setup

Start both servers in separate terminals:

```bash
# Terminal 1 - Encoding Server
cargo run --bin encoding_server --release

# Terminal 2 - Embedding Server
cargo run --bin embedding_server --release
```

The encoding server runs on port 3000 and the embedding server on port 3001.

## Testing

To run all tests:
```bash
cargo test --release
```

To run a specific test with output:
```bash
cargo test --package tiptoe-rs --lib -- client::tests::test_remote_client --exact --nocapture
```