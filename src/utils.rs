use anyhow::Result;
use candle_core::Tensor;
use crate::embeddings::TextEmbedder;
use simplepir::Matrix;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write;
use ndarray::{Array2, s};

const SCALE_FACTOR: f32 = 1_00.0;

pub fn strings_to_embedding_matrix(texts: &[String], embedder: &TextEmbedder) -> Result<Matrix> {
    if texts.is_empty() {
        anyhow::bail!("No texts provided for embedding");
    }

    // Get embeddings for each text
    let mut embeddings = Vec::with_capacity(texts.len());
    for text in texts {
        let embedding = embedder.embed(text)?;
        let embedding_vec = embedding.to_vec2::<f32>()?[0].clone();
        embeddings.push(embedding_vec);
    }

    // Create matrix data
    let mut matrix_data = vec![vec![0u64; 384]; 384];
    
    // Scale and copy embeddings into matrix
    for (i, embedding) in embeddings.iter().enumerate() {
        for (j, &val) in embedding.iter().enumerate() {
            matrix_data[j][i] = (val * SCALE_FACTOR).round() as u64;
        }
    }

    Ok(Matrix::from_data(matrix_data))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strings_to_embedding_matrix() -> Result<()> {
        let embedder = TextEmbedder::new()?;
        let texts = vec![
            "This is the first text".to_string(),
            "This is the second text".to_string(),
            "And this is the third one".to_string(),
        ];

        let matrix = strings_to_embedding_matrix(&texts, &embedder)?;

        // Test matrix dimensions - should be 384x384
        assert_eq!(matrix.nrows(), 384, "Matrix should have 384 rows");
        assert_eq!(matrix.ncols(), 384, "Matrix should have 384 columns");

        // Test that non-zero values are within expected range
        for (i, row) in matrix.data.iter().enumerate() {
            for &value in row {
                if i < 3 { // Only first 3 rows should have non-zero values
                    assert!(value < (1u64 << 63), "Value should be less than 2^63");
                } else {
                    assert_eq!(value, 0, "Padding rows should be zero");
                }
            }
        }

        // Test that similar texts have similar embeddings
        let similar_texts = vec![
            "I love programming".to_string(),
            "I enjoy coding".to_string(),
        ];
        let similar_matrix = strings_to_embedding_matrix(&similar_texts, &embedder)?;

        let different_texts = vec![
            "I love programming".to_string(),
            "The weather is nice today".to_string(),
        ];
        let different_matrix = strings_to_embedding_matrix(&different_texts, &embedder)?;

        // Calculate simple dot product similarity (only for the first two rows since those contain the actual embeddings)
        let similar_dot_product: u64 = similar_matrix.data[0]
            .iter()
            .zip(&similar_matrix.data[1])
            .map(|(&a, &b)| a.wrapping_mul(b))
            .sum();

        let different_dot_product: u64 = different_matrix.data[0]
            .iter()
            .zip(&different_matrix.data[1])
            .map(|(&a, &b)| a.wrapping_mul(b))
            .sum();

        // Similar texts should have a higher dot product
        assert!(
            similar_dot_product > different_dot_product,
            "Similar texts should have more similar embeddings"
        );

        Ok(())
    }

    #[test]
    fn test_empty_input() {
        let embedder = TextEmbedder::new().unwrap();
        let texts: Vec<String> = vec![];

        let result = strings_to_embedding_matrix(&texts, &embedder);
        assert!(result.is_err(), "Empty input should return an error");
    }

    #[test]
    fn test_padding() -> Result<()> {
        let embedder = TextEmbedder::new()?;
        let texts = vec!["Single text".to_string()];

        let matrix = strings_to_embedding_matrix(&texts, &embedder)?;

        // Check dimensions
        assert_eq!(matrix.nrows(), 384, "Matrix should be padded to 384 rows");
        assert_eq!(matrix.ncols(), 384, "Matrix should have 384 columns");

        // Check that padding rows are all zeros
        for row in matrix.data.iter().skip(1) {
            assert!(row.iter().all(|&x| x == 0), "Padding rows should be all zeros");
        }

        Ok(())
    }

    #[test]
    fn test_minimal_example() -> Result<()> {
        let embedder = TextEmbedder::new()?;
        
        // Get individual embeddings
        let cat_embedding = embedder.embed("cat")?;
        let dog_embedding = embedder.embed("dog")?;

        // Convert embeddings to u64 values
        let cat_raw: Vec<f32> = cat_embedding.flatten_all()?.to_vec1()?;
        let dog_raw: Vec<f32> = dog_embedding.flatten_all()?.to_vec1()?;

        // Convert to u64 using same scaling as in the matrix function
        let cat_u64: Vec<u64> = cat_raw.iter()
            .map(|&x| {
                let quantized = (x * 1_000_000.0).round() / 1_000_000.0;
                let scaled = (quantized + 1.0) * ((1u64 << 62) as f32);
                scaled.round() as u64
            })
            .collect();

        let dog_u64: Vec<u64> = dog_raw.iter()
            .map(|&x| {
                let quantized = (x * 1_000_000.0).round() / 1_000_000.0;
                let scaled = (quantized + 1.0) * ((1u64 << 62) as f32);
                scaled.round() as u64
            })
            .collect();

        println!("\nFirst 20 values of 'cat' embedding after u64 conversion:");
        for (i, &val) in cat_u64.iter().take(20).enumerate() {
            print!("{} ", val);
            if (i + 1) % 4 == 0 { println!(); }
        }

        println!("\nFirst 20 values of 'dog' embedding after u64 conversion:");
        for (i, &val) in dog_u64.iter().take(20).enumerate() {
            print!("{} ", val);
            if (i + 1) % 4 == 0 { println!(); }
        }

        // Now create the matrix and verify the values match
        let texts = vec![
            "cat".to_string(),
            "dog".to_string(),
        ];
        let matrix = strings_to_embedding_matrix(&texts, &embedder)?;

        println!("Matrix data:");
        for (i, row) in matrix.data.iter().enumerate().take(2) {
            println!("First 20 values of row {}:", i);
            for (j, &val) in row.iter().take(20).enumerate() {
                print!("{} ", val);
                if (j + 1) % 4 == 0 { println!(); }
            }
            println!();
        }

        // Verify first 20 values match
        for i in 0..20 {
            assert_eq!(matrix.data[0][i], cat_u64[i], "Cat embedding mismatch at position {}", i);
            assert_eq!(matrix.data[1][i], dog_u64[i], "Dog embedding mismatch at position {}", i);
        }

        Ok(())
    }
}
